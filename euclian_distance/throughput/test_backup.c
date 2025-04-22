#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <stdint.h>
#include <mkl.h>
#include "common.h"
#include <linux/idxd.h>
#include <sys/mman.h>

int VECTOR_DIM;
int N_V;
int N_DSA = 1;
int WS = 128; // workqueue size
int BS = 1024; // Batch size

float euclidean_mkl(const float* x, const float* y, float* diff) {
    memcpy(diff, x, VECTOR_DIM * sizeof(float));
    cblas_saxpy(VECTOR_DIM, -1.0f, y, 1, diff, 1);
    return cblas_snrm2(VECTOR_DIM, diff, 1);
}

float euclidean_mkl_partial(const float* x, const float* y, const uint32_t* indices, int count, float* x_part, float* y_part) {
    for (int i = 0; i < count; i++) {
        x_part[i] = x[indices[i]];
        y_part[i] = y[indices[i]];
    }
    cblas_saxpy(count, -1.0f, y_part, 1, x_part, 1);
    return cblas_snrm2(count, x_part, 1);
}

void compute_all_distances(float** base, float** delta, float* out, float** diff_buffers) {
    #pragma omp parallel for
    for (int i = 0; i < N_V; i++) {
        out[i] = euclidean_mkl(base[i], delta[i], diff_buffers[omp_get_thread_num()]);
    }
}

void compute_partial_distances(float** base, float** delta, float* out,
                                uint32_t** indices, int* count,
                                float** x_part_buffers, float** y_part_buffers) {
    #pragma omp parallel for
    for (int i = 0; i < N_V; i++) {
        int tid = omp_get_thread_num();
        out[i] = euclidean_mkl_partial(base[i], delta[i], indices[i], count[i], x_part_buffers[tid], y_part_buffers[tid]);
    }
}

double get_elapsed_sec(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
}

int create_delta_records_using_dsa(void *wq_portal,
                                   struct dsa_hw_desc *desc_buf,
                                   struct dsa_completion_record *comp_buf,
                                   struct dsa_delta_record **dr_buf,
                                   float **base, float **delta,
                                   uint32_t **indices, int *count) {
    struct dsa_completion_record batch_comp __attribute__((aligned(32)));
    struct dsa_hw_desc batch_desc = { };
    batch_comp.status = 0;
    batch_desc.opcode = DSA_OPCODE_BATCH;
    batch_desc.flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
    batch_desc.desc_count = N_V;
    batch_desc.desc_list_addr = (uint64_t)&(desc_buf[0]);
    batch_desc.completion_addr = (uintptr_t)&batch_comp;

    _mm_sfence();
    movdir64b(wq_portal, &batch_desc);

    while (batch_comp.status == 0) { ; }

    if (batch_comp.status != 1) {
        printf("dsa offload failed: %x\n", batch_comp.status);
        return -1;
    }

    return 0;
}

int compute_diff_indices_from_delta_records(uint32_t **indices, int *count,
                                   struct dsa_completion_record *comp_buf,
                                   struct dsa_delta_record **dr_buf) {
    #pragma omp parallel for
    for (int i = 0; i < N_V; i++) {
        int max_idx = comp_buf[i].delta_rec_size / 10;
        for (int j = 0; j < max_idx; j++) {
            int idx = 2 * dr_buf[i][j].offset;
            indices[i][count[i]++] = idx;
            indices[i][count[i]++] = idx + 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        printf("Usage: %s <N_v> <dimension> <delta_rate> <num_repeat> <mode: full|partial>\n", argv[0]);
        return 1;
    }

    N_V = atoi(argv[1]);
    VECTOR_DIM = atoi(argv[2]);
    float delta_rate = atof(argv[3]);
    int num_repeat = atoi(argv[4]);
    int is_partial = strcmp(argv[5], "partial") == 0;

    srand(42);

    float*** base = malloc(num_repeat * sizeof(float**));
    float*** delta = malloc(num_repeat * sizeof(float**));
    for (int r = 0; r < num_repeat; r++) {
        base[r] = malloc(N_V * sizeof(float*));
        delta[r] = malloc(N_V * sizeof(float*));
        for (int i = 0; i < N_V; i++) {
            base[r][i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
            delta[r][i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
            for (int j = 0; j < VECTOR_DIM; j++) {
                float val = ((float)rand() / RAND_MAX);
                base[r][i][j] = val;
                delta[r][i][j] = val;
            }
            int num_delta = (int)(VECTOR_DIM * delta_rate);
            for (int d = 0; d < num_delta; d++) {
                delta[r][i][d] += 1.0f;
            }
        }
    }

    float* results = malloc(N_V * sizeof(float));
    memset(results, 0, N_V * sizeof(float));
    int n_threads = omp_get_max_threads();
    float** diff_buffers = malloc(n_threads * sizeof(float*));
    #pragma omp parallel for
    for (int i = 0; i < n_threads; i++) {
        diff_buffers[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
    }

    struct timespec t_start, t_end, t_1, t_2;

    if (!is_partial) {
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        for (int r = 0; r < num_repeat; r++) {
            compute_all_distances(base[r], delta[r], results, diff_buffers);
        }
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = get_elapsed_sec(t_start, t_end);
        double flops = 2.0 * VECTOR_DIM * N_V * num_repeat;
        double gflops = flops / elapsed / 1e9;
        printf("[FULL L2] Total Time: %.6f sec | Throughput: %.2f GFLOPS\n", elapsed, gflops);
    } else {
        float* results_part = malloc(N_V * sizeof(float));
        float** x_part_buffers = malloc(n_threads * sizeof(float*));
        float** y_part_buffers = malloc(n_threads * sizeof(float*));
        for (int i = 0; i < n_threads; i++) {
            x_part_buffers[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
            y_part_buffers[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
        }

        uint32_t** indices = malloc(N_V * sizeof(uint32_t*));
        int* count = malloc(N_V * sizeof(int));
        for (int i = 0; i < N_V; i++) {
            indices[i] = malloc(VECTOR_DIM * sizeof(uint32_t));
            memset(indices[i], 0, (VECTOR_DIM * sizeof(uint32_t)));
            count[i] = 0;
        }

        void *wq_portal = map_wq();
        struct dsa_hw_desc *desc_buf = aligned_alloc(64, N_V * sizeof(struct dsa_hw_desc));
        struct dsa_completion_record *comp_buf = aligned_alloc(32, N_V * sizeof(struct dsa_completion_record));
        struct dsa_delta_record **dr_buf = malloc(N_V * sizeof(struct dsa_delta_record *));
        for (int i = 0; i < N_V; i++) {
            dr_buf[i] = aligned_alloc(64, (VECTOR_DIM * sizeof(float) / 8 * 10));
            memset(dr_buf[i], 0, (VECTOR_DIM * sizeof(float) / 8 * 10));

        }
        memset(desc_buf, 0, N_V * sizeof(struct dsa_hw_desc));
        memset(comp_buf, 0, N_V * sizeof(struct dsa_completion_record));
        for (int i = 0; i < N_V; i++) {
            desc_buf[i].opcode = DSA_OPCODE_CR_DELTA;
            desc_buf[i].flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
            desc_buf[i].xfer_size = VECTOR_DIM * sizeof(float);
            desc_buf[i].completion_addr = (uintptr_t)&(comp_buf[i]);
            desc_buf[i].delta_addr = (uint64_t)(uintptr_t)dr_buf[i];
            desc_buf[i].max_delta_size = VECTOR_DIM * sizeof(float) / 8 * 10;
        }

        double total_dsa = 0.0, total_cpu = 0.0;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        for (int r = 0; r < num_repeat; r++) {
            for (int i = 0; i < N_V; i++) {
                count[i] = 0;
                comp_buf[i].status = 0;
                desc_buf[i].src_addr = (uintptr_t)base[r][i];
                desc_buf[i].src2_addr = (uintptr_t)delta[r][i];
            }

            clock_gettime(CLOCK_MONOTONIC, &t_1);
            create_delta_records_using_dsa(wq_portal, desc_buf, comp_buf, dr_buf,
                                           base[r], delta[r], indices, count);
            clock_gettime(CLOCK_MONOTONIC, &t_2);
            total_dsa += get_elapsed_sec(t_1, t_2);
	    compute_diff_indices_from_delta_records(indices, count, comp_buf, dr_buf);

            compute_partial_distances(base[r], delta[r], results_part,
                                      indices, count, x_part_buffers, y_part_buffers);
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = get_elapsed_sec(t_start, t_end);
        double flops = 2.0 * VECTOR_DIM * N_V * num_repeat;
        double gflops = flops / elapsed / 1e9;
	total_cpu = elapsed - total_dsa;
        printf("[PARTIAL L2] Total Time: %.6f sec | Throughput: %.2f GFLOPS\n", elapsed, gflops);
        printf("[PARTIAL L2] DSA Ratio: %.2f%% | CPU Ratio: %.2f%%\n",
               100.0 * total_dsa / elapsed, 100.0 * total_cpu / elapsed);
    }

    for (int r = 0; r < num_repeat; r++) {
        for (int i = 0; i < N_V; i++) {
            free(base[r][i]);
            free(delta[r][i]);
        }
        free(base[r]);
        free(delta[r]);
    }
    free(base);
    free(delta);
    for (int i = 0; i < n_threads; i++) {
        free(diff_buffers[i]);
    }
    free(diff_buffers);

    return 0;
}
