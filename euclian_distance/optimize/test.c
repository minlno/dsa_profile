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

float euclidean_mkl(const float* x, const float* y, float* diff) {
    memcpy(diff, x, VECTOR_DIM * sizeof(float));
    cblas_saxpy(VECTOR_DIM, -1.0f, y, 1, diff, 1);
    return cblas_snrm2(VECTOR_DIM, diff, 1);
}

float euclidean_mkl_partial(const float* x, const float* y, const uint32_t* indices, int count, float* x_part, float* y_part) {
	//printf("count: %d\n", count);
    for (int i = 0; i < count; i++) {
        x_part[i] = x[indices[i]];
        y_part[i] = y[indices[i]];
    }
    memcpy(x_part, x_part, count * sizeof(float));
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

int compute_diff_indices_using_dsa(void *wq_portal,
                                   struct dsa_hw_desc *desc_buf,
                                   struct dsa_completion_record *comp_buf,
                                   struct dsa_delta_record **dr_buf,
                                   float **base, float **delta,
                                   uint32_t **indices, int *count,
                                   double *dsa_time, double *delta_idx_time) {

    struct timespec t_start, t_end;
    if (N_V > 1024) {
        printf("N_V should be smaller than 1024 (maximum batch size)\n");
        return -1;
    }

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

    clock_gettime(CLOCK_MONOTONIC, &t_start);
    while (batch_comp.status == 0) { ; }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    *dsa_time = get_elapsed_sec(t_start, t_end);

    if (batch_comp.status != 1) {
        printf("dsa offload failed: %x\n", batch_comp.status);
        for (int i = 0; i < N_V; i++) {
            if (comp_buf[i].status != 1) {
                printf("dsa offload failed: %x\n", comp_buf[i].status);
                break;
            }
        }
        return -1;
    }

    printf("max_idx: %d\n", comp_buf[0].delta_rec_size/10);
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    #pragma omp parallel for
    for (int i = 0; i < N_V; i++) {
        int max_idx = comp_buf[i].delta_rec_size / 10;
        for (int j = 0; j < max_idx; j++) {
            int idx = 2 * dr_buf[i][j].offset;
            indices[i][count[i]++] = idx;
            indices[i][count[i]++] = idx + 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    *delta_idx_time = get_elapsed_sec(t_start, t_end);

    return 0;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <N_v> <dimension> <delta_rate>\n", argv[0]);
        return 1;
    }
    struct timespec t_start, t_end;
    double dsa_time, delta_idx_time;

    N_V = atoi(argv[1]);
    VECTOR_DIM = atoi(argv[2]);
    float delta_rate = atof(argv[3]);

    srand(42);

    float** base = malloc(N_V * sizeof(float*));
    float** delta = malloc(N_V * sizeof(float*));
    float** base_p = malloc(N_V * sizeof(float*));
    float** delta_p = malloc(N_V * sizeof(float*));
    float* results_full = malloc(N_V * sizeof(float));
    float* results_partial = malloc(N_V * sizeof(float));

    for (int i = 0; i < N_V; i++) {
        base[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
        delta[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
        base_p[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
        delta_p[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
        for (int j = 0; j < VECTOR_DIM; j++) {
	    base[i][j] = ((float)rand() / RAND_MAX);
	    base_p[i][j] = base[i][j];
	}
        memcpy(delta[i], base[i], VECTOR_DIM * sizeof(float));
        memcpy(delta_p[i], base_p[i], VECTOR_DIM * sizeof(float));
        int num_delta = (int)(VECTOR_DIM * delta_rate);
        for (int d = 0; d < num_delta; d++) {
            delta[i][d] += 1.0f;
            delta_p[i][d] += 1.0f;
        }
    }

    uint32_t** indices = malloc(N_V * sizeof(uint32_t*));
    int* count = malloc(N_V * sizeof(int));

    for (int i = 0; i < N_V; i++) {
        indices[i] = malloc(VECTOR_DIM * sizeof(uint32_t));
        count[i] = 0;
    }

    void *wq_portal = map_wq();
    if (wq_portal == MAP_FAILED) {
        printf("DSA WQ MAP FAILED\n");
        return 1;
    }

    struct dsa_hw_desc *desc_buf = aligned_alloc(64, N_V * sizeof(struct dsa_hw_desc));
    struct dsa_completion_record *comp_buf = aligned_alloc(32, N_V * sizeof(struct dsa_completion_record));
    struct dsa_delta_record **dr_buf = malloc(N_V * sizeof(struct dsa_delta_record *));

    memset(desc_buf, 0, N_V * sizeof(struct dsa_hw_desc));
    memset(comp_buf, 0, N_V * sizeof(struct dsa_completion_record));
    for (int i = 0; i < N_V; i++) {
        dr_buf[i] = aligned_alloc(64, (VECTOR_DIM * sizeof(float) / 8 * 10));
        memset(dr_buf[i], 0, VECTOR_DIM * sizeof(float) / 8 * 10);
        comp_buf[i].status = 0;
        desc_buf[i].opcode = DSA_OPCODE_CR_DELTA;
        desc_buf[i].flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
        desc_buf[i].xfer_size = VECTOR_DIM * sizeof(float);
        desc_buf[i].src_addr = (uintptr_t)base[i];
        desc_buf[i].src2_addr = (uintptr_t)delta[i];
        desc_buf[i].completion_addr = (uintptr_t)&(comp_buf[i]);
        desc_buf[i].delta_addr = (uint64_t)(uintptr_t)dr_buf[i];
        desc_buf[i].max_delta_size = VECTOR_DIM * sizeof(float) / 8 * 10;
    }

    int n_threads = omp_get_max_threads();
    float** diff_buffers = malloc(n_threads * sizeof(float*));
    float** x_part_buffers = malloc(n_threads * sizeof(float*));
    float** y_part_buffers = malloc(n_threads * sizeof(float*));

    for (int i = 0; i < n_threads; i++) {
        diff_buffers[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
        x_part_buffers[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
        y_part_buffers[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
    }

    for (int i = 0; i < N_V; i++) {
	cflush((char*)base[i], VECTOR_DIM * sizeof(float));
	cflush((char*)delta[i], VECTOR_DIM * sizeof(float));
    }
    cpuid();
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    compute_all_distances(base, delta, results_full, diff_buffers);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    printf("[FULL L2] Time: %.6f sec\n", get_elapsed_sec(t_start, t_end));


    
    for (int i = 0; i < N_V; i++) {
	cflush((char*)base_p[i], VECTOR_DIM * sizeof(float));
	cflush((char*)delta_p[i], VECTOR_DIM * sizeof(float));
    }
    cpuid();
    compute_diff_indices_using_dsa(wq_portal, desc_buf, comp_buf, dr_buf, base, delta, indices, count, &dsa_time, &delta_idx_time);

    clock_gettime(CLOCK_MONOTONIC, &t_start);
    compute_partial_distances(base_p, delta_p, results_partial, indices, count, x_part_buffers, y_part_buffers);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    printf("[PARTIAL L2 with DSA] Time: %.6f sec\n", get_elapsed_sec(t_start, t_end));
    printf("[DSA Offload] Time: %.6f sec\n", dsa_time);
    printf("[Delta Index Extraction] Time: %.6f sec\n", delta_idx_time);
    printf("[PARTIAL L2 TOTAL] Time: %.6f sec\n", delta_idx_time + dsa_time + get_elapsed_sec(t_start, t_end));

    for (int i = 0; i < N_V; i++) {
	cflush((char*)base[i], VECTOR_DIM * sizeof(float));
	cflush((char*)delta[i], VECTOR_DIM * sizeof(float));
    }
    cpuid();
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    compute_all_distances(base, delta, results_full, diff_buffers);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    printf("[FULL L2] Time: %.6f sec\n", get_elapsed_sec(t_start, t_end));

    for (int i = 0; i < 5 && i < N_V; i++) {
        printf("dist_full[%d] = %.4f | dist_partial[%d] = %.4f\n", i, results_full[i], i, results_partial[i]);
    }

    for (int i = 0; i < N_V; i++) {
        free(base[i]);
        free(delta[i]);
        free(base_p[i]);
        free(delta_p[i]);
        free(indices[i]);
        free(dr_buf[i]);
    }
    free(base);
    free(delta);
    free(base_p);
    free(delta_p);
    free(results_full);
    free(results_partial);
    free(indices);
    free(count);
    free(desc_buf);
    free(comp_buf);
    free(dr_buf);
    munmap(wq_portal, WQ_PORTAL_SIZE);

    for (int i = 0; i < n_threads; i++) {
        free(diff_buffers[i]);
        free(x_part_buffers[i]);
        free(y_part_buffers[i]);
    }
    free(diff_buffers);
    free(x_part_buffers);
    free(y_part_buffers);

    return 0;
}

