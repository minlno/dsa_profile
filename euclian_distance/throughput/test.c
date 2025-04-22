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

void submit_delta_creation(void *wq_portal, struct dsa_hw_desc *desc) {
    _mm_sfence();
    movdir64b(wq_portal, desc);
}

void wait_for_completion(struct dsa_completion_record *comp,
		struct dsa_completion_record *comp2) {
    while (comp->status == 0) { ; }
    if (comp->status != 1) {
        printf("DSA offload failed: %x, %x\n", comp->status, comp2->status);
	for (int i = 0; i < N_V; i++) {
		if (comp2[i].status != 1) {
			printf("error: %d, %x\n",i, comp2[i].status);
			break;
		}
	}
        exit(1);
    }
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

#define CHECK_ALLOC(ptr, name) \
    if (!(ptr)) { \
        fprintf(stderr, "Allocation failed: %s\n", name); \
        exit(EXIT_FAILURE); \
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

    float*** base = malloc(num_repeat * sizeof(float**)); CHECK_ALLOC(base, "base");
    float*** delta = malloc(num_repeat * sizeof(float**)); CHECK_ALLOC(delta, "delta");
    for (int r = 0; r < num_repeat; r++) {
        base[r] = malloc(N_V * sizeof(float*)); CHECK_ALLOC(base[r], "base[r]");
        delta[r] = malloc(N_V * sizeof(float*)); CHECK_ALLOC(delta[r], "delta[r]");
        for (int i = 0; i < N_V; i++) {
            base[r][i] = aligned_alloc(64, VECTOR_DIM * sizeof(float)); CHECK_ALLOC(base[r][i], "base[r][i]");
            delta[r][i] = aligned_alloc(64, VECTOR_DIM * sizeof(float)); CHECK_ALLOC(delta[r][i], "delta[r][i]");
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
    CHECK_ALLOC(results, "results");
    memset(results, 0, N_V * sizeof(float));
    int n_threads = omp_get_max_threads();
    float** diff_buffers = malloc(n_threads * sizeof(float*)); 
    CHECK_ALLOC(diff_buffers, "diff_buffers");
    for (int i = 0; i < n_threads; i++) {
        diff_buffers[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float)); 
	CHECK_ALLOC(diff_buffers[i], "diff_buffers[i]");
    }

    if (is_partial) {
        float* results_part = malloc(N_V * sizeof(float)); 
	CHECK_ALLOC(results_part, "results_part");
        float** x_part_buffers = malloc(n_threads * sizeof(float*));
	CHECK_ALLOC(x_part_buffers, "x_part_buffers");
        float** y_part_buffers = malloc(n_threads * sizeof(float*)); 
	CHECK_ALLOC(y_part_buffers, "y_part_buffers");
        for (int i = 0; i < n_threads; i++) {
            x_part_buffers[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float)); 
	    CHECK_ALLOC(x_part_buffers[i], "x_part_buffers[i]");
            y_part_buffers[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float)); 
	    CHECK_ALLOC(y_part_buffers[i], "y_part_buffers[i]");
        }

        uint32_t*** indices = malloc(num_repeat * sizeof(uint32_t**)); 
	CHECK_ALLOC(indices, "indices");
        int** count = malloc(num_repeat * sizeof(int*)); 
	CHECK_ALLOC(count, "count");
        struct dsa_delta_record ***dr_buf = malloc(num_repeat * sizeof(struct dsa_delta_record **));
	CHECK_ALLOC(dr_buf, "dr_buf");
        struct dsa_hw_desc **desc_buf = malloc(num_repeat * sizeof(struct dsa_hw_desc*)); 
	CHECK_ALLOC(desc_buf, "desc_buf");
        struct dsa_completion_record **comp_buf = malloc(num_repeat * sizeof(struct dsa_completion_record*)); 
	CHECK_ALLOC(comp_buf, "comp_buf");
        struct dsa_hw_desc *batch_desc_array = malloc(num_repeat * sizeof(struct dsa_hw_desc)); 
	CHECK_ALLOC(batch_desc_array, "batch_desc_array");
        struct dsa_completion_record *batch_comp_array = aligned_alloc(32, num_repeat * sizeof(struct dsa_completion_record)); 
	CHECK_ALLOC(batch_comp_array, "batch_comp_array");

        for (int r = 0; r < num_repeat; r++) {
            indices[r] = malloc(N_V * sizeof(uint32_t*)); 
	    CHECK_ALLOC(indices[r], "indices[r]");
            count[r] = malloc(N_V * sizeof(int)); 
	    CHECK_ALLOC(count[r], "count[r]");
            dr_buf[r] = malloc(N_V * sizeof(struct dsa_delta_record*)); 
	    CHECK_ALLOC(dr_buf[r], "dr_buf[r]");
            desc_buf[r] = aligned_alloc(64, N_V * sizeof(struct dsa_hw_desc)); 
	    CHECK_ALLOC(desc_buf[r], "desc_buf[r]");
            comp_buf[r] = aligned_alloc(32, N_V * sizeof(struct dsa_completion_record)); 
	    CHECK_ALLOC(comp_buf[r], "comp_buf[r]");

            for (int i = 0; i < N_V; i++) {
                indices[r][i] = malloc(VECTOR_DIM * sizeof(uint32_t)); 
		CHECK_ALLOC(indices[r][i], "indices[r][i]");
                dr_buf[r][i] = aligned_alloc(64, (VECTOR_DIM * sizeof(float) / 8 * 10)); 
		CHECK_ALLOC(dr_buf[r][i], "dr_buf[r][i]");
                memset(dr_buf[r][i], 0, (VECTOR_DIM * sizeof(float) / 8 * 10));
                memset(&desc_buf[r][i], 0, sizeof(struct dsa_hw_desc));
                memset(&comp_buf[r][i], 0, sizeof(struct dsa_completion_record));
                desc_buf[r][i].opcode = DSA_OPCODE_CR_DELTA;
                desc_buf[r][i].flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
                desc_buf[r][i].xfer_size = VECTOR_DIM * sizeof(float);
                desc_buf[r][i].completion_addr = (uintptr_t)&(comp_buf[r][i]);
                desc_buf[r][i].delta_addr = (uint64_t)(uintptr_t)dr_buf[r][i];
                desc_buf[r][i].max_delta_size = VECTOR_DIM * sizeof(float) / 8 * 10;
            }

            memset(&batch_desc_array[r], 0, sizeof(struct dsa_hw_desc));
            memset(&batch_comp_array[r], 0, sizeof(struct dsa_completion_record));
            batch_desc_array[r].opcode = DSA_OPCODE_BATCH;
            batch_desc_array[r].flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
            batch_desc_array[r].desc_count = N_V;
            batch_desc_array[r].desc_list_addr = (uint64_t)&(desc_buf[r][0]);
            batch_desc_array[r].completion_addr = (uintptr_t)&(batch_comp_array[r]);
        }

        void *wq_portal = map_wq(0);
	void *wq_portal2 = map_wq(1);

        struct timespec t_start, t_end, t_1, t_2;
        double total_dsa = 0.0, total_cpu = 0.0;

        clock_gettime(CLOCK_MONOTONIC, &t_start);

        clock_gettime(CLOCK_MONOTONIC, &t_1);
        for (int i = 0; i < N_V; i++) {
            desc_buf[0][i].src_addr = (uintptr_t)base[0][i];
            desc_buf[0][i].src2_addr = (uintptr_t)delta[0][i];
        }
	submit_delta_creation(wq_portal, &batch_desc_array[0]);
        clock_gettime(CLOCK_MONOTONIC, &t_2);
        total_dsa += get_elapsed_sec(t_1, t_2);

        for (int r = 0; r < num_repeat; r++) {
            clock_gettime(CLOCK_MONOTONIC, &t_1);
            if (r + 1 < num_repeat) {
                for (int i = 0; i < N_V; i++) {
                    desc_buf[r+1][i].src_addr = (uintptr_t)base[r+1][i];
                    desc_buf[r+1][i].src2_addr = (uintptr_t)delta[r+1][i];
                }
                clock_gettime(CLOCK_MONOTONIC, &t_1);
		submit_delta_creation(wq_portal, &batch_desc_array[r+1]);
            }

	    wait_for_completion(&batch_comp_array[r], comp_buf[r]);
            clock_gettime(CLOCK_MONOTONIC, &t_2);
            total_dsa += get_elapsed_sec(t_1, t_2);

            compute_diff_indices_from_delta_records(indices[r], count[r], comp_buf[r], dr_buf[r]);
            compute_partial_distances(base[r], delta[r], results_part, indices[r], count[r], x_part_buffers, y_part_buffers);
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = get_elapsed_sec(t_start, t_end);
	total_cpu = elapsed - total_dsa;
        double flops = 2.0 * VECTOR_DIM * N_V * num_repeat;
        double gflops = flops / elapsed / 1e9;

        printf("[PARTIAL L2 PIPELINED] Total Time: %.6f sec | Throughput: %.2f GFLOPS\n", elapsed, gflops);
        printf("[PARTIAL L2 PIPELINED] DSA Ratio: %.2f%% | CPU Ratio: %.2f%%\n", 100.0 * total_dsa / elapsed, 100.0 * total_cpu / elapsed);
    } else {
        struct timespec t_start, t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        for (int r = 0; r < num_repeat; r++) {
            compute_all_distances(base[r], delta[r], results, diff_buffers);
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = get_elapsed_sec(t_start, t_end);
        double flops = 2.0 * VECTOR_DIM * N_V * num_repeat;
        double gflops = flops / elapsed / 1e9;

        printf("[FULL L2 MKL] Total Time: %.6f sec | Throughput: %.2f GFLOPS\n", elapsed, gflops);
    }

    return 0;
}
