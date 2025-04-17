#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include "common.h"
#include <linux/idxd.h>

int VECTOR_DIM;
int N_V;

// AVX512 전체 계산 (기존 버전)
float euclidean_avx512(const float* x, const float* y) {
    __m512 diff, sqr;
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < VECTOR_DIM; i += 16) {
        __m512 a = _mm512_loadu_ps(x + i);
        __m512 b = _mm512_loadu_ps(y + i);
        diff = _mm512_sub_ps(a, b);
        sqr = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, sqr);
    }
    float buf[16];
    _mm512_storeu_ps(buf, sum);
    float final_sum = 0.0f;
    for (int i = 0; i < 16; i++) final_sum += buf[i];
    return sqrtf(final_sum);
}

// AVX512 기반 부분 계산 (diff 인덱스만 사용)
float euclidean_avx512_partial(const float* x, const float* y, const uint32_t* indices, int count) {
    float sum = 0.0f;
    int i = 0;

    for (; i <= count - 16; i += 16) {
        __m512 a = _mm512_i32gather_ps(_mm512_loadu_si512(&indices[i]), x, 4);
        __m512 b = _mm512_i32gather_ps(_mm512_loadu_si512(&indices[i]), y, 4);
        __m512 diff = _mm512_sub_ps(a, b);
        __m512 sqr = _mm512_mul_ps(diff, diff);
        float buf[16];
        _mm512_storeu_ps(buf, sqr);
        for (int j = 0; j < 16; ++j) sum += buf[j];
    }

    for (; i < count; i++) {
        float d = x[indices[i]] - y[indices[i]];
        sum += d * d;
    }

    return sqrtf(sum);
}

// 전체 거리 계산 (전체 계산 방식)
void compute_all_distances(float** base, float** delta, float* out) {
    #pragma omp parallel for
    for (int i = 0; i < N_V; i++) {
        out[i] = euclidean_avx512(base[i], delta[i]);
    }
}

// DSA delta 인덱스 생성 시뮬레이션
int simulate_diff_indices(const float* x, const float* y, uint32_t* indices) {
    int count = 0;
    for (int i = 0; i < VECTOR_DIM; i++) {
        if (x[i] != y[i]) {
            indices[count++] = i;
        }
    }
    return count;
}

// 전체 쌍에 대해 partial L2 계산
void compute_partial_distances(float** base, float** delta, float* out, 
		uint32_t **indices, int *count) {
    #pragma omp parallel for
    for (int i = 0; i < N_V; i++) {
        out[i] = euclidean_avx512_partial(base[i], delta[i], indices[i], count[i]);
    }
}

// 난수 초기화
void random_fill(float* vec, int dim) {
    for (int i = 0; i < dim; i++) vec[i] = ((float)rand() / RAND_MAX);
}

// base 기준으로 일부 delta 적용
void apply_delta(float* dst, const float* src, float delta_rate) {
    memcpy(dst, src, sizeof(float) * VECTOR_DIM);
    int num_delta = (int)(VECTOR_DIM * delta_rate);
    for (int i = 0; i < num_delta; i++) {
        int idx = rand() % VECTOR_DIM;
        dst[idx] += 1.0f;
    }
}

// 시간 계산
double get_elapsed_sec(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
}

int compute_diff_indices_using_dsa(void *wq_portal, 
		struct dsa_hw_desc *desc_buf, 
		struct dsa_completion_record *comp_buf, 
		struct dsa_delta_record **dr_buf, 
		float **base, float **delta, uint32_t **indices, int *count,
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
	while (batch_comp.status == 0) {
		;
	}
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

   	clock_gettime(CLOCK_MONOTONIC, &t_start);
	// indices, count 채우기.
    	#pragma omp parallel for
	for (int i = 0; i < N_V; i++) {
		int max_idx = comp_buf[i].delta_rec_size / 10;
		for (int j = 0; j < max_idx; j++) {
			int idx = 2 * dr_buf[i][j].offset; // offset 은 8B당 1. float array는 4B당 1.
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

    N_V = atoi(argv[1]);
    VECTOR_DIM = atoi(argv[2]);
    float delta_rate = atof(argv[3]);

    srand(42);

    float** base = malloc(N_V * sizeof(float*));
    float** delta = malloc(N_V * sizeof(float*));
    float* results_full = malloc(N_V * sizeof(float));
    float* results_partial = malloc(N_V * sizeof(float));

    for (int i = 0; i < N_V; i++) {
        base[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
        delta[i] = aligned_alloc(64, VECTOR_DIM * sizeof(float));
        random_fill(base[i], VECTOR_DIM);
        apply_delta(delta[i], base[i], delta_rate);
    }

    uint32_t **indices = malloc(N_V * sizeof(uint32_t*));
    int *count = malloc(N_V * sizeof(int));

    for (int i = 0; i < N_V; i++) {
	    indices[i] = malloc(VECTOR_DIM * sizeof(uint32_t));
    }

    void *wq_portal;
    struct dsa_hw_desc *desc_buf;
    struct dsa_completion_record *comp_buf;
    struct dsa_delta_record **dr_buf;

    desc_buf = (struct dsa_hw_desc*)aligned_alloc(64, N_V * sizeof(struct dsa_hw_desc));
    comp_buf = (struct dsa_completion_record*)aligned_alloc(32, N_V * sizeof(struct dsa_completion_record));
    dr_buf = malloc(N_V * sizeof(struct dsa_delta_record *));
    for (int i = 0; i < N_V; i++)
    	dr_buf[i] = (struct dsa_delta_record*)aligned_alloc(64, (VECTOR_DIM * sizeof(float) / 8 * 10));

    memset(desc_buf, 0, N_V * sizeof(struct dsa_hw_desc));
    memset(comp_buf, 0, N_V * sizeof(struct dsa_completion_record));
    for (int i = 0; i < N_V; i++) {
	        memset(dr_buf[i], 0, VECTOR_DIM * sizeof(float) / 8 * 10);
    }

    wq_portal = map_wq();
    if (wq_portal == MAP_FAILED) {
	    printf("DSA WQ MAP FAILED\n");
	    return 1;
    }
	for (int i = 0; i < N_V; i++) {
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

    // 전체 계산 시간
    struct timespec t_start, t_end;
    struct timespec t_start1, t_end1;
    int err;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    compute_all_distances(base, delta, results_full);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double full_time = get_elapsed_sec(t_start, t_end);
    printf("[FULL L2] Time: %.6f sec\n", full_time);

    // partial 계산 시간
    double partial_dsa_time, partial_diff_time, partial_delta_time;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    /*
    for (int i = 0; i < N_V; ++i) {
        count[i] = simulate_diff_indices(base[i], delta[i], indices[i]);
    }
    */
    clock_gettime(CLOCK_MONOTONIC, &t_start1);
    err = compute_diff_indices_using_dsa(wq_portal, desc_buf, comp_buf, dr_buf, base, delta, indices, count, &partial_dsa_time, &partial_delta_time);
    clock_gettime(CLOCK_MONOTONIC, &t_end1);
    if (err < 0)
	    goto out;

    compute_partial_distances(base, delta, results_partial, indices, count);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double partial_time = get_elapsed_sec(t_start, t_end);
    partial_diff_time = get_elapsed_sec(t_start1, t_end1);
    printf("[PARTIAL L2 with DSA diff] Total Time: %.6f sec\n", partial_time);
    printf("[PARTIAL L2 with DSA diff] --DIFF Time: %.6f sec\n", partial_diff_time);
    printf("[PARTIAL L2 with DSA diff] ----CR_DELTA (DSA) Time: %.6f sec\n", partial_dsa_time);
    printf("[PARTIAL L2 with DSA diff] ----DELTA DIFF (CPU) Time: %.6f sec\n", partial_delta_time);
    printf("[PARTIAL L2 with DSA diff] --L2 CALC (CPU) Time: %.6f sec\n", partial_time - partial_diff_time);

    for (int i = 0; i < 5 && i < N_V; i++) {
        printf("dist_full[%d] = %.4f | dist_partial[%d] = %.4f\n",
               i, results_full[i], i, results_partial[i]);
    }
out:

    for (int i = 0; i < N_V; i++) {
        free(base[i]);
        free(delta[i]);
	free(indices[i]);
    }
    free(base); free(delta);
    free(results_full); free(results_partial);
    free(indices); free(count);

    munmap(wq_portal, WQ_PORTAL_SIZE);
    for (int i = 0; i < N_V; i++) {
	    free(dr_buf[i]);
    }
    free(desc_buf); free(comp_buf); free(dr_buf);

    return 0;
}

