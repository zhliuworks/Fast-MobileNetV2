#include "../../kernels/Gemm.cuh"

/* row-major layout */
// #define matA(i, j)     matA[(i) * K + (j)]
// #define matB(i, j)     matB[(i) * N + (j)]
// #define matC(i, j)     matC[(i) * N + (j)]
// #define matC_ref(i, j) matC_ref[(i) * N + (j)]

/* col-major layout */
#define matA(i, j)     matA[(j) * M + (i)]
#define matB(i, j)     matB[(j) * K + (i)]
#define matC(i, j)     matC[(j) * M + (i)]
#define matC_ref(i, j) matC_ref[(j) * M + (i)]

__host__ int main() {
    int NUM_THREADS = 32;
    int M = 2035, N = 516, K = 1028;
    float *matA = (float*)malloc(M * K * sizeof(float));
    float *matB = (float*)malloc(K * N * sizeof(float));
    float *matC = (float*)malloc(M * N * sizeof(float));
    float *matC_ref = (float*)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            matA(i, j) = (i + j) / 64.0f;
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            matB(i, j) = (i + j) / 128.0f;
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matC(i, j) = - (i + j) / 256.0f;
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            register float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += matA(i, k) * matB(k, j);
            }
            matC_ref(i, j) = sum + matC(i, j);
        }
    }

    fastGemmWrapper(matA, matB, matC, M, N, K, NUM_THREADS);

    // function test
    for (int i = 0; i < M * N; i++) {
        assert(matC[i] == matC_ref[i]);
    }

    // performance test was done in the choosing of kernels ...

    free(matA); free(matB); free(matC); free(matC_ref);
    return 0;
}