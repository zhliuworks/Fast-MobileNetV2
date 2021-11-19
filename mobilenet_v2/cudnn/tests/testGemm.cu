#include "../wrappers/Gemm.cuh"

__host__ int main() {
    int M = 2, N = 3, K = 4;
    float matA_h[M * K] = {1.0f, 3.0f, 2.0f, 5.0f,
                           2.0f, 4.0f, 1.0f, 3.0f};
    float matB_h[K * N] = {3.0f, 1.0f, 2.0f,
                           1.0f, 2.0f, 4.0f,
                           5.0f, 3.0f, 1.0f,
                           2.0f, 6.0f, 0.0f};
    float matC_h[M * N] = {1.0f, -1.0f, 2.0f,
                           0.0f, 1.0f, -2.0f};  // bias
    float matC_ans[M * N] = {27.0f, 42.0f, 18.0f,
                             21.0f, 32.0f, 19.0f};
    
    // heuristic
    cudnnGemmWrapper(matA_h, matB_h, matC_h, M, N, K, -1);
    for (int i = 0; i < M * N; i++) {
        if (matC_h[i] != matC_ans[i]) {
            fprintf(stderr, "idx: %d,\tgemm: %f,\tans: %f\n", i, matC_h[i], matC_ans[i]);
        }
    }

    return 0;
}