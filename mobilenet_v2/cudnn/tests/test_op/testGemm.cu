#include "../../wrappers/Gemm.cuh"

__host__ int main() {
    int M = 2, N = 3, K = 4;
    float matA[M * K] = {1.0f, 3.0f, 2.0f, 5.0f,
                           2.0f, 4.0f, 1.0f, 3.0f};
    float matB[K * N] = {3.0f, 1.0f, 2.0f,
                           1.0f, 2.0f, 4.0f,
                           5.0f, 3.0f, 1.0f,
                           2.0f, 6.0f, 0.0f};
    float matC[M * N] = {1.0f, -1.0f, 2.0f,
                           0.0f, 1.0f, -2.0f};  // bias
    float matC_ref[M * N] = {27.0f, 42.0f, 18.0f,
                             21.0f, 32.0f, 19.0f};
    
    cudnnGemmWrapper(matA, matB, matC, M, N, K, -1); // heuristic
    for (int i = 0; i < M * N; i++) {
        assert(matC[i] == matC_ref[i]);
    }

    return 0;
}