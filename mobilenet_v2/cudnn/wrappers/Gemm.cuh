#pragma once
#include "utils.cuh"

/* Gemm Forward with cuDNN: Fusion of Gemm + Bias operators */
__host__ void cudnnGemmWrapper(
    float   *matA_h,    // matrix A: [M * K]
    float   *matB_h,    // matrix B: [K * N]
    float   *matC_h,    // matrix C: [M * N] (initialized with bias)
    int      M,
    int      N,
    int      K,
    int      algorithm  // -1 -> apply heuristics, [0-23] -> specific
) {

    // allocate device memory
    float *matA_d;
    float *matB_d;
    float *matC_d;

    CUDA_CHECK(cudaMalloc(
        (void**)&matA_d, M * K * sizeof(float)
    ));
    CUDA_CHECK(cudaMalloc(
        (void**)&matB_d, K * N * sizeof(float)
    ));
    CUDA_CHECK(cudaMalloc(
        (void**)&matC_d, M * N * sizeof(float)
    ));

    // initialize device matrices with host matrices
    CUBLAS_CHECK(cublasSetVector(
        M * K, sizeof(float), matA_h, 1, matA_d, 1
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUBLAS_CHECK(cublasSetVector(
        K * N, sizeof(float), matB_h, 1, matB_d, 1
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUBLAS_CHECK(cublasSetVector(
        M * N, sizeof(float), matC_h, 1, matC_d, 1
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // algorithm
    cublasGemmAlgo_t gemm_algo;
    if (algorithm == -1 || (algorithm >= 0 && algorithm <= 23)) {
        gemm_algo = static_cast<cublasGemmAlgo_t>(algorithm);
    } else {
        fprintf(stderr, "algorithm argument invalid\n");
        exit(EXIT_FAILURE);
    }

    // perform GEMM
    cublasHandle_t ctx;
    CUBLAS_CHECK(cublasCreate(&ctx));
    float alpha = 1.0f;
    float beta = 1.0f;

    // `C = α * op(A) * op(B) + β * C`
    // Note that cuBLAS interprets matrices as column-major ordered,
    // so we will trick cuBLAS into computing C^T = (A*B)^T = B^T*A^T
    CUBLAS_CHECK(cublasGemmEx(
        ctx,
        CUBLAS_OP_N,    // op(B) = B
        CUBLAS_OP_N,    // op(A) = A
        N, M, K,
        &alpha,
        matB_d, CUDA_R_32F, N,
        matA_d, CUDA_R_32F, K,
        &beta,
        matC_d, CUDA_R_32F, N,
        CUDA_R_32F,
        gemm_algo
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    // upload result to host
    CUBLAS_CHECK(cublasGetVector(
        M * N, sizeof(float), matC_d, 1, matC_h, 1
    ));

    // finalize
    CUDA_CHECK(cudaFree(matA_d));
    CUDA_CHECK(cudaFree(matB_d));
    CUDA_CHECK(cudaFree(matC_d));
    CUBLAS_CHECK(cublasDestroy(ctx));
}