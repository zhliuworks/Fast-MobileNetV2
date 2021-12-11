#pragma once
#include "utils.cuh"

/* Matrix Multiplication (SGEMM) with CUDA kernel (+ bias) */

/* row-major layout */
// #define matA_d(i, j)     matA_d[(i) * K + (j)]
// #define matB_d(i, j)     matB_d[(i) * N + (j)]
// #define matC_d(i, j)     matC_d[(i) * N + (j)]
// #define matA_s(i, j)     matA_s[(i) * n + (j)]
// #define matB_s(i, j)     matB_s[(i) * n + (j)]

/* col-major layout */
#define matA_d(i, j)     matA_d[(j) * M + (i)]
#define matB_d(i, j)     matB_d[(j) * K + (i)]
#define matC_d(i, j)     matC_d[(j) * M + (i)]
#define matA_s(i, j)     matA_s[(j) * n + (i)]
#define matB_s(i, j)     matB_s[(j) * n + (i)]


// kernel v1: 4.097066 ms (row-major)
//            1.095126 ms (col-major)
// so we choose column-major layout in all of the subsequent experiments
__global__ void fastGemmKernel_v1(
    float   *matA_d,     // matA: [M * K] on device
    float   *matB_d,     // matB: [K * N] on device
    float   *matC_d,     // matC: [M * N] on device, initialized with bias
    int      M,
    int      N,
    int      K,
    int      n           // n = dimBlock.x = dimBlock.y
) {
    int x = blockIdx.x * n + threadIdx.x;
    int y = blockIdx.y * n + threadIdx.y;

    if (x < M && y < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += matA_d(x, k) * matB_d(k, y);
        }
        matC_d(x, y) += sum;
    }
}

// kernel v2: 0.905893 ms
// partition matA and matB into tiles, and load them into shared memory
__global__ void fastGemmKernel_v2(
    float   *matA_d,     // matA: [M * K] on device
    float   *matB_d,     // matB: [K * N] on device
    float   *matC_d,     // matC: [M * N] on device, initialized with bias
    int      M,
    int      N,
    int      K,
    int      n           // n = dimBlock.x = dimBlock.y
) {
    extern __shared__ float mat_s[];
    float *matA_s = mat_s;
    float *matB_s = &mat_s[n * n];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * n + tx;
    int y = blockIdx.y * n + ty;

    float sum = 0.0f;
    for (int k = 0; k < K; k += n) {
        if (k + ty < K) {
            matA_s(tx, ty) = matA_d(x, k + ty);
        } else {
            matA_s(tx, ty) = 0.0f;
        }

        if (k + tx < K) {
            matB_s(tx, ty) = matB_d(k + tx, y);
        } else {
            matB_s(tx, ty) = 0.0f;
        }
        __syncthreads();

        for (int t = 0; t < n; t++) {
            sum += matA_s(tx, t) * matB_s(t, ty);
        }
        __syncthreads();
    }

    if (x < M && y < N) {
        matC_d(x, y) += sum;
    }
}

// kernel v3: 0.952614 ms
// increase per-thread work, each thread load 4x1 A and 1x1 B,
// and compute 4x1 C, with 4x fewer threads
// BUT too many if-else divergence
__global__ void fastGemmKernel_v3(
    float   *matA_d,     // matA: [M * K] on device
    float   *matB_d,     // matB: [K * N] on device
    float   *matC_d,     // matC: [M * N] on device, initialized with bias
    int      M,
    int      N,
    int      K,
    int      n           // n/4 = dimBlock.x, n = dimBlock.y
) {
    extern __shared__ float mat_s[];
    float *matA_s = mat_s;
    float *matB_s = &mat_s[n * n];

    int tx0 = threadIdx.x << 2;
    int tx[4] = {tx0, tx0 + 1, tx0 + 2, tx0 + 3};
    int ty = threadIdx.y;

    int x0 = blockIdx.x * n + tx0;
    int x[4] = {x0, x0 + 1, x0 + 2, x0 + 3};
    int y = blockIdx.y * n + ty;

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < K; k += n) {
        if (k + ty < K) {
            matA_s(tx[0], ty) = matA_d(x[0], k + ty);
            matA_s(tx[1], ty) = matA_d(x[1], k + ty);
            matA_s(tx[2], ty) = matA_d(x[2], k + ty);
            matA_s(tx[3], ty) = matA_d(x[3], k + ty);
        } else {
            matA_s(tx[0], ty) = 0.0f;
            matA_s(tx[1], ty) = 0.0f;
            matA_s(tx[2], ty) = 0.0f;
            matA_s(tx[3], ty) = 0.0f;
        }

        if (k + tx[0] < K) {
            matB_s(tx[0], ty) = matB_d(k + tx[0], y);
        } else {
            matB_s(tx[0], ty) = 0.0f;
        }
        if (k + tx[1] < K) {
            matB_s(tx[1], ty) = matB_d(k + tx[1], y);
        } else {
            matB_s(tx[1], ty) = 0.0f;
        }
        if (k + tx[2] < K) {
            matB_s(tx[2], ty) = matB_d(k + tx[2], y);
        } else {
            matB_s(tx[2], ty) = 0.0f;
        }
        if (k + tx[3] < K) {
            matB_s(tx[3], ty) = matB_d(k + tx[3], y);
        } else {
            matB_s(tx[3], ty) = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < n; t++) {
            sum[0] += matA_s(tx[0], t) * matB_s(t, ty);
            sum[1] += matA_s(tx[1], t) * matB_s(t, ty);
            sum[2] += matA_s(tx[2], t) * matB_s(t, ty);
            sum[3] += matA_s(tx[3], t) * matB_s(t, ty);
        }
        __syncthreads();
    }

    if (x[0] < M && y < N) {
        matC_d(x[0], y) += sum[0];
    }
    if (x[1] < M && y < N) {
        matC_d(x[1], y) += sum[1];
    }
    if (x[2] < M && y < N) {
        matC_d(x[2], y) += sum[2];
    }
    if (x[3] < M && y < N) {
        matC_d(x[3], y) += sum[3];
    }
}

// more optimization strategies:
// - vectorized load/store from/to global memory
// - warp-level tiling
// - prefetching
// - double-buffering
// ...


// use `fastGemmKernel_v2`
__host__ void fastGemmWrapper(
    float   *matA_h,     // matA: [M * K] on host
    float   *matB_h,     // matB: [K * N] on host
    float   *matC_h,     // matC: [M * N] on host, initialized with bias
    int      M,
    int      N,
    int      K,
    int      n           // n = dimBlock.x = dimBlock.y
) {
    float *matA_d, *matB_d, *matC_d;
    int matA_bytes = M * K * sizeof(float);
    int matB_bytes = K * N * sizeof(float);
    int matC_bytes = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&matA_d, matA_bytes));
    CUDA_CHECK(cudaMalloc((void**)&matB_d, matB_bytes));
    CUDA_CHECK(cudaMalloc((void**)&matC_d, matC_bytes));

    CUDA_CHECK(cudaMemcpy(matA_d, matA_h, matA_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matB_d, matB_h, matB_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matC_d, matC_h, matC_bytes, cudaMemcpyHostToDevice));

    // dim3 dimBlock(n >> 2, n, 1);  // for kernel v3
    dim3 dimBlock(n, n, 1);
    dim3 dimGrid((M + n - 1) / n, (N + n - 1) / n, 1);

    // fastGemmKernel_v1<<<dimGrid, dimBlock>>>(matA_d, matB_d, matC_d, M, N, K, n);  // for kernel v1
    fastGemmKernel_v2<<<dimGrid, dimBlock, (sizeof(float) * n * n << 1)>>>(matA_d, matB_d, matC_d, M, N, K, n);
    cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(matC_h, matC_d, matC_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(matA_d));
    CUDA_CHECK(cudaFree(matB_d));
    CUDA_CHECK(cudaFree(matC_d));
}