#pragma once
#include "utils.cuh"

/* Tensor Addition with CUDA kernel */

// kernel v1: 0.005258 ms
__global__ void fastAddKernel_v1(
    float   *src_d,    // source tensor on device
    float   *dest_d    // destination tensor on device
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dest_d[idx] += src_d[idx];
}

// kernel v2: 0.005212 ms
// src and dest are contiguous
__global__ void fastAddKernel_v2(
    float   *data_d,   // <destination tensor, source tensor>
    int      size      // tensor size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data_d[idx] += data_d[idx + size];
}

// kernel v3: 0.005451 ms
// 2x fewer threads, 2x more per-thread work
__global__ void fastAddKernel_v3(
    float   *data_d,   // <destination tensor, source tensor>
    int      size      // tensor size
) {
    int idx = (blockIdx.x * blockDim.x << 1) + threadIdx.x;
    data_d[idx] += data_d[idx + size];
    data_d[idx + blockDim.x] += data_d[idx + size + blockDim.x];
}


// use `fastAddKernel_v2`
__host__ void fastAddWrapper(
    float   *src_h,      // source tensor on host
    float   *dest_h,     // destination tensor on host
    int      size,       // tensor size
    int      num_threads // number of threads
) {
    float *data_d;
    int bytes = size * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&data_d, bytes << 1));
    
    CUDA_CHECK(cudaMemcpy(data_d, dest_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data_d + size, src_h, bytes, cudaMemcpyHostToDevice));

    dim3 dimGrid(size / num_threads, 1, 1);
    dim3 dimBlock(num_threads, 1, 1);
    
    fastAddKernel_v2<<<dimGrid, dimBlock>>>(data_d, size);
    cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(dest_h, data_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(data_d));
}