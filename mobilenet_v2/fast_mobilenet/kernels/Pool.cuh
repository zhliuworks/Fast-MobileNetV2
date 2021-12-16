#pragma once
#include "utils.cuh"

/* Global Average Pooling with CUDA kernel */
/*
 *  reference: Optimizing Parallel Reduction in CUDA
 *  available at: https://vuduc.org/teaching/cse6230-hpcta-fa12/slides/cse6230-fa12--05b-reduction-notes.pdf
 */

// kernel v1: 0.004946 ms
__global__ void fastPoolKernel_v1(
    float   *x_data_d,      // input feature maps on device
    int      x_size,        // input size
    float   *y_data_d,      // output feature maps on device
    int      window_size    // pooling window size
) {
    // perform reduction (GAP) within a thread block, processed in shared memory
    extern __shared__ float shared[];  // size: num_threads

    // each thread loads one element from global to shared memory
    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tx] = (idx < x_size) ? x_data_d[idx] : 0.0f;
    __syncthreads();

    // do reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tx % (2 * stride) == 0) {
            shared[tx] += shared[tx + stride];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (tx == 0) {
        y_data_d[blockIdx.x] = shared[0] / window_size;
    }
}

// kernel v2: 0.004317 ms
// strided index and non-divergent branch 
__global__ void fastPoolKernel_v2(
    float   *x_data_d,      // input feature maps on device
    int      x_size,        // input size
    float   *y_data_d,      // output feature maps on device
    int      window_size    // pooling window size
) {
    extern __shared__ float shared[];

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tx] = (idx < x_size) ? x_data_d[idx] : 0.0f;
    __syncthreads();

    // strided index, non-divergent
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int index = 2 * stride * tx;
        if (index < blockDim.x) {
            shared[index] += shared[index + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        y_data_d[blockIdx.x] = shared[0] / window_size;
    }    
}

// kernel v3: 0.004208 ms
// sequential addressing
__global__ void fastPoolKernel_v3(
    float   *x_data_d,      // input feature maps on device
    int      x_size,        // input size
    float   *y_data_d,      // output feature maps on device
    int      window_size    // pooling window size
) {
    extern __shared__ float shared[];

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tx] = (idx < x_size) ? x_data_d[idx] : 0.0f;
    __syncthreads();

    // sequential addressing
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            shared[tx] += shared[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        y_data_d[blockIdx.x] = shared[0] / window_size;
    }
}

// kernel v4: 0.003918 ms
// unroll the last warp
__device__ void warpReduce(volatile float *shared, int tx) {
    shared[tx] += shared[tx + 32];
    shared[tx] += shared[tx + 16];
    shared[tx] += shared[tx + 8];
    shared[tx] += shared[tx + 4];
    shared[tx] += shared[tx + 2];
    shared[tx] += shared[tx + 1];
}
__global__ void fastPoolKernel_v4(
    float   *x_data_d,      // input feature maps on device
    int      x_size,        // input size
    float   *y_data_d,      // output feature maps on device
    int      window_size    // pooling window size
) {
    extern __shared__ float shared[];

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tx] = (idx < x_size) ? x_data_d[idx] : 0.0f;
    __syncthreads();

    // reduced util 32, and then unroll (sum directly)
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tx < stride) {
            shared[tx] += shared[tx + stride];
        }
        __syncthreads();
    }

    if (tx < 32) {
        warpReduce(shared, tx);
    }

    if (tx == 0) {
        y_data_d[blockIdx.x] = shared[0] / window_size;
    }  
}


// use `fastPoolKernel_v4`
__host__ void fastPoolWrapper(
    float       *x_data_h,      // input feature maps on host
    TensorShape *x_shape,       // input shape
    float       *y_data_h,      // output feature maps on host (user-allocated)
    TensorShape *y_shape        // output shape
) {
    assert(x_shape->n == 1);

    float *x_data_d, *y_data_d;
    int x_size = x_shape->n * x_shape->c * x_shape->h * x_shape->w;
    int x_bytes = x_size * sizeof(float);
    int y_bytes = x_shape->c * sizeof(float);
    int window_size = x_shape->h * x_shape->w;

    y_shape->n = 1;
    y_shape->c = x_shape->c;
    y_shape->h = 1;
    y_shape->w = 1;

    CUDA_CHECK(cudaMalloc((void**)&x_data_d, x_bytes));
    CUDA_CHECK(cudaMalloc((void**)&y_data_d, y_bytes));

    CUDA_CHECK(cudaMemcpy(x_data_d, x_data_h, x_bytes, cudaMemcpyHostToDevice));

    dim3 dimGrid(x_shape->c, 1, 1);
    dim3 dimBlock(window_size, 1, 1);

    fastPoolKernel_v4<<<dimGrid, dimBlock, window_size * sizeof(float)>>>(x_data_d, x_size, y_data_d, window_size);
    cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(y_data_h, y_data_d, y_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_data_d));
    CUDA_CHECK(cudaFree(y_data_d));
}