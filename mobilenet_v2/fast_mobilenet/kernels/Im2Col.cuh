#pragma once
#include "utils.cuh"

#define IM_DATA_POS im_data[channel * in_s + in_row * in_w + in_col]
#define COL_DATA_POS col_data[(channel * kernel_s + kernel_row * kernel_w + kernel_col) * out_s + out_row * out_w + out_col]

__host__ void im2col_cpu(
    float   *im_data,
    int      in_channel,
    int      in_h,
    int      in_w,
    int      kernel_h,
    int      kernel_w,
    int      padding_h,
    int      padding_w,
    int      stride_h,
    int      stride_w,
    float   *col_data
) {
    int out_h = (in_h - kernel_h + (padding_h << 1)) / stride_h + 1;
    int out_w = (in_w - kernel_w + (padding_w << 1)) / stride_w + 1;

    int in_s = in_h * in_w;
    int out_s = out_h * out_w;
    int kernel_s = kernel_h * kernel_w;

    for (int channel = 0; channel < in_channel; channel++) {
        for (int out_row = 0, in_row_start = -padding_h; out_row < out_h; out_row++, in_row_start += stride_h) {
            for (int out_col = 0, in_col_start = -padding_w; out_col < out_w; out_col++, in_col_start += stride_w) {

                for (int kernel_row = 0, in_row = in_row_start; kernel_row < kernel_h; kernel_row++, in_row++) {
                    if (in_row < 0 || in_row >= in_h) {
                        #pragma unroll
                        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                            COL_DATA_POS = 0.0f;
                        }  // kernel_col
                    }
                    else {
                        #pragma unroll
                        for (int kernel_col = 0, in_col = in_col_start; kernel_col < kernel_w; kernel_col++, in_col++) {
                            if (in_col < 0 || in_col >= in_w) {
                                COL_DATA_POS = 0.0f;
                            } else {
                                COL_DATA_POS = IM_DATA_POS;
                            }
                        }  // kernel_col, in_col
                    }
                } // kernel_row, in_row         
            }  // out_col, in_col_start
        }  // out_row, in_row_start
    }  // channel
}

__global__ void im2col_gpu_kernel(
    float   *im_data,
    int      in_channel,
    int      in_h,
    int      in_w,
    int      in_s,
    int      out_h,
    int      out_w,
    int      out_s,
    int      kernel_h,
    int      kernel_w,
    int      kernel_s,
    int      padding_h,
    int      padding_w,
    int      stride_h,
    int      stride_w,
    float   *col_data,
    int      n      // in_channel * out_h * out_w
) {
    // using grid-stride loop
    // https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) 
    {
        int channel = idx % in_channel;
        int out_row = (idx / in_channel) % out_h;
        int out_col = (idx / in_channel) / out_h;

        int in_row_start = -padding_h + stride_h * out_row;
        int in_col_start = -padding_w + stride_w * out_col;

        for (int kernel_row = 0, in_row = in_row_start; kernel_row < kernel_h; kernel_row++, in_row++) {
            if (in_row < 0 || in_row >= in_h) {
                #pragma unroll
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                    COL_DATA_POS = 0.0f;
                }
            }
            else {
                #pragma unroll
                for (int kernel_col = 0, in_col = in_col_start; kernel_col < kernel_w; kernel_col++, in_col++) {
                    if (in_col < 0 || in_col >= in_w) {
                        COL_DATA_POS = 0.0f;
                    } else {
                        COL_DATA_POS = IM_DATA_POS;
                    }
                }
            }
        }
    }
}

__host__ void im2col_gpu_wrapper(
    float   *im_data,
    int      in_channel,
    int      in_h,
    int      in_w,
    int      kernel_h,
    int      kernel_w,
    int      padding_h,
    int      padding_w,
    int      stride_h,
    int      stride_w,
    float   *col_data,
    int      block_size,
    int      grid_size = 0
) {
    int out_h = (in_h - kernel_h + (padding_h << 1)) / stride_h + 1;
    int out_w = (in_w - kernel_w + (padding_w << 1)) / stride_w + 1;

    int in_s = in_h * in_w;
    int out_s = out_h * out_w;
    int kernel_s = kernel_h * kernel_w;

    float *im_data_d;
    int im_bytes = in_channel * in_s * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&im_data_d, im_bytes));
    CUDA_CHECK(cudaMemcpy(im_data_d, im_data, im_bytes, cudaMemcpyHostToDevice));

    float *col_data_d;
    int col_bytes = in_channel * kernel_s * out_s * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&col_data_d, col_bytes));
    CUDA_CHECK(cudaMemcpy(col_data_d, col_data, col_bytes, cudaMemcpyHostToDevice));

    if (grid_size == 0) {
        grid_size = (in_channel * out_s + block_size - 1) / block_size;
    }

    dim3 dimGrid(grid_size, 1, 1);
    dim3 dimBlock(block_size, 1, 1);

    im2col_gpu_kernel<<<dimGrid, dimBlock>>>(
        im_data_d, in_channel, in_h, in_w, in_s, out_h, out_w, out_s, kernel_h, kernel_w,
        kernel_s, padding_h, padding_w, stride_h, stride_w, col_data_d, in_channel * out_s
    );
    cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(col_data, col_data_d, col_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(im_data_d));
    CUDA_CHECK(cudaFree(col_data_d));
}