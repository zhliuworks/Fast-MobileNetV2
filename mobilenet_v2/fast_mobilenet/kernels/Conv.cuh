#pragma once
#include "utils.cuh"
#include "Im2Col.cuh"

/* Convolution with CUDA kernel (+ bias, ReLU6 activation) */

#define w_data_d(i, j)      w_data_d[(j) * out_channel + (i)]
#define x_data_d(i, j)      x_data_d[(j) * in_channel + (i)]
#define col_data_d(i, j)    col_data_d[(j) * col_h + (i)]
#define y_data_d(i, j)      y_data_d[(j) * out_channel + (i)]
#define w_data_s(i, j)      w_data_s[(j) * n + (i)]
#define x_data_s(i, j)      x_data_s[(j) * n + (i)]
#define col_data_s(i, j)    col_data_s[(j) * n + (i)]

// 1x1 Conv
// Note: use fixed convolution settings
// (padding = 0, stride = 1, dilation = 1, group = 1)
__global__ void fastConvKernel_1x1(
    float       *w_data_d,      // [M * K]
    float       *x_data_d,      // [K * N]
    float       *y_data_d,      // [M * N]
    int          out_channel,   // M
    int          in_map_size,   // N
    int          in_channel,    // K
    bool         activation,    // whether to use ReLU6()
    int          n              // n = dimBlock.x = dimBlock.y
) {
    extern __shared__ float shared[];
    float *w_data_s = shared;
    float *x_data_s = &shared[n * n];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * n + tx;
    int y = blockIdx.y * n + ty;
    float y_data_r;

    float sum = 0.0f;
    for (int k = 0; k < in_channel; k += n) {
        if (k + ty < in_channel) {
            w_data_s(tx, ty) = w_data_d(x, k + ty);
        } else {
            w_data_s(tx, ty) = 0.0f;
        }

        if (k + tx < in_channel) {
            x_data_s(tx, ty) = x_data_d(k + tx, y);
        } else {
            x_data_s(tx, ty) = 0.0f;
        }
        __syncthreads();

        for (int t = 0; t < n; t++) {
            sum += w_data_s(tx, t) * x_data_s(t, ty);
        }
        __syncthreads();
    }

    if (x < out_channel && y < in_map_size) {
        y_data_r = sum + y_data_d(x, y);
        if (activation) {
            y_data_r = (y_data_r > 0.0f) ? ((y_data_r < 6.0f) ? y_data_r : 6.0f) : 0.0f;
        }
        y_data_d(x, y) = y_data_r;
    }
}

// Conv -- Im2Col + GEMM
// Note: the kernel takes transformed column matrix as input, and performs GEMM,
// the Im2Col operation is done in `im2col_gpu_wrapper()` in `Im2Col.cuh`
__global__ void fastConvKernel_im2col_gemm(
    float       *w_data_d,      // [M * K]
    float       *col_data_d,    // [K * N]
    float       *y_data_d,      // [M * N]
    int          out_channel,   // M
    int          col_w,         // N (out_h * out_w)
    int          col_h,         // K (in_channel * kernel_h * kernel_w)
    bool         activation,    // whether to use ReLU6()
    int          n              // n = dimBlock.x = dimBlock.y
) {
    extern __shared__ float shared[];
    float *w_data_s = shared;
    float *col_data_s = &shared[n * n];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * n + tx;
    int y = blockIdx.y * n + ty;
    float y_data_r;

    float sum = 0.0f;
    for (int k = 0; k < col_h; k += n) {
        if (k + ty < col_h) {
            w_data_s(tx, ty) = w_data_d(x, k + ty);
        } else {
            w_data_s(tx, ty) = 0.0f;
        }

        if (k + tx < col_h) {
            col_data_s(tx, ty) = col_data_d(k + tx, y);
        } else {
            col_data_s(tx, ty) = 0.0f;
        }
        __syncthreads();

        for (int t = 0; t < n; t++) {
            sum += w_data_s(tx, t) * col_data_s(t, ty);
        }
        __syncthreads();
    }

    if (x < out_channel && y < col_w) {
        y_data_r = sum + y_data_d(x, y);
        if (activation) {
            y_data_r = (y_data_r > 0.0f) ? ((y_data_r < 6.0f) ? y_data_r : 6.0f) : 0.0f;
        }
        y_data_d(x, y) = y_data_r;
    }
}

// Conv -- Winograd algorithm
__global__ void fastConvKernel_winograd(

) {
    
}

// For 1x1 Conv, use `fastConvKernel_1x1`
// For 3x3 Conv, use `fastConvKernel_im2col_gemm`
__host__ void fastConvWrapper(
    float       *x_data_h,      // input feature maps
    TensorShape *x_shape,       // input shape
    float       *w_data_h,      // filter
    FilterShape *w_shape,       // filter shape
    float       *b_data_h,      // bias
    bool         winograd,      // for 3x3: true -> Winograd algo. / false -> Im2Col + GEMM, and for 1x1: just GEMM
    ConvConfig  *conv_cfg,      // convolution configs (padding, stride, dilation, group)
    bool         activation,    // whether to apply activation function (ReLU6)
    float       *y_data_h,      // output feature maps
    TensorShape *y_shape,       // output shape
    int          n
) {

    // shape check
    if (w_shape->c * conv_cfg->group != x_shape->c) {
        fprintf(stderr, "input channel of w and x are not matched\n");
        exit(EXIT_FAILURE);
    }

    /** 1. Input **/
    float *x_data_d;
    int x_bytes = x_shape->n * x_shape->c * x_shape->h * x_shape->w * sizeof(float);
    CUDA_CHECK(cudaMalloc(&x_data_d, x_bytes));
    CUDA_CHECK(cudaMemcpy(x_data_d, x_data_h, x_bytes, cudaMemcpyHostToDevice));

    /** 2. Filter **/
    float *w_data_d;
    int w_map_size = w_shape->h * w_shape->w;
    int w_bytes = w_shape->k * x_shape->c * w_map_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&w_data_d, w_bytes));
    CUDA_CHECK(cudaMemcpy(w_data_d, w_data_h, w_bytes, cudaMemcpyHostToDevice));

    /** 3. Output **/
    y_shape->n = x_shape->n;
    y_shape->c = w_shape->k;
    y_shape->h = (x_shape->h - w_shape->h + (conv_cfg->padding << 1)) / conv_cfg->stride + 1;
    y_shape->w = (x_shape->w - w_shape->w + (conv_cfg->padding << 1)) / conv_cfg->stride + 1;
    
    float *y_data_d;
    int y_map_size = y_shape->h * y_shape->w;
    int y_size = y_shape->n * y_shape->c * y_map_size;
    int y_bytes = y_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&y_data_d, y_bytes));
    for (int i = 0; i < y_size; i++) {
        CUDA_CHECK(cudaMemcpy(y_data_d + i, b_data_h + i / y_map_size, sizeof(float), cudaMemcpyHostToDevice));
    }

    /** 4. Convolution Forward **/

    // 1x1 Conv (fixed settings as above)
    if (w_shape->h == 1 && w_shape->w == 1) {
        int in_map_size = x_shape->h * x_shape->w;
        dim3 dimBlock(n, n, 1);
        dim3 dimGrid((w_shape->k + n - 1) / n, (in_map_size + n - 1) / n, 1);

        fastConvKernel_1x1<<<dimGrid, dimBlock, (sizeof(float) * n * n << 1)>>>(
            x_data_d, w_data_d, y_data_d, in_map_size, w_shape->k, x_shape->c, activation, n
        );
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
    }

    // 3x3 Conv: Winograd algo.
    else if (winograd) {
        // winograd algorithm implementation
        // ...
    }

    // 3x3 Conv: Im2Col + GEMM
    else {
        int col_h = x_shape->c * w_map_size;
        int col_w = y_map_size;
        int col_size = col_h * col_w;
        int col_bytes = col_size * sizeof(float);
        float *col_data = (float*)malloc(col_bytes);
        int block_size = 256;  // hyperparameter
        int grid_size = 0;     // hyperparameter

        im2col_gpu_wrapper(
            x_data_h, x_shape->c, x_shape->h, x_shape->w, w_shape->h, w_shape->w,
            conv_cfg->padding, conv_cfg->padding, conv_cfg->stride, conv_cfg->stride,
            col_data, block_size, grid_size
        );

        /* `only group = 1`
        float *col_data_d;
        CUDA_CHECK(cudaMalloc(&col_data_d, col_bytes));
        CUDA_CHECK(cudaMemcpy(col_data_d, col_data, col_bytes, cudaMemcpyHostToDevice));

        dim3 dimBlock(n, n, 1);
        dim3 dimGrid((w_shape->k + n - 1) / n, (col_w + n - 1) / n, 1);

        fastConvKernel_im2col_gemm<<<dimGrid, dimBlock, (sizeof(float) * n * n << 1)>>>(
            col_data_d, w_data_d, y_data_d, col_w, w_shape->k, col_h, activation, n
        );
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(col_data_d));
        */

        int in_channel_per_group = x_shape->c / conv_cfg->group;
        int out_channel_per_group = w_shape->k / conv_cfg->group;

        float *col_data_d;
        CUDA_CHECK(cudaMalloc(&col_data_d, col_bytes));
        CUDA_CHECK(cudaMemcpy(col_data_d, col_data, col_bytes, cudaMemcpyHostToDevice));

        for (int g = 0; g < conv_cfg->group; g++) {
            float *col_data_g = col_data_d + g * (col_size / conv_cfg->group);
            float *w_data_g = w_data_d + g * (w_map_size * in_channel_per_group * out_channel_per_group);
            float *y_data_g = y_data_d + g * (y_map_size * out_channel_per_group);

            dim3 dimBlock(n, n, 1);
            dim3 dimGrid((out_channel_per_group + n - 1) / n, (col_w + n - 1) / n, 1);

            fastConvKernel_im2col_gemm<<<dimGrid, dimBlock, (sizeof(float) * n * n << 1)>>>(
                col_data_g, w_data_g, y_data_g,
                col_w, out_channel_per_group, col_h / conv_cfg->group,
                activation, n
            );
            cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaFree(col_data_d));
        free(col_data);
    }

    CUDA_CHECK(cudaMemcpy(
        y_data_h, y_data_d,
        y_shape->n * y_shape->c * y_shape->h * y_shape->w * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // finalize
    CUDA_CHECK(cudaFree(x_data_d));
    CUDA_CHECK(cudaFree(w_data_d));
    CUDA_CHECK(cudaFree(y_data_d));
}