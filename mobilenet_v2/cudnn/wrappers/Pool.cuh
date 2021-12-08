#pragma once
#include "utils.cuh"

/* Global Average Pooling Forward with cuDNN */
__host__ void cudnnPoolForwardWrapper(
    float       *x_data_h,      // input feature maps from host
    TensorShape  x_shape,       // input shape
    PoolConfig   pool_cfg,      // pooling configs (window_size, padding, stride)
    float       *y_data_h,      // output feature maps from host (user-allocated)
    TensorShape *y_shape        // output shape (user-allocated)
) {

    /** 1. Handle **/
    cudnnHandle_t ctx;
    CUDNN_CHECK(cudnnCreate(&ctx));

    /** 2. Input **/
    // input tensor descriptor
    cudnnTensorDescriptor_t x_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        x_shape.n, x_shape.c, x_shape.h, x_shape.w
    ));
    // allocate device memory for input data
    float *x_data_d;
    CUDA_CHECK(cudaMalloc(
        &x_data_d,
        x_shape.n * x_shape.c * x_shape.h * x_shape.w * sizeof(float)
    ));
    // download input data to device
    CUDA_CHECK(cudaMemcpy(
        x_data_d, x_data_h,
        x_shape.n * x_shape.c * x_shape.h * x_shape.w * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    /** 3. Pooling **/
    cudnnPoolingDescriptor_t pool_desc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,
        pool_cfg.window_size, pool_cfg.window_size,
        pool_cfg.padding, pool_cfg.padding,
        pool_cfg.stride, pool_cfg.stride
    ));

    /** 4. Output **/
    // infer the output shape
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(
        pool_desc, x_desc,
        &(y_shape->n), &(y_shape->c), &(y_shape->h), &(y_shape->w)
    ));
    // output descriptor
    cudnnTensorDescriptor_t y_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        y_shape->n, y_shape->c, y_shape->h, y_shape->w
    ));
    // allocate device memory for output data
    float *y_data_d;
    CUDA_CHECK(cudaMalloc(
        &y_data_d,
        y_shape->n * y_shape->c * y_shape->h * y_shape->w * sizeof(float)
    ));

    /** 5. Perform Forward Pooling **/
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(
        ctx,
        pool_desc,
        &alpha,
        x_desc, x_data_d,
        &beta,
        y_desc, y_data_d
    ));

    // upload output data to host
    CUDA_CHECK(cudaMemcpy(
        y_data_h, y_data_d,
        y_shape->n * y_shape->c * y_shape->h * y_shape->w * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    /** 6. Finalize **/
    CUDA_CHECK(cudaFree(x_data_d));
    CUDA_CHECK(cudaFree(y_data_d));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
    CUDNN_CHECK(cudnnDestroy(ctx));
}