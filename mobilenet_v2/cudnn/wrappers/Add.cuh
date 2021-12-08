#pragma once
#include "utils.cuh"

/* Add Tensor with cuDNN */
__host__ void cudnnAddWrapper(
    float       *src_h,    // source tensor
    float       *dest_h,   // destination tensor
    TensorShape  shape     // tensor shape
) {

    // calculate size
    int size = shape.n * shape.c * shape.h * shape.w;

    // allocate device memory
    float *src_d;
    float *dest_d;

    CUDA_CHECK(cudaMalloc(
        (void**)&src_d, size * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&dest_d, size * sizeof(float)
    ));

    // download host data to device
    CUDA_CHECK(cudaMemcpy(
        src_d, src_h, size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dest_d, dest_h, size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // tensor descriptor
    cudnnTensorDescriptor_t src_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&src_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        src_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        shape.n, shape.c, shape.h, shape.w
    ));

    cudnnTensorDescriptor_t dest_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dest_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        dest_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        shape.n, shape.c, shape.h, shape.w
    ));

    // perform addition
    cudnnHandle_t ctx;
    CUDNN_CHECK(cudnnCreate(&ctx));
    float alpha = 1.0f;
    float beta = 1.0f;

    CUDNN_CHECK(cudnnAddTensor(
        ctx, &alpha, src_desc, src_d,
        &beta, dest_desc, dest_d
    ));

    // upload device data to host
    CUDA_CHECK(cudaMemcpy(
        dest_h, dest_d, size * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // finalize
    CUDA_CHECK(cudaFree(src_d));
    CUDA_CHECK(cudaFree(dest_d));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(src_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dest_desc));
    CUDNN_CHECK(cudnnDestroy(ctx));
}