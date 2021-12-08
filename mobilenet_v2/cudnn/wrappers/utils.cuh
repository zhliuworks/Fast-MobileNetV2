#pragma once
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) {                             \
    cudaError_t err = (call);                          \
    if (err != cudaSuccess) {                          \
        fprintf(stderr, "[CUDA ERROR line %d] %s\n",   \
                __LINE__, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                            \
    }                                                  \
}

#define CUDNN_CHECK(call) {                            \
    cudnnStatus_t err = (call);                        \
    if (err != CUDNN_STATUS_SUCCESS) {                 \
        fprintf(stderr, "[CUDNN ERROR line %d] %s\n",  \
                __LINE__, cudnnGetErrorString(err));   \
        exit(EXIT_FAILURE);                            \
    }                                                  \
}

#define CUBLAS_CHECK(call) {                           \
    cublasStatus_t err = (call);                       \
    if (err != CUBLAS_STATUS_SUCCESS) {                \
        fprintf(stderr, "[CUBLAS ERROR line %d] %d\n", \
                __LINE__, err);                        \
        exit(EXIT_FAILURE);                            \
    }                                                  \
}

struct TensorShape {
    int n;
    int c;
    int h;
    int w;
    TensorShape(int n=1, int c=1, int h=1, int w=1) : 
        n(n), c(c), h(h), w(w) {}
};

struct FilterShape {
    int k;
    int c;
    int h;
    int w;
    FilterShape(int k=1, int c=1, int h=1, int w=1) :
        k(k), c(c), h(h), w(w) {}
};

struct ConvConfig {
    int padding;
    int stride;
    int dilation;
    int group;
    ConvConfig(int padding=1, int stride=1, int dilation=1, int group=1) :
        padding(padding), stride(stride), dilation(dilation), group(group) {}
};

struct PoolConfig {
    int window_size;
    int padding;
    int stride;
    PoolConfig(int window_size=1, int padding=1, int stride=1) :
        window_size(window_size), padding(padding), stride(stride) {}
};