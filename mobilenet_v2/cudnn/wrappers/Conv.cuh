#pragma once
#include "utils.cuh"

/* Convolution Forward with cuDNN: Fusion of Conv + Bias + ReLU6 operators */
__host__ void cudnnConvForwardWrapper(
    float       *x_data_h,      // input feature maps from host
    TensorShape  x_shape,       // input shape
    float       *w_data_h,      // filter from host
    FilterShape  w_shape,       // filter shape
    float       *b_data_h,      // bias from host
    int          b_shape,       // bias shape
    ConvConfig   conv_cfg,      // convolution configs (padding, stride, dilation, group)
    int          algorithm,     // convolution algorithm ([0,7]->specific algo / -1->search)
    bool         activation,    // whether to apply activation function (ReLU6)
    bool         tensor_core,   // whether to allow use of tensor core
    float       *y_data_h,      // output feature maps from host (user-allocated)
    TensorShape *y_shape,       // output shape (user-allocated)
    bool         print_algo=false   // whether to print used conv algorithm
) {

    // shape check
    if (w_shape.c * conv_cfg.group != x_shape.c) {
        fprintf(stderr, "input channel of w and x are not matched\n");
        exit(EXIT_FAILURE);
    }
    if (w_shape.k != b_shape) {
        fprintf(stderr, "output channel of w and b are not matched\n");
        exit(EXIT_FAILURE);        
    }

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

    /** 3. Filter **/
    // filter descriptor
    cudnnFilterDescriptor_t w_desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        w_shape.k, w_shape.c, w_shape.h, w_shape.w
    ));
    // allocate device memory for filter
    float *w_data_d;
    CUDA_CHECK(cudaMalloc(
        &w_data_d,
        w_shape.k * w_shape.c * w_shape.h * w_shape.w * sizeof(float)
    ));
    // download filter to device
    CUDA_CHECK(cudaMemcpy(
        w_data_d, w_data_h,
        w_shape.k * w_shape.c * w_shape.h * w_shape.w * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    /** 4. Bias **/
    // bias descriptor
    cudnnTensorDescriptor_t b_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        b_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        // dim1 = 1, dim2 = dim1 of filter: 
        /* https://docs.nvidia.com/deeplearning/cudnn/api/
           index.html#cudnnConvolutionBiasActivationForward */
        1, b_shape, 1, 1
    ));

    // allocate device memory for bias
    float *b_data_d;
    CUDA_CHECK(cudaMalloc(
        &b_data_d,
        b_shape * sizeof(float)
    ));
    // download bias to device
    CUDA_CHECK(cudaMemcpy(
        b_data_d, b_data_h,
        b_shape * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    /** 5. Convolution Descriptor **/
    // convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        conv_cfg.padding, conv_cfg.padding,
        conv_cfg.stride, conv_cfg.stride,
        conv_cfg.dilation, conv_cfg.dilation,
        CUDNN_CROSS_CORRELATION,  // not `CUDNN_CONVOLUTION`
        CUDNN_DATA_FLOAT
    ));
    // set group count
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, conv_cfg.group));
    // whether or not to allow using tensor core operations
    if (tensor_core) {
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
    } else {
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));
    }

    /** 6. Output **/
    // infer the output shape
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, x_desc, w_desc,
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

    /** 7. Convolution Forward Algorithm **/
    cudnnConvolutionFwdAlgo_t conv_algo;
    if (algorithm >= 0 && algorithm <= 7) {
        // set a specific algorithm
        conv_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algorithm);
    } else if (algorithm == -1) {
        // search an optimal algorithm
        /*  for cudnn < 8.0
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
            ctx,
            x_desc, filt_desc, conv_desc, y_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0, &conv_algo
        ));
        */
        const int req_algo_count = 8;
        int algo_count;
        cudnnConvolutionFwdAlgoPerf_t perf_results[req_algo_count];
        CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
            ctx,
            x_desc, w_desc, conv_desc, y_desc,
            req_algo_count, &algo_count, perf_results
        ));
        conv_algo = perf_results[0].algo;
    } else {
        fprintf(stderr, "algorithm argument invalid\n");
        exit(EXIT_FAILURE);
    }

    /** 8. Activation **/
    // activation descriptor
    cudnnActivationDescriptor_t acti_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&acti_desc));
    if (activation) {
        // ReLU activation (first apply ReLU, and then ReLU6)
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            acti_desc,
            CUDNN_ACTIVATION_RELU,
            CUDNN_NOT_PROPAGATE_NAN,
            0.0f
        ));
    } else {
        // bypass the activation
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            acti_desc,
            CUDNN_ACTIVATION_IDENTITY,
            CUDNN_NOT_PROPAGATE_NAN,
            0.0f
        ));
        // `CUDNN_ACTIVATION_IDENTITY` must use algo. `IMPLICIT_PRECOMP_GEMM`
        /* https://docs.nvidia.com/deeplearning/cudnn/api/
           index.html#cudnnConvolutionBiasActivationForward */
        conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }

    /* print used algorithm */
    if (print_algo) {
        printf("Using convolution algorithm: ");
        switch (conv_algo) {
            case 0: printf("IMPLICIT_GEMM");         break;
            case 1: printf("IMPLICIT_PRECOMP_GEMM"); break;
            case 2: printf("GEMM");                  break;
            case 3: printf("DIRECT");                break;
            case 4: printf("FFT");                   break;
            case 5: printf("FFT_TILING");            break;
            case 6: printf("WINOGRAD");              break;
            case 7: printf("WINOGRAD_NONFUSED");     break;
            default: exit(EXIT_FAILURE);
        }
        printf("\n");
    }

    /** 9. Workspace **/
    size_t ws_size;
    float *ws_data;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        ctx, x_desc, w_desc, conv_desc, y_desc,
        conv_algo, &ws_size
    ));
    CUDA_CHECK(cudaMalloc(
        &ws_data, ws_size
    ));
    // printf("Workspace size: %lu\n", ws_size);

    /** 10. Perform Forward Convolution **/

    /*  convolution + bias
    float alpha = 1.0f;
    float beta = 1.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        ctx,
        &alpha,
        x_desc, x_data_d,
        w_desc, w_data_d,
        conv_desc, conv_algo,
        ws_data, ws_size,
        &beta,
        y_desc, y_data_d
    ));
    */

    /* convolution + bias + ReLU activation */
    // note that `CUDNN_ACTIVATION_CLIPPED_RELU` is not allowed in `cudnnConvolutionBiasActivationForward()`,
    // we can apply ReLU first, and then ReLU6 with `cudnnActivationForward()`
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
        ctx,
        &alpha,
        x_desc, x_data_d,
        w_desc, w_data_d,
        conv_desc, conv_algo,
        ws_data, ws_size,
        &beta,
        // zDesc and destDesc need to match:
        // https://docs.nvidia.com/deeplearning/cudnn/api/
        // index.html#cudnnConvolutionBiasActivationForward
        y_desc, y_data_d,
        b_desc, b_data_d,
        acti_desc,
        y_desc, y_data_d
    ));

    /* ReLU6 activation */
    if (activation) {
        // reset activation descriptor to ReLU6
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            acti_desc,
            CUDNN_ACTIVATION_CLIPPED_RELU,
            CUDNN_NOT_PROPAGATE_NAN,
            6.0f
        ));

        CUDNN_CHECK(cudnnActivationForward(
            ctx,
            acti_desc,
            &alpha,
            y_desc, y_data_d,
            &beta,
            y_desc, y_data_d
        ));
    }

    // upload output data to host
    CUDA_CHECK(cudaMemcpy(
        y_data_h, y_data_d,
        y_shape->n * y_shape->c * y_shape->h * y_shape->w * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    /** 11. Finalize **/
    CUDA_CHECK(cudaFree(x_data_d));
    CUDA_CHECK(cudaFree(w_data_d));
    CUDA_CHECK(cudaFree(b_data_d));
    CUDA_CHECK(cudaFree(y_data_d));
    CUDA_CHECK(cudaFree(ws_data));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(b_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(acti_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(ctx));
}