#pragma once
#include "wrappers/Conv.cuh"
#include "wrappers/Gemm.cuh"
#include "wrappers/Pool.cuh"
#include "wrappers/Add.cuh"
#include "../nn/MobileNetV2.h"

/* MobileNetV2 with cuDNN */
__host__ void cudnnMobileNetV2(
    float       *x_data,
    TensorShape  x_shape,
    std::string  weights_path,
    float       *y_data,
    int          y_shape,
    bool         use_tensor_core=false
) {

    MobileNetV2 *mobileNetV2 = new MobileNetV2(weights_path);
    Layer *entry = mobileNetV2->getEntry();

    float *x_data_mid = (float*)malloc(sizeof(float) * mobileNetV2->getMaxIntermediateSize());
    TensorShape x_shape_mid;

    float *y_data_mid = (float*)malloc(sizeof(float) * mobileNetV2->getMaxIntermediateSize());
    TensorShape y_shape_mid;

    float *residual_data = (float*)malloc(sizeof(float) * mobileNetV2->getMaxIntermediateSize());

    std::vector<std::pair<float*, std::vector<int>>> params;

    float *w_data;
    FilterShape w_shape;

    float *b_data;
    int b_shape;

    ConvConfig conv_cfg;
    bool activation;
    
    int conv_algo = 1;
    /*
    0: IMPLICIT_GEMM
    1: IMPLICIT_PRECOMP_GEMM
    2: GEMM
    */

    PoolConfig pool_cfg;

    int size = x_shape.n * x_shape.c * x_shape.h * x_shape.w;
    memcpy(x_data_mid, x_data, sizeof(float) * size);
    x_shape_mid = x_shape;

    // forward computation
    while (entry) {
        switch (entry->getType()) {
            case 0:
                /* Conv2d */
                params = entry->getParameters();

                w_data = params[0].first;
                w_shape = FilterShape(
                    params[0].second[0],
                    params[0].second[1],
                    params[0].second[2],
                    params[0].second[3]
                );

                b_data = params[1].first;
                b_shape = params[1].second[0];

                conv_cfg = ConvConfig(
                    entry->getPadding(),
                    entry->getStride(),
                    1,
                    entry->getGroup()
                );

                if (entry->getNext() && entry->getNext()->getType() == 3) {
                    activation = true;
                } else {
                    activation = false;
                }

                cudnnConvForwardWrapper(
                    x_data_mid, x_shape_mid,
                    w_data, w_shape,
                    b_data, b_shape,
                    conv_cfg, conv_algo,
                    activation, use_tensor_core,
                    y_data_mid, &y_shape_mid,
                    false
                );

                size = y_shape_mid.n * y_shape_mid.c * y_shape_mid.h * y_shape_mid.w;
                memcpy(x_data_mid, y_data_mid, sizeof(float) * size);
                x_shape_mid = y_shape_mid;

                if (entry->getIsBypass()) {
                    memcpy(residual_data, x_data_mid, sizeof(float) * size);
                }

                // std::cout << '(' << x_shape_mid.n << ", " << x_shape_mid.c <<
                // ", " << x_shape_mid.h << ", " << x_shape_mid.w << ')' << std::endl;
                break;

            case 1:
                /* GlobalAveragePool */
                pool_cfg = PoolConfig(x_shape_mid.h, 0, 1);
                cudnnPoolForwardWrapper(
                    x_data_mid, x_shape_mid,
                    pool_cfg,
                    y_data_mid, &y_shape_mid
                );

                size = y_shape_mid.n * y_shape_mid.c * y_shape_mid.h * y_shape_mid.w;
                memcpy(x_data_mid, y_data_mid, sizeof(float) * size);
                x_shape_mid = y_shape_mid;

                // std::cout << '(' << x_shape_mid.n << ", " << x_shape_mid.c <<
                // ", " << x_shape_mid.h << ", " << x_shape_mid.w << ')' << std::endl;
                break;

            case 2:
                /* Linear */
                params = entry->getParameters();
                w_data = params[0].first;
                b_data = params[1].first;

                cudnnGemmWrapper(w_data, x_data_mid, b_data, y_shape, x_shape.n, x_shape_mid.c, -1);  
                memcpy(y_data, b_data, sizeof(float) * y_shape);

                // std::cout << '(' << x_shape_mid.n << ", " << y_shape << ')' << std::endl; 
                break;

            case 3:
                /* ReLU6 */
                break;

            case 4:
                /* ResidualAdd */
                cudnnAddWrapper(residual_data, x_data_mid, x_shape_mid);
                if (entry->getIsBypass()) {
                    memcpy(residual_data, x_data_mid, sizeof(float) * size);
                }
                break;
        }
        entry = entry->getNext();
    }

    free(x_data_mid);
    free(y_data_mid);
    free(residual_data);
}