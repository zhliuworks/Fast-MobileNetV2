#pragma once
#include "kernels/Conv.cuh"
#include "kernels/Gemm.cuh"
#include "kernels/Pool.cuh"
#include "kernels/Add.cuh"
#include "../nn/MobileNetV2.h"

const int NUM_THREADS = 32;

/* MobileNetV2 with our optimized cuda kernels */
__host__ void fastMobileNetV2(
    float       *x_data,
    TensorShape  x_shape,
    std::string  weights_path,
    float       *y_data,
    int          y_shape
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

    ConvConfig conv_cfg;
    bool activation;

    // winograd / im2col
    bool use_winograd = false;

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

                fastConvWrapper(
                    x_data_mid, &x_shape_mid,
                    w_data, &w_shape,
                    b_data, use_winograd,
                    &conv_cfg, activation,
                    y_data_mid, &y_shape_mid,
                    NUM_THREADS
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
                fastPoolWrapper(x_data_mid, &x_shape_mid, y_data_mid, &y_shape_mid);
                
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

                fastGemmWrapper(x_data_mid, w_data, b_data, x_shape.n, y_shape, x_shape_mid.c, NUM_THREADS);

                memcpy(y_data, b_data, sizeof(float) * y_shape);

                // std::cout << '(' << x_shape_mid.n << ", " << y_shape << ')' << std::endl; 
                break;

            case 3:
                /* ReLU6 */
                break;

            case 4:
                /* ResidualAdd */
                size = x_shape_mid.n * x_shape_mid.c * x_shape_mid.h * x_shape_mid.w;
                fastAddWrapper(residual_data, x_data_mid, size, NUM_THREADS);
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