#include "../wrappers/Conv.cuh"

__host__ int main() {
    // input
    TensorShape x_shape(1, 3, 244, 244);
    float *x_data_h;
    x_data_h = (float*)malloc(1 * 3 * 244 * 244 * sizeof(float));

    // filter
    FilterShape w_shape(32, 3, 3, 3);
    float *w_data_h;
    w_data_h = (float*)malloc(32 * 3 * 3 * 3 * sizeof(float));

    // bias
    int b_shape = 32;
    float *b_data_h;
    b_data_h = (float*)malloc(32 * sizeof(float));

    // convolution config
    // ConvConfig(int padding, int stride, int dilation, int group)
    ConvConfig conv_cfg(1, 2, 1, 1);

    // output
    TensorShape y_shape(-1, -1, -1, -1);  // let cuDNN infer the output shape
    float *y_data_h;
    y_data_h = (float*)malloc(1 * 32 * 122 * 122 * sizeof(float));

    // test forward
    /* search algo, activation=true, tensor_core=false */
    cudnnConvForwardWrapper(x_data_h, x_shape, w_data_h, w_shape, b_data_h, b_shape,
                            conv_cfg, -1, true, false, y_data_h, &y_shape);
    assert(y_shape.n == 1);
    assert(y_shape.c == 32);
    assert(y_shape.h == 122);
    assert(y_shape.w == 122);

    /* search algo, activation=true, tensor_core=true */
    cudnnConvForwardWrapper(x_data_h, x_shape, w_data_h, w_shape, b_data_h, b_shape,
                            conv_cfg, -1, true, true, y_data_h, &y_shape); 
    assert(y_shape.n == 1);
    assert(y_shape.c == 32);
    assert(y_shape.h == 122);
    assert(y_shape.w == 122);

    /* search algo, activation=false, tensor_core=true */
    cudnnConvForwardWrapper(x_data_h, x_shape, w_data_h, w_shape, b_data_h, b_shape,
                            conv_cfg, -1, false, true, y_data_h, &y_shape);
    assert(y_shape.n == 1);
    assert(y_shape.c == 32);
    assert(y_shape.h == 122);
    assert(y_shape.w == 122);

    /* set algo=0, activation=true, tensor_core=false */
    cudnnConvForwardWrapper(x_data_h, x_shape, w_data_h, w_shape, b_data_h, b_shape,
                            conv_cfg, 0, true, false, y_data_h, &y_shape);                             
    assert(y_shape.n == 1);
    assert(y_shape.c == 32);
    assert(y_shape.h == 122);
    assert(y_shape.w == 122);

    free(x_data_h);
    free(w_data_h);
    free(b_data_h);
    free(y_data_h);

    return 0;
}