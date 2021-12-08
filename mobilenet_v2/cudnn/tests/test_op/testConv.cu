#include "../../wrappers/Conv.cuh"
#include <vector>

__host__ int main() {
    // input
    TensorShape x_shape(1, 2, 3, 3);
    float x_data_h[18] = {2, 0, 1, -3, -2, 0, 2, -3, -3, -2, 1, 2, 1, 3, -2, -1, 1, 3};

    // filter
    FilterShape w_shape(2, 2, 2, 2);
    float w_data_h[16] = {2, -1, 1, 0, 1, -1, 1, 2, 3, -1, 1, -2, -2, 3, -3, 3};

    // bias
    int b_shape = 2;
    float b_data_h[2] = {1, -1};

    // convolution config
    // ConvConfig(int padding, int stride, int dilation, int group)
    ConvConfig conv_cfg(0, 1, 1, 1);

    // output
    TensorShape y_shape;  // let cuDNN infer the output shape
    float y_data_h[8];
    std::vector<float> y_ref;

    // test forward
    bool print_algo = false;
    
    /* search algo, activation=false */
    cudnnConvForwardWrapper(x_data_h, x_shape, w_data_h, w_shape, b_data_h, b_shape,
                            conv_cfg, -1, false, false, y_data_h, &y_shape, print_algo);
    assert(y_shape.n == 1);
    assert(y_shape.c == 2);
    assert(y_shape.h == 2);
    assert(y_shape.w == 2);
    y_ref = {6.0f, -4.0f, -2.0f, 6.0f, 19.0f, -15.0f, 13.0f, -10.0f};
    for (int i = 0; i < 8; i++) {
        assert(y_data_h[i] == y_ref[i]);
    }

    /* search algo, activation=true */
    cudnnConvForwardWrapper(x_data_h, x_shape, w_data_h, w_shape, b_data_h, b_shape,
                            conv_cfg, -1, true, false, y_data_h, &y_shape, print_algo);
    assert(y_shape.n == 1);
    assert(y_shape.c == 2);
    assert(y_shape.h == 2);
    assert(y_shape.w == 2);
    y_ref = {6.0f, 0.0f, 0.0f, 6.0f, 6.0f, 0.0f, 6.0f, 0.0f};
    for (int i = 0; i < 8; i++) {
        assert(y_data_h[i] == y_ref[i]);
    }

    /* algo=0, activation=true */
    cudnnConvForwardWrapper(x_data_h, x_shape, w_data_h, w_shape, b_data_h, b_shape,
                            conv_cfg, 0, true, false, y_data_h, &y_shape, print_algo);
    assert(y_shape.n == 1);
    assert(y_shape.c == 2);
    assert(y_shape.h == 2);
    assert(y_shape.w == 2);
    y_ref = {6.0f, 0.0f, 0.0f, 6.0f, 6.0f, 0.0f, 6.0f, 0.0f};
    for (int i = 0; i < 8; i++) {
        assert(y_data_h[i] == y_ref[i]);
    }

    return 0;
}