#include "../wrappers/Pool.cuh"

__host__ int main() {
    // input
    TensorShape x_shape(1, 1024, 7, 7);
    float *x_data_h;
    x_data_h = (float*)malloc(1 * 1024 * 7 * 7 * sizeof(float));

    // convolution config
    // ConvConfig(int padding, int stride, int dilation, int group)
    ConvConfig conv_cfg(1, 2, 1, 1);

    // global average pooling config
    // PoolConfig(int window_size, int padding, int stride)
    PoolConfig pool_cfg(7, 0, 1);

    // output
    TensorShape y_shape(-1, -1, -1, -1);  // let cuDNN infer the output shape
    float *y_data_h;
    y_data_h = (float*)malloc(1 * 1024 * 1 * 1 * sizeof(float));

    // test forward
    cudnnPoolForwardWrapper(x_data_h, x_shape, pool_cfg, y_data_h, &y_shape);
    
    assert(y_shape.n == 1);
    assert(y_shape.c == 1024);
    assert(y_shape.h == 1);
    assert(y_shape.w == 1);

    free(x_data_h);
    free(y_data_h);
    
    return 0;
}