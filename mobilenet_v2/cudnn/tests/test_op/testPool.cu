#include "../../wrappers/Pool.cuh"

__host__ int main() {
    // input
    TensorShape x_shape(1, 2, 3, 3);
    float x_data_h[18] = {2, 0, 1, -3, -2, 0, 2, -3, -3, -2, 1, 2, 1, 3, -2, -1, 1, 3};

    // global average pooling config
    // PoolConfig(int window_size, int padding, int stride)
    PoolConfig pool_cfg(3, 0, 1);

    // output
    TensorShape y_shape;  // let cuDNN infer the output shape
    float y_data_h[2];
    float y_ref[2] = {-2.0f/3.0f, 2.0f/3.0f};
    
    // test forward
    cudnnPoolForwardWrapper(x_data_h, x_shape, pool_cfg, y_data_h, &y_shape);
    
    assert(y_shape.n == 1);
    assert(y_shape.c == 2);
    assert(y_shape.h == 1);
    assert(y_shape.w == 1);
    assert(y_data_h[0] == y_ref[0]);
    assert(y_data_h[1] == y_ref[1]);

    return 0;
}