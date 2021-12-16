#include "../../kernels/Pool.cuh"

__host__ int main() {
    float x_data_h[65536];
    for (int i = 0; i < 65536; i++) {
        x_data_h[i] = i - 32768.0f;
    }
    TensorShape x_shape(1, 1280, 8, 8);

    float y_data_h[1280];
    TensorShape y_shape;

    fastPoolWrapper(x_data_h, &x_shape, y_data_h, &y_shape);

    // function test
    assert(y_shape->n == 1);
    assert(y_shape->c == 1280);
    assert(y_shape->h == 1);
    assert(y_shape->w == 1);

    float avg;
    for (int i = 0; i < 1280; i++) {
        avg = 0.0f;
        for (int j = 0; j < 64; j++) {
            avg += x_data_h[64 * i + j];
        }
        assert(y_data_h[i] == avg / 64.0f);
    }
   
    // performance test was done in the choosing of kernels ...

    return 0;
}