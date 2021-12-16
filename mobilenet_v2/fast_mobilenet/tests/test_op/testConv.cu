#include "../../kernels/Conv.cuh"
// turn on  -> function test
// turn off -> performance test
#define FUNCTEST

__host__ int main() {
    int NUM_THREADS = 32;
    
#ifdef FUNCTEST
    /* function test */

    // test1: 1x1 conv + bias
    TensorShape x_shape_1(1, 3, 2, 2);
    float x_data_1[12] = {-3, -3, 0, -3, -1, 3, 1, -1, 2, -1, -1, 3};

    FilterShape w_shape_1(2, 3, 1, 1);
    float w_data_1[6] = {-1, -2, -3, -3, 1, 0};

    float b_data_1[2] = {1, -1};

    ConvConfig conv_cfg_1(0, 1, 1, 1);  // (padding, stride, dilation=1, group)

    TensorShape y_shape_1;  // infer the output shape
    float y_data_1[8];
    float y_ref_1[8] = {0, 1, 2, -3, 7, 11, 0, 7};

    fastConvWrapper(x_data_1, &x_shape_1, w_data_1, &w_shape_1, b_data_1,
                    &conv_cfg_1, false, y_data_1, &y_shape_1, NUM_THREADS);

    assert(y_shape_1.n == 1);
    assert(y_shape_1.c == 2);
    assert(y_shape_1.h == 2);
    assert(y_shape_1.w == 2);
    for (int i = 0; i < 8; i++) {
        assert(y_data_1[i] == y_ref_1[i]);
    }

    // test2: 1x1 conv + bias + relu6
    TensorShape x_shape_2(1, 3, 2, 2);
    float x_data_2[12] = {-3, -3, 0, -3, -1, 3, 1, -1, 2, -1, -1, 3};

    FilterShape w_shape_2(2, 3, 1, 1);
    float w_data_2[6] = {-1, -2, -3, -3, 1, 0};

    float b_data_2[2] = {1, -1};

    ConvConfig conv_cfg_2(0, 1, 1, 1);  // (padding, stride, dilation=1, group)

    TensorShape y_shape_2;  // infer the output shape
    float y_data_2[8];
    float y_ref_2[8] = {0, 1, 2, 0, 6, 6, 0, 6};

    fastConvWrapper(x_data_2, &x_shape_2, w_data_2, &w_shape_2, b_data_2,
                    &conv_cfg_2, true, y_data_2, &y_shape_2, NUM_THREADS);

    assert(y_shape_2.n == 1);
    assert(y_shape_2.c == 2);
    assert(y_shape_2.h == 2);
    assert(y_shape_2.w == 2);
    for (int i = 0; i < 8; i++) {
        assert(y_data_2[i] == y_ref_2[i]);
    }

    // test3: 3x3 conv + bias
    TensorShape x_shape_3(1, 2, 4, 4);
    float x_data_3[32] = {-3, -3, 0, -3, -1, 3, 1, -1, 2, -1, -1, 3, -1, -2, -3, -3, 1, 0, 1, -1, -3, 0, -2, 2, 3, -1, 0, 1, 1, -2, 2, 2};

    FilterShape w_shape_3(3, 2, 3, 3);
    float w_data_3[54] = {0, 1, 1, 0, 0, 0, -2, -2, 2, 3, 0, -3, -1, -2, -2, 3, 0, -1, 2, 0, -3, 3, -2, 3, -3, 2, 3, 1, -1, -3, -3, -1, 2, -1, -2, -3, 2, -1, 3, -2, 2, 1, -1, -3, 0, 0, 0, 2, 3, -3, 2, -2, 1, -1};

    float b_data_3[3] = {-2, 1, 0};

    ConvConfig conv_cfg_3(1, 2, 1, 1);  // (padding, stride, dilation=1, group)

    TensorShape y_shape_3;  // infer the output shape
    float y_data_3[12];
    float y_ref_3[12] = {4, -14, -4, -13, 10, -32, -21, 2, -12, -12, 8, 23};

    fastConvWrapper(x_data_3, &x_shape_3, w_data_3, &w_shape_3, b_data_3,
                    &conv_cfg_3, false, y_data_3, &y_shape_3, NUM_THREADS);

    assert(y_shape_3.n == 1);
    assert(y_shape_3.c == 3);
    assert(y_shape_3.h == 2);
    assert(y_shape_3.w == 2);
    for (int i = 0; i < 12; i++) {
        assert(y_data_3[i] == y_ref_3[i]);
    }

    // test4: 3x3 group conv + bias
    TensorShape x_shape_4(1, 3, 4, 4);
    float x_data_4[48] = {-3, -3, 0, -3, -1, 3, 1, -1, 2, -1, -1, 3, -1, -2, -3, -3, 1, 0, 1, -1, -3, 0, -2, 2, 3, -1, 0, 1, 1, -2, 2, 2, 0, 1, 1, 0, 0, 0, -2, -2, 2, 3, 0, -3, -1, -2, -2, 3};

    FilterShape w_shape_4(3, 1, 3, 3);
    float w_data_4[27] = {0, -1, 2, 0, -3, 3, -2, 3, -3, 2, 3, 1, -1, -3, -3, -1, 2, -1, -2, -3, 2, -1, 3, -2, 2, 1, -1};

    float b_data_4[3] = {-2, 1, 0};

    ConvConfig conv_cfg_4(1, 1, 1, 3);  // (padding, stride, dilation=1, group)

    TensorShape y_shape_4;  // infer the output shape
    float y_data_4[48];
    float y_ref_4[48] = {-14, 15, -11, 2, 16, -9, -24, 15, -1, 2, 11, -13, -9, -6, 5, 4, -8, 2, -5, 9, 20, 8, 3, -2, -10, -14, -1, 2, 12, 3, -10, -4, -2, 3, 2, -7, 1, 10, 2, -9, 1, 1, -4, 0, 1, -14, -22, 20};
    
    fastConvWrapper(x_data_4, &x_shape_4, w_data_4, &w_shape_4, b_data_4,
                    &conv_cfg_4, false, y_data_4, &y_shape_4, NUM_THREADS);

    assert(y_shape_4.n == 1);
    assert(y_shape_4.c == 3);
    assert(y_shape_4.h == 4);
    assert(y_shape_4.w == 4);
    for (int i = 0; i < 48; i++) {
        assert(y_data_4[i] == y_ref_4[i]);
    }

#else
    /* performance test */

    TensorShape x_shape(1, 32, 96, 96);
    float *x_data = (float*)malloc(32 * 96 * 96 * sizeof(float));

    FilterShape w_shape(96, 1, 3, 3);
    float *w_data = (float*)malloc(96 * 3 * 3 * sizeof(float));

    float *b_data = (float*)malloc(96 * sizeof(float));

    ConvConfig conv_cfg(1, 2, 1, 32);  // (padding, stride, dilation=1, group)

    TensorShape y_shape;
    float *y_data = (float*)malloc(96 * 48 * 48 * sizeof(float));

    /* im2col, relu6 */
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float elapsed_time;
    int repeated_times = 10;

    cudaEventRecord(start);
    for (int i = 0; i < repeated_times; i++) {
        fastConvWrapper(x_data, &x_shape, w_data, &w_shape, b_data,
                        &conv_cfg, true, y_data, &y_shape, NUM_THREADS);
    }
    cudaEventRecord(end);

    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("im2col elapsed time %f ms\n", elapsed_time / repeated_times);
    /*
    block_size = 256
    - NUM_THREADS = 8:  1396.377319 ms
    - NUM_THREADS = 16: 1399.162231 ms
    - NUM_THREADS = 32: 1403.039429 ms
    - ... trivial impact

    NUM_THREADS = 32
    - block_size = 64:  1403.547485 ms
    - block_size = 128: 1396.180908 ms
    - block_size = 256: 1403.039429 ms
    - block_size = 512: 1408.702393 ms
    - ... trivial impact
    */

    assert(y_shape.n == 1);
    assert(y_shape.c == 96);
    assert(y_shape.h == 48);
    assert(y_shape.w == 48);

#endif
    return 0;
}