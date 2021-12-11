#include "../../kernels/Im2Col.cuh"

__host__ int main() {
    int in_channel = 2;
    int in_height = 4;
    int in_width = 4;

    float im_data[32] = {-3, -3, 0, -3,
                         -1, 3, 1, -1,
                         2, -1, -1, 3,
                         -1, -2, -3, -3, 
                          
                         1, 0, 1, -1,
                         -3, 0, -2, 2,
                         3, -1, 0, 1,
                         1, -2, 2, 2};

    int padding = 1;
    int stride = 2;
    int kernel_size = 3;

    int out_height = (in_height - kernel_size + (padding << 1)) / stride + 1;
    int out_width = (in_width - kernel_size + (padding << 1)) / stride + 1;
    int col_h = in_channel * kernel_size * kernel_size;
    int col_w = out_height * out_width;
    int col_n = col_h * col_w;

    float col_data_ref[] = {0, 0, 0, 3,
                            0, 0, -1, 1,
                            0, 0, 3, -1,
                            0, -3, 0, -1,
                            -3, 0, 2, -1,
                            -3, -3, -1, 3,
                            0, 3, 0, -2,
                            -1, 1, -1, -3,
                            3, -1, -2, -3,
                            0, 0, 0, 0,
                            0, 0, -3, -2,
                            0, 0, 0, 2,
                            0, 0, 0, -1,
                            1, 1, 3, 0,
                            0, -1, -1, 1,
                            0, 0, 0, -2,
                            -3, -2, 1, 2,
                            0, 2, -2, 2};

    // cpu computation
    float *col_data_cpu = (float*)malloc(col_n * sizeof(float));

    im2col_cpu(
        im_data, in_channel, in_height, in_width, kernel_size, kernel_size,
        padding, padding, stride, stride, col_data_cpu
    );

    for (int i = 0; i < col_n; i++) {
        assert(col_data_cpu[i] == col_data_ref[i]);
    }

    // gpu computation
    float *col_data_gpu = (float*)malloc(col_n * sizeof(float));
    int block_size = 256;
    int grid_size = 0;

    im2col_gpu_wrapper(
        im_data, in_channel, in_height, in_width, kernel_size, kernel_size,
        padding, padding, stride, stride, col_data_gpu,
        block_size, grid_size
    );

    for (int i = 0; i < col_n; i++) {
        assert(col_data_gpu[i] == col_data_ref[i]);
    }

    free(col_data_cpu);
    free(col_data_gpu);  
    return 0;
}