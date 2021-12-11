#include "../../kernels/Add.cuh"

__host__ int main() {
    int NUM_THREADS = 128;
    int SIZE_TENSOR = 262144;

    float src_h[SIZE_TENSOR];
    float dest_h[SIZE_TENSOR];

    for (int i = 0; i < SIZE_TENSOR; i++) {
        src_h[i] = 1.0f;
        dest_h[i] = -1.0f;
    }

    fastAddWrapper(src_h, dest_h, SIZE_TENSOR, NUM_THREADS);

    // function test
    for (int i = 0; i < SIZE_TENSOR; i++) {
        assert(dest_h[i] == 0.0f);
    }

    // performance test was done in the choosing of kernels ...

    return 0;
}