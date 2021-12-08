#include "../../wrappers/Add.cuh"

__host__ int main() {
    TensorShape shape(1, 3, 2, 2);

    float src[12] = {
        1.f, 3.f, 2.f, -1.f,
        4.f, 0.f, -2.f, 3.f,
        5.f, 1.f, 0.f, -1.f
    };

    float dest[12] = {
        -1.f, 2.f, 0.f, -1.f,
        -3.f, -5.f, 2.f, 0.f,
        -9.f, 2.f, 3.f, 1.f
    };

    float ref[12] = {
        0.f, 5.f, 2.f, -2.f,
        1.f, -5.f, 0.f, 3.f,
        -4.f, 3.f, 3.f, 0.f
    };

    cudnnAddWrapper(src, dest, shape);

    for (int i = 0; i < 12; i++) {
        assert(dest[i] == ref[i]);
    }

    return 0;
}