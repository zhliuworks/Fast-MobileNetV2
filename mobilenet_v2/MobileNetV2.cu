#include "cudakernels/Conv.cuh"
#include "cudakernels/Gemm.cuh"
#include "cudakernels/ReduceMean.cuh"
#include "nn/MobileNetV2.h"
#include <cudnn.h>

__host__ int main() {
    // compare our cuda kernels and cudnn in the inference of MobileNetV2
    // accuracy and performance count
    return 0;
}