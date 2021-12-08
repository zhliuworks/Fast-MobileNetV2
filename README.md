# Fast-MobileNetV2
Optimized CUDA Kernels for Fast MobileNetV2 Inference

Updated:  2021/12/08

## Develop Steps

- [x] 1⃣️  Implement MobileNetV2 with PyTorch, and parse the given ONNX model with Python to analyze the network structure. ---  [`mobilenet_v2/nn/onnx/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/nn/onnx)
- [x] 2⃣️  Implement MobileNetV2 with C++ (only sequential layer structures and weights, no forward computation), and parse the given ONNX model with Python to extract the weights. --- [`mobilenet_v2/nn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/nn)
- [x] 3⃣️  Implement wrappers and tests for cuDNN/cuBLAS primitives: Conv, Gemm, and Pool. --- [`mobilenet_v2/cudnn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/cudnn)
- [x] 4⃣️  Implement cuDNN-accelerated MobileNetV2 with wrappers and C++ network implemented above. --- [`mobilenet_v2/cudnn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/cudnn)
- [ ] 5⃣️  Implement and optimize CUDA kernels: Conv, Gemm, and Pool. --- [`mobilenet_v2/fast_mobilenet/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/fast_mobilenet)
- [ ] 6⃣️  Implement and optimize our Fast-MobileNetV2 as a whole, and compare with cuDNN. --- [`mobilenet_v2/fast_mobilenet/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/fast_mobilenet)

## Test Environment

* Tesla V100 GPU
* `CUDA version`  10.2.89
* `CUDNN version`  8.2.4
* `cublas_v2.h`
* Run Python source of this repo in a conda environment, and we have `Python version` 3.9.7
* Do **NOT** Run CUDA source of this repo in a conda environment

## Tech Stack

* [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
* ONNX Python API
* cuDNN and cuBLAS API
* CUDA C++ Programming
* GPU Architecture and Compiler Optimization

## Reference

[1]  Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

[2]  NVIDIA Corporation. "NVIDIA cuDNN Documentation." *available at: https://docs.nvidia.com/deeplearning/cudnn/api/index.html*

[3] NVIDIA Corporation. "NVIDIA cuBLAS Documentation." *available at: https://docs.nvidia.com/cuda/cublas/index.html*
