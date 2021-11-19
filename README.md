# Fast-MobileNetV2
Optimized CUDA Kernels for Fast MobileNetV2 Inference

Updated:  2021/11/20

## Develop Steps

- [x] 1⃣️  Implement MobileNetV2 with PyTorch, and parse the given ONNX model with Python to analyze the network structure. ---  [`parse_onnx/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/parse_onnx)
- [x] 2⃣️  Implement MobileNetV2 with C++ (only sequential layer structures and weights, no forward computation), and parse the given ONNX model with Python to extract the weights. --- [`mobilenet_v2/nn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/nn)
- [x] 3⃣️  Implement wrappers and tests for cuDNN/cuBLAS primitives: Conv, Gemm, and Pool. --- [`mobilenet_v2/cudnn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/cudnn)
- [ ] 4⃣️  Implement cuDNN-accelerated MobileNetV2 with wrappers and C++ network implemented above. --- [`mobilenet_v2/cudnn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/cudnn)
- [ ] 5⃣️  Implement and optimize CUDA kernels: Conv, Gemm, and Pool. --- [`mobilenet_v2/fast_mobilenet/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/fast_mobilenet)
- [ ] 6⃣️  Implement and optimize our Fast-MobileNetV2 as a whole, and compare with cuDNN. --- [`mobilenet_v2/fast_mobilenet/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/fast_mobilenet)

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
