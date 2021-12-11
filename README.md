# Fast-MobileNetV2
Optimized CUDA Kernels for Fast MobileNetV2 Inference

## Logs

Updated:  2021/12/11

TODO: 

* Winograd conv kernel
* Build the whole network using our lego kernel blocks
* Overall optimization
* Presentation

## Develop Steps

- [x] 1⃣️  Implement MobileNetV2 with PyTorch, and parse the given ONNX model with Python to analyze the network structure. ---  [`mobilenet_v2/nn/onnx/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/nn/onnx)
- [x] 2⃣️  Implement MobileNetV2 with C++ (only sequential layer structures and weights, no forward computation), and parse the given ONNX model with Python to extract the weights. --- [`mobilenet_v2/nn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/nn)
- [x] 3⃣️  Implement wrappers and tests for cuDNN/cuBLAS primitives: Conv, Gemm, and Pool. --- [`mobilenet_v2/cudnn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/cudnn)
  * Here, Gemm can be implemented using cuBLAS, or seen as 1x1 Conv2d using cuDNN, we take the former way)
- [x] 4⃣️  Implement cuDNN-accelerated MobileNetV2 with wrappers and C++ network implemented above. --- [`mobilenet_v2/cudnn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/cudnn)
- [x] 5⃣️  Implement and optimize CUDA kernels: Conv, Gemm, and Pool. --- [`mobilenet_v2/fast_mobilenet/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/fast_mobilenet)
  * Here, Conv can be implemented using *Im2Col + Gemm*, or *Winograd Algorithm* (now only the former)
- [ ] 6⃣️  Implement our Fast-MobileNetV2 as a whole, and compare with cuDNN version. --- [`mobilenet_v2/fast_mobilenet/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/fast_mobilenet)
- [ ] 7⃣️  Optimization: e.g. parameters tuning, model-specific / hardware-specific optimization, ...

## Test Environment

* NVIDIA Tesla V100 GPU
* CUDA version 10.2.89
* CUDNN version 8.2.4
* Run Python source of this repo in an Anaconda environment, and we have Python version 3.9.7
* Do **NOT** Run CUDA source of this repo in an Anaconda environment

## Tech Stack

* [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
* ONNX Python API
* cuDNN and cuBLAS API
* CUDA C++ Programming
* GPU Architecture and Compiler Optimization

## Reference

[1]  Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*. 2018.

[2]  NVIDIA Corporation. "NVIDIA cuDNN Documentation." *available at: https://docs.nvidia.com/deeplearning/cudnn/api/index.html*

[3] NVIDIA Corporation. "NVIDIA cuBLAS Documentation." *available at: https://docs.nvidia.com/cuda/cublas/index.html*

[4] Lavin, Andrew, and Scott Gray. "Fast algorithms for convolutional neural networks." *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*. 2016.

[5] Mark Harris. "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops." *available at: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/*

[6] Mark Harris. "Optimizing Parallel Reduction in CUDA." *available at: https://vuduc.org/teaching/cse6230-hpcta-fa12/slides/cse6230-fa12--05b-reduction-notes.pdf*
