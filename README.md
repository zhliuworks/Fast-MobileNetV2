# Fast-MobileNetV2
Optimized CUDA Kernels for Fast MobileNetV2 Inference

## Logs

Updated:  2021/12/16

**TODO:** 

- [ ] Winograd conv kernel
- [ ] Inference error problem
- [ ] Comparison of ours and cudnn, and overall optimization -- `main.cu`

## Develop Steps

- [x] 1⃣️  Implement MobileNetV2 with PyTorch, and parse the given ONNX model with Python to analyze the network structure. ---  [`mobilenet_v2/nn/onnx/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/nn/onnx)
- [x] 2⃣️  Implement MobileNetV2 with C++ (only sequential layer structures and weights, no forward computation), and parse the given ONNX model with Python to extract the weights. --- [`mobilenet_v2/nn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/nn)
- [x] 3⃣️  Implement wrappers and tests for cuDNN/cuBLAS primitives: Conv, Gemm, and Pool. --- [`mobilenet_v2/cudnn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/cudnn)
  * Here, Gemm can be implemented using cuBLAS, or seen as 1x1 Conv2d using cuDNN, we take the former way)
- [x] 4⃣️  Implement cuDNN-accelerated MobileNetV2 with wrappers and C++ network implemented above. --- [`mobilenet_v2/cudnn/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/cudnn)
- [x] 5⃣️  Implement and optimize CUDA kernels: Conv, Gemm, and Pool. --- [`mobilenet_v2/fast_mobilenet/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/fast_mobilenet)
  * Here, Conv can be implemented using *Im2Col + Gemm*, or *Winograd Algorithm* (now only the former)
- [x] 6⃣️  Implement our Fast-MobileNetV2 as a whole. --- [`mobilenet_v2/fast_mobilenet/`](https://github.com/zhliuworks/Fast-MobileNetV2/tree/master/mobilenet_v2/fast_mobilenet)
- [ ] 7⃣️  Compare and Optimize: e.g. parameters tuning, model-specific / hardware-specific optimization, ...

## Test Steps

#### nn

* Re-implement MobileNetV2 ONNX model with PyTorch and test inference:

  ```shell
  (conda) >> cd mobilenet_v2/nn/onnx/
  (conda) >> python pytorchMobileNetV2.py
  ```

* Save weights in MobileNetV2 ONNX model to plain-text files:

  ```shell
  (conda) >> cd mobilenet_v2/nn/weights/
  (conda) >> python save_weights.py
  ```

* Show MobileNetV2 topology in C++ and check loaded weights:

  ```shell
  >> cd mobilenet_v2/nn/examples/
  >> make show
  >> ./show.out
  >> make check
  >> ./check.out
  ```

#### cudnn

* Show version of CUDA and CUDNN:

  ```shell
  >> cd mobilenet_v2/cudnn/
  >> bash version.sh
  ```

* Operator tests:

  ```shell
  >> cd mobilenet_v2/cudnn/tests/test_op/
  >> make
  >> ./testConv.o
  >> ./testGemm.o
  >> ./testPool.o
  >> ./testAdd.o
  ```

* Network test:

  ```shell
  (conda) >> cd mobilenet_v2/cudnn/tests/test_net/
  (conda) >> python generate_data.py
  (conda) >> conda deactivate
  >> make
  >> ./testCudnnMobileNetV2.o
  >> source ~/.bashrc
  (conda) >> python compare_cudnn_onnx.py
  ```

#### our kernels

* Operator tests:

  ```shell
  >> cd mobilenet_v2/fast_mobilenet/tests/test_op/
  >> make
  >> ./testConv.o
  >> ./testGemm.o
  >> ./testPool.o
  >> ./testAdd.o
  >> ./testIm2Col.o
  ```

* Network test:

  ```shell
  (conda) >> cd mobilenet_v2/fast_mobilenet/tests/test_net/
  (conda) >> python generate_data.py
  (conda) >> conda deactivate
  >> make
  >> ./testFastMobileNetV2.o
  >> source ~/.bashrc
  (conda) >> python compare_fast_onnx.py
  ```

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
