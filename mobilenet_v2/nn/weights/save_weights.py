import os

import onnx
from onnx.numpy_helper import to_array
from tqdm import trange


def npy2str(arr):
    dim = len(arr.shape)
    assert dim >= 1 and dim <= 4
    if dim == 1:
        return ' '.join(str(i) for i in arr)
    elif dim == 2:
        ret1 = []
        for arr1 in arr:
            ret1.append(' '.join(str(i) for i in arr1))
        return ' '.join(ret1)
    elif dim == 3:
        ret2 = []
        for arr2 in arr:
            ret1 = []
            for arr1 in arr2:
                ret1.append(' '.join(str(i) for i in arr1))
            ret2.append(' '.join(ret1))
        return ' '.join(ret2)
    else:
        ret3 = []
        for arr3 in arr:
            ret2 = []
            for arr2 in arr3:
                ret1 = []
                for arr1 in arr2:
                    ret1.append(' '.join(str(i) for i in arr1))
                ret2.append(' '.join(ret1))
            ret3.append(' '.join(ret2))
        return ' '.join(ret3)


def save_onnx_weights(input_path, output_path):
    os.makedirs(os.path.join(output_path, 'gemm'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'conv'), exist_ok=True)
    initializers = onnx.load(input_path).graph.initializer

    with open(os.path.join(output_path, 'gemm', '0.w'), 'w') as f:
        f.write(npy2str(to_array(initializers[0])))
    with open(os.path.join(output_path, 'gemm', '0.b'), 'w') as f:
        f.write(npy2str(to_array(initializers[1])))

    for i in trange((len(initializers) - 3) >> 1):
        with open(os.path.join(output_path, 'conv', f'{i}.w'), 'w') as f:
            f.write(npy2str(to_array(initializers[(i << 1) + 2])))
        with open(os.path.join(output_path, 'conv', f'{i}.b'), 'w') as f:
            f.write(npy2str(to_array(initializers[(i << 1) + 3])))


if __name__ == '__main__':
    save_onnx_weights('../onnx/mobilenet_v2.onnx', '.')
