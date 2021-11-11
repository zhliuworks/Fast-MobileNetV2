'''
  PyTorch Implementation of MobileNetV2
  reference: https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
'''

import torch.nn as nn
import torch.onnx
import onnx
from onnx.numpy_helper import to_array
import onnxruntime as rt
import numpy as np
import math


def conv_bn(input, output, stride):
    return nn.Sequential(
        nn.Conv2d(input, output, 3, stride, 1, bias=True),
        # nn.BatchNorm2d(output),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(input, output):
    return nn.Sequential(
        nn.Conv2d(input, output, 1, 1, 0, bias=True),
        # nn.BatchNorm2d(output),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, input, output, stride, expansion_ratio):
        super(InvertedResidual, self).__init__()
        self.residual_connect = stride == 1 and input == output
        hidden_dim = input * expansion_ratio

        if expansion_ratio == 1:
            self.conv = nn.Sequential(
                ## depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                ## pointwise convolution, linear
                nn.Conv2d(hidden_dim, output, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(output),
            )
        else:
            self.conv = nn.Sequential(
                ## pointwise convolution, ReLU6
                nn.Conv2d(input, hidden_dim, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                ## depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                ## pointwise convolution, linear
                nn.Conv2d(hidden_dim, output, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(output),
            )

    def forward(self, x):
        if self.residual_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self, num_class=1000):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        inverted_residual_settings = [
            # t, c, n, s
            # t: expansion ratio
            # c: number of output channels
            # n: repeated times of identical layers
            # s: stride of first layer
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # build first conv2d
        self.features = [conv_bn(3, input_channel, 2)]

        # build several inverted residuals
        for expansion_ratio, output_channel, repeated_times, stride_first in inverted_residual_settings:
            for i in range(repeated_times):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, stride_first, expansion_ratio))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, expansion_ratio))
                input_channel = output_channel

        # build last several layers
        self.features.append(conv_1x1_bn(input_channel, last_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(last_channel, num_class)

        # initialize weights from given onnx file
        self.initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


    def initialize_weights(self):
        initializers = onnx.load('mobilenet_v2.onnx').graph.initializer

        # classifier
        assert self.classifier.weight.data.shape == torch.tensor(to_array(initializers[0])).shape
        assert self.classifier.bias.data.shape == torch.tensor(to_array(initializers[1])).shape
        self.classifier.weight.data = torch.tensor(to_array(initializers[0]))
        self.classifier.bias.data = torch.tensor(to_array(initializers[1]))

        # first conv2d
        assert self.features[0][0].weight.data.shape == torch.tensor(to_array(initializers[2])).shape
        assert self.features[0][0].bias.data.shape == torch.tensor(to_array(initializers[3])).shape
        self.features[0][0].weight.data = torch.tensor(to_array(initializers[2]))
        self.features[0][0].bias.data = torch.tensor(to_array(initializers[3]))

        # inverted residuals
        idx_inv_res = range(1, len(self.features) - 1)
        idx_init = 4
        for i in idx_inv_res:
            assert isinstance(self.features[i], InvertedResidual)
            for j in range(len(self.features[i].conv)):
                if isinstance(self.features[i].conv[j], nn.Conv2d):
                    assert self.features[i].conv[j].weight.data.shape == torch.tensor(to_array(initializers[idx_init])).shape
                    assert self.features[i].conv[j].bias.data.shape == torch.tensor(to_array(initializers[idx_init + 1])).shape
                    self.features[i].conv[j].weight.data = torch.tensor(to_array(initializers[idx_init]))
                    self.features[i].conv[j].bias.data = torch.tensor(to_array(initializers[idx_init + 1]))
                    idx_init += 2

        # last conv2d
        assert self.features[len(self.features) - 1][0].weight.data.shape == torch.tensor(to_array(initializers[idx_init])).shape
        assert self.features[len(self.features) - 1][0].bias.data.shape == torch.tensor(to_array(initializers[idx_init + 1])).shape
        self.features[len(self.features) - 1][0].weight.data = torch.tensor(to_array(initializers[idx_init]))
        self.features[len(self.features) - 1][0].bias.data = torch.tensor(to_array(initializers[idx_init + 1]))
        idx_init += 2

        assert idx_init == len(initializers) - 1


def compare_inference_results():
    input_data = np.random.randn(1, 3, 244, 244).astype(np.float32)

    sess_ref = rt.InferenceSession('mobilenet_v2.onnx')
    input_name = sess_ref.get_inputs()[0].name
    output_name = sess_ref.get_outputs()[0].name
    pred_ref = sess_ref.run([output_name], {input_name: input_data})[0]

    sess_my = rt.InferenceSession('mobilenet_v2_my.onnx')
    input_name = sess_my.get_inputs()[0].name
    output_name = sess_my.get_outputs()[0].name
    pred_my = sess_my.run([output_name], {input_name: input_data})[0]

    # comparing floating point numbers
    for val_ref, val_my in zip(pred_ref[0], pred_my[0]):
        if not math.isclose(val_ref, val_my, rel_tol=1e-4):
            print(f"not closely matched: {val_ref} {val_my}")


if __name__ == '__main__':
    net = MobileNetV2()
    dummy_input = torch.randn(1, 3, 244, 244)
    torch.onnx.export(net, dummy_input, "mobilenet_v2_my.onnx", input_names=['input.1'], output_names=['output.1'])
    compare_inference_results()
