import math
import os

import numpy as np
import onnxruntime as rt
from tqdm import trange


def compare_cudnn_onnx(inputs_path, outputs_path, num, onnx_model_path, rel_tol=1e-3):
    sess = rt.InferenceSession(onnx_model_path)
    sess_input_name = sess.get_inputs()[0].name
    sess_output_name = sess.get_outputs()[0].name

    for i in trange(num):
        input_data = np.load(os.path.join(inputs_path, f'{i}.npy'))
        preds_onnx = sess.run([sess_output_name], {
                              sess_input_name: input_data})[0][0]
        with open(os.path.join(outputs_path, f'{i}'), 'r') as f:
            cudnn_output_list = f.read().split(' ')[:-1]
            preds_cudnn = np.array([float(x) for x in cudnn_output_list])

        for pred_onnx, pred_cudnn in zip(preds_onnx, preds_cudnn):
            assert(math.isclose(pred_onnx, pred_cudnn, rel_tol=rel_tol))


if __name__ == '__main__':
    compare_cudnn_onnx('./inputs', './outputs', 10,
                       '../../../nn/onnx/mobilenet_v2.onnx', rel_tol=1e-2)
