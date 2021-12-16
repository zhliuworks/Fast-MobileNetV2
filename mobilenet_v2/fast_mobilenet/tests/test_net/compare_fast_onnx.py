import math
import os

import numpy as np
import onnxruntime as rt
from tqdm import trange


def compare_fast_onnx(inputs_path, outputs_path, num, onnx_model_path, rel_tol=1e-3):
    sess = rt.InferenceSession(onnx_model_path)
    sess_input_name = sess.get_inputs()[0].name
    sess_output_name = sess.get_outputs()[0].name

    for i in trange(num):
        input_data = np.load(os.path.join(inputs_path, f'{i}.npy'))
        preds_onnx = sess.run([sess_output_name], {
                              sess_input_name: input_data})[0][0]
        with open(os.path.join(outputs_path, f'{i}'), 'r') as f:
            fast_output_list = f.read().split(' ')[:-1]
            preds_fast = np.array([float(x) for x in fast_output_list])

        for pred_onnx, pred_fast in zip(preds_onnx, preds_fast):
            if not math.isclose(pred_onnx, pred_fast, rel_tol=rel_tol):
                print('onnx: ', pred_onnx, '\t', 'ours: ', pred_fast, '\n')


if __name__ == '__main__':
    compare_fast_onnx('./inputs', './outputs', 10,
                       '../../../nn/onnx/mobilenet_v2.onnx', rel_tol=1.8)
