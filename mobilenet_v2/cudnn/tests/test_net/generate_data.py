import os

import numpy as np
from tqdm import trange

np.random.seed(123)


def generate_data(data_path, num):
    for i in trange(num):
        np_data = np.random.randn(1, 3, 244, 244).astype(np.float32)
        np.save(os.path.join(data_path, f'{i}.npy'), np_data)
        arr = np_data.reshape(-1)
        seq = ' '.join([str(x) for x in arr])
        with open(os.path.join(data_path, f'{i}'), 'w') as f:
            f.write(seq)


if __name__ == '__main__':
    os.makedirs('./inputs', exist_ok=True)
    os.makedirs('./outputs', exist_ok=True)
    generate_data('./inputs', 10)
