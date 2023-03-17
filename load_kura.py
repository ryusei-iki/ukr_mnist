import torch
import matplotlib.pyplot as plt
import numpy as np


def load_kura(data_num=200, type='numpy'):
    if (type == 'numpy'):
        x = np.zeros((data_num, 3))
        z = np.random.uniform(-1, 1, (data_num, 2))
        x[:, 2] = z[:, 0]**2 + z[:, 1]**2
        x[:, 0:2] = z
        x = x.astype(np.float32)
        x = torch.tensor(x)
    elif (type == 'tensor'):
        z = torch.rand(data_num, 2) * 2 - 1
        x = torch.zeros(data_num, 3)
        # x = np.zeros((data_num, 3))
        # z = np.random.uniform(-1, 1, (data_num, 2))
        x[:, 2] = z[:, 0]**2 + z[:, 1]**2
        x[:, 0:2] = z
        # x = x.astype(np.float32)
        # x = torch.tensor(x)
    return x


if '__main__' == __name__:
    x = load_kura()
    x = x.to('cpu').detach().numpy().copy()
    fig = plt.figure(figsize=(21, 13))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1], aspect='equal')

    ax1.scatter(x[:, 0], x[:, 1], x[:, 2])
    ax2.scatter(x[:, 0], x[:, 1])

    plt.savefig('kura.png')
