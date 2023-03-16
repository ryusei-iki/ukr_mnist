import torch
import matplotlib.pyplot as plt


def load_kura(data_num=200, ):
    z = torch.rand(data_num, 2) * 2 - 1
    x = torch.zeros(data_num, 3)
    x[:, 2] = z[:, 0]**2 + z[:, 1]**2
    x[:, 0:2] = z
    return x


if '__main__' == __name__:
    x = load_kura()
    x = x.to('cpu').detach().numpy().copy()
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.scatter(x[:, 0], x[:, 1], x[:, 2])
    plt.savefig('kura.png')
