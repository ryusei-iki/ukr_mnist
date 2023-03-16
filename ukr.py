import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.cuda.FloatTensor)


class ukr:
    def __init__(self, x, z, real_z, eta, sigma, epochs):
        self.x = x
        self.z = z
        self.real_z = real_z
        self.eta = eta
        self.sigma = sigma
        self.epochs = epochs

    def train(self, ):
        fig = plt.figure(figsize=(21, 13))
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1], aspect='equal')
        for epoch in range(self.epochs):
            ax1.cla()
            ax2.cla()
            x = self.f(self.z)
            loss = self.E(x)
            print('{}epoch:{}'.format(epoch, loss))
            loss.backward()
            with torch.no_grad():
                self.z = self.z - self.eta*self.z.grad
            self.z.requires_grad = True
            x_np = x.to('cpu').detach().numpy().copy()
            z_np = self.z.to('cpu').detach().numpy().copy()
            if (epoch % 10 == 0):
                ax1.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], c=self.real_z[:, 0])
                ax2.scatter(z_np[:, 0], z_np[:, 1], c=self.real_z[:, 0])
                plt.savefig('output_data/{}.png'.format(epoch))
        return 0

    def f(self, z):
        d = (z[:, None, :] - self.z[None, :, :])**2
        d = torch.sum(d, dim=2)
        d = torch.exp(-1/(2*self.sigma**2)*d)
        x = torch.einsum('ij,jd->id', d, self.x)
        x = x / torch.sum(d, dim=1, keepdims=True)
        return x

    def E(self, x):
        loss = (self.x - x)**2
        loss = torch.sum(loss)/self.x.shape[0]/self.x.shape[1]
        return loss
