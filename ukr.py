import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.cuda.FloatTensor)


class ukr:
    def __init__(self, x, x_np, z, real_z, eta, sigma, epochs):
        self.x = x
        self.z = z
        self.x_np = x_np
        self.real_z = real_z
        self.eta = eta
        self.sigma = sigma
        self.epochs = epochs

    def train(self, ):
        fig = plt.figure(figsize=(21, 13))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1], aspect='equal')
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')

        for epoch in range(self.epochs):
            ax1.cla()
            ax2.cla()
            ax3.cla()
            x_estimate = self.f(self.z)
            loss = self.E(x_estimate)
            print('{}epoch:{}'.format(epoch, loss))
            loss.backward()
            with torch.no_grad():
                self.z = self.z - self.eta * self.z.grad
            self.z.requires_grad = True
            x_estimate_np = x_estimate.to('cpu').detach().numpy().copy()
            z_np = self.z.to('cpu').detach().numpy().copy()
            if (epoch % 10 == 0):
                z_wire = np.linspace(np.min(z_np), np.max(z_np), 20)
                z_wire_x, z_wire_y = np.meshgrid(z_wire, z_wire)
                z_wire_x = z_wire_x.reshape(-1, 1)
                z_wire_y = z_wire_y.reshape(-1, 1)
                z_wire = np.concatenate([z_wire_x, z_wire_y], axis=1)

                z_wire = z_wire.astype(np.float32)
                z_wire_tensor = torch.tensor(z_wire)
                x_wire = self.f(z_wire_tensor)
                x_wire = x_wire.to('cpu').detach().numpy().copy()
                x_wire = x_wire.reshape(20, 20, 3)
                ax1.scatter(x_estimate_np[:, 0], x_estimate_np[:, 1], x_estimate_np[:, 2], c=self.real_z[:, 0])
                ax1.scatter(self.x_np[:, 0], self.x_np[:, 1], self.x_np[:, 2], c='r')
                ax2.scatter(z_np[:, 0], z_np[:, 1], c=self.real_z[:, 0])
                ax3.plot_wireframe(x_wire[:, :, 0], x_wire[:, :, 1], x_wire[:, :, 2])
                ax3.scatter(self.x_np[:, 0], self.x_np[:, 1], self.x_np[:, 2], c='r')
                plt.savefig('output_data/{}.png'.format(epoch))
        return 0

    def f(self, z):
        d = (z[:, None, :] - self.z[None, :, :])**2
        d = torch.sum(d, dim=2)
        d = torch.exp(-1 / (2 * self.sigma**2) * d)
        x = torch.einsum('ij,jd->id', d, self.x)
        x = x / torch.sum(d, dim=1, keepdims=True)
        return x

    def E(self, x):
        loss = (self.x - x)**2
        loss = torch.sum(loss) / self.x.shape[0]
        return loss
