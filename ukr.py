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
        for epoch in range(self.epochs):
            loss = self.E(self.z)
            print('{}epoch:{}'.format(epoch, loss))
            loss.backward()
            with torch.no_grad():
                self.z = self.z - self.eta*self.z.grad
            self.z.requires_grad = True
        return 0

    def f(self, z):
        d = (z[:, None, :] - self.z[None, :, :])**2
        d = torch.sum(d, dim=2)
        d = torch.exp(-1/(2*self.sigma**2)*d)
        x = torch.einsum('ij,jd->id', d, self.x)
        x = x / torch.sum(d, dim=1, keepdims=True)
        return x

    def E(self, z):
        x = self.f(z)
        loss = (self.x - x)**2
        loss = torch.sum(loss)/100
        return loss