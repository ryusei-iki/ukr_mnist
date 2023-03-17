import torch
from load_kura import load_kura
from ukr import ukr
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# 学習データの設定
data_num = 300

# 学習パラメータの設定
eta = 1
sigma = 0.3
epochs = 400
z_dim = 2
z = torch.normal(mean=0, std=0.001, size=(data_num, z_dim), requires_grad=True)
fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0], aspect='equal')
z_np = z.to('cpu').detach().numpy().copy()
ax.scatter(z_np[:, 0], z_np[:, 1])
plt.savefig('base.png')
# 保存dictの設定
history = {}
# history[]

x = load_kura(data_num)
x_np = x.to('cpu').detach().numpy().copy()
real_z = x_np[:, :2]

model = ukr(x, x_np, z, real_z, eta, sigma, epochs)
history = model.train()
