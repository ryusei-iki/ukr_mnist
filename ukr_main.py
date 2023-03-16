import torch
from load_kura import load_kura
from ukr import ukr
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# 学習データの設定
data_num = 100

# 学習パラメータの設定
eta = 0.1
sigma = 0.1
epochs = 100
z_dim = 2
z = torch.normal(mean=0, std=0.05, size=(data_num, z_dim), requires_grad=True)
# 保存dictの設定
history = {}
# history[]

x = load_kura(data_num)
x_np = x.to('cpu').detach().numpy().copy()
real_z = x_np[:, :2]

model = ukr(x, z, real_z, eta, sigma, epochs)
history = model.train()
