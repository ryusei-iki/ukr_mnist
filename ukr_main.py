import torch
import numpy as np
import matplotlib.pyplot as plt
from load_kura import load_kura
from ukr import ukr

# 学習データの設定
data_num = 200

# 学習パラメータの設定
eta = 1
sigma = 0.1

# 保存dictの設定
history = {}
# history[]

x = data_num
x_np = x.to('cpu').detach().numpy().copy()
z = x_np[:, :2]

model = ukr(x, z, eta, sigma)
history = model.train()

