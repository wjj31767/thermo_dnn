from torch.nn import init
from model import ThermoNet
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as Data
from dataset import THERMO
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
net = ThermoNet(18,64,16,'dense').cuda()
train_iter = Data.DataLoader(THERMO('data/0.022000625/'), 512, shuffle=True)
for params in net.parameters():
    init.normal_(params, mean=0, std=0.02)

loss =nn.MSELoss()


optimizer = optim.Adam(net.parameters(), lr=0.0003)
# scheduler = ReduceLROnPlateau(optimizer,patience=1,verbose=True,factor=0.5)
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

num_epochs = 200
global_test_acc = 0
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    net.train()
    loss_sum = 0.0
    for X, y in tqdm(train_iter):
        X = X.cuda()
        y = y.cuda()
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        loss_sum += l
        # 梯度清零
        optimizer.zero_grad()
        l.backward()

        optimizer.step()  # “softmax回归的简洁实现”一节将用到

        train_l_sum += l.item()
        n += y.shape[0]
    # if test_acc>global_test_acc:
    #     torch.save(net.state_dict(),"model.pth")
    #     global_test_acc = test_acc
    # scheduler.step(loss_sum)
    print('epoch %d, loss %.4f'
          % (epoch + 1, train_l_sum / n))

