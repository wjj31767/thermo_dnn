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
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
net = ThermoNet(18,64,16,'dense')
checkpoint = torch.load("model.pth")
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
data = THERMO('data/0.022000625/',False)
train_iter = Data.DataLoader(data, 400, shuffle=False)
input = []
output = []
ori_output = []
train_l_sum = 0.0
loss =nn.MSELoss()
for X,y in tqdm(train_iter):
    input.append(X)
    y_hat = net(X)
    output.append(y_hat[:,4])
    ori_output.append(y[:,4])
    l = loss(y_hat, y).sum()
    l.backward()
    train_l_sum += l.item()
print("loss sum: ",train_l_sum)
input = torch.cat(input)
output = torch.cat(output)
ori_output = torch.cat(ori_output)
print(ori_output)
# print(input.shape,output.shape,ori_output.shape)
im = output.detach().numpy()

# mean = data.summean[:,4]
# std = data.sumstd[:,4]
# im = im*std+mean
plt.hist(im)
plt.show()
print(im.min(),im.max())
im = ori_output.detach().numpy()
# mean = data.summean[:,4]
# std = data.sumstd[:,4]
# im = im*std+mean
plt.hist(im)
plt.show()
print(im.min(),im.max())
im = (output-ori_output).detach().numpy()
# mean = data.summean[:,4]
# std = data.sumstd[:,4]
# im = im*std+mean
plt.hist(im,805)
plt.show()
print(im.min(),im.max(),abs(im).min())

