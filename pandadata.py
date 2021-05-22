import pandas as pd
import matplotlib.pyplot as plt
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

import glob
import os
norm = 'rescale'
testdata = pd.read_csv('testData.csv')
net = ThermoNet(18,64,16,'dense')
ckpt_list = glob.glob(str('*checkpoint_epoch_*.pth'))
if len(ckpt_list) > 0:
    ckpt_list.sort(key=os.path.getmtime)
    checkpoint = torch.load(ckpt_list[-2])
    net.load_state_dict(checkpoint['model_state'])
    global_train_l = checkpoint['loss']
    print(global_train_l)
net.eval()
data = THERMO('data/',norm)

CH4 = []
CO2 = []
H2O = []
O2 = []
N2 = []
T = []
RRCH4 = []
RRH2O = []
RRO2 = []
RRCO2 = []
Points0 = []
RRCH4_predict = []
RRH2O_predict = []
for i in range(testdata.shape[0]):
    CH4.append(testdata.CH4[i])
    CO2.append(testdata.CO2[i])
    H2O.append(testdata.H2O[i])
    O2.append(testdata.O2[i])
    N2.append(testdata.N2[i])
    T.append(testdata['T'][i])
    RRCH4.append(testdata['RR.CH4'][i])
    RRH2O.append(testdata['RR.H2O'][i])
    # RRO2.append(testdata['RR.O2'][i])
    # RRCO2.append(testdata['RR.CO2'][i])
    Points0.append(testdata['Points:0'][i])

input = torch.zeros((testdata.shape[0],18,1),dtype=torch.float64)
input[:,4,:] = torch.from_numpy(np.array(CH4)).view(-1,1)
input[:,5,:] = torch.from_numpy(np.array(CO2)).view(-1,1)
input[:,9,:] = torch.from_numpy(np.array(H2O)).view(-1,1)
input[:,15,:] = torch.from_numpy(np.array(O2)).view(-1,1)
input[:,13,:] = torch.from_numpy(np.array(N2)).view(-1,1)
input[:,17,:] = torch.from_numpy(np.array(T)).view(-1,1)
if norm=='stand':
    mask = data.mask[:18]
    input = input.squeeze()
    print(input[:,mask].shape,mask.shape,data.summean[:,:18][:,mask].shape)
    input[:,mask] = (input[:,mask] - data.summean[:,:18][:,mask]) / data.sumstd[:,:18][:,mask]
    input = input.type(torch.float32)
elif norm == 'rescale':
    mask = data.mask[:18]
    input = input.squeeze()
    input[:, mask] = (input[:, mask] - data.summin[:, :18][:, mask]) / (
                data.summax[:, :18][:, mask] - data.summin[:, :18][:, mask])
    input = input.type(torch.float32)
else:
    input = input.type(torch.float32).squeeze()
input = input.unsqueeze(-1)
print(input.shape)
output = net(input)
output = output.detach().numpy()
if norm=='stand':
    output = output*data.sumstd[:,18:]+data.summean[:,18:]
elif norm=='rescale':
    output = output*(data.summax[:,18:]-data.summin[:,18:])+data.summin[:,18:]
# self.dict = {"CH": 0,
#              "CH2": 1,
#              "CH2O": 2,
#              "CH3": 3,
#              "CH4": 4,
#              "CO": 5,
#              "CO2": 6,
#              "H": 7,
#              "H2": 8,
#              "H2O": 9,
#              "H2O2": 10,
#              "HCO": 11,
#              "HO2": 12,
#              "N2": 13,
#              "O": 14,
#              "O2": 15,
#              "OH": 16,
#              "T": 17,
#                     "RR.CH":0,
#                     "RR.CH2":1,
#                     "RR.CH2O":2,
#                     "RR.CH3":3,
#                     "RR.CH4":4,
#                     "RR.CO":5,
#                     "RR.CO2":6,
#                     "RR.H":7,
#                     "RR.H2":8,
#                     "RR.H2O":9,
#                     "RR.H2O2":10,
#                     "RR.HCO":11,
#                     "RR.HO2":12,
#                     "RR.O":13,
#                     "RR.O2":14,
#                     "RR.OH":15}

plt.plot(Points0,RRCH4,label='original')
plt.plot(Points0,output[:,4],label='predict')
plt.title("CH4")
plt.legend()
plt.show()

# plt.plot(Points0,RRCO2,label='original')
# if norm:
#     plt.plot(Points0,output[:,6].detach().numpy()*data.sumstd[0][22]+data.summean[0][22],label='predict')
# else:
#     plt.plot(Points0,output[:,6].detach().numpy(),label='predict')
# plt.title("CO2")
# plt.legend()
# plt.show()

plt.plot(Points0,RRH2O,label='original')
plt.plot(Points0,output[:,9],label='predict')
plt.title("H2O")
plt.legend()
plt.show()

# plt.plot(Points0,RRO2,label='original')
# if norm:
#     plt.plot(Points0,output[:,14].detach().numpy()*data.sumstd[0][22]+data.summean[0][22],label='predict')
# else:
#     plt.plot(Points0,output[:,14].detach().numpy(),label='predict')
# plt.title("O2")
# plt.legend()
# plt.show()