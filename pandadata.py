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

norm = False
testdata = pd.read_csv('testData.csv')
net = ThermoNet(18,64,16,'linear')
checkpoint = torch.load("model.pth")
net.load_state_dict(checkpoint['model_state_dict'])
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
    Points0.append(testdata['Points:0'][i])

input = torch.zeros((testdata.shape[0],18,1),dtype=torch.float64)
input[:,4,:] = torch.from_numpy(np.array(CH4)).view(-1,1)
input[:,5,:] = torch.from_numpy(np.array(CO2)).view(-1,1)
input[:,9,:] = torch.from_numpy(np.array(H2O)).view(-1,1)
input[:,15,:] = torch.from_numpy(np.array(O2)).view(-1,1)
input[:,13,:] = torch.from_numpy(np.array(N2)).view(-1,1)
input[:,17,:] = torch.from_numpy(np.array(T)).view(-1,1)
if norm:
    mask = data.mask[:18]
    input = input.squeeze()
    print(input[:,mask].shape,mask.shape,data.summean[:,:18][:,mask].shape)
    input[:,mask] = (input[:,mask] - data.summean[:,:18][:,mask]) / data.sumstd[:,:18][:,mask]
    input = input.type(torch.float32)
else:
    input = input.type(torch.float32).squeeze()
output = net(input)

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
#                     "RR.CH":18,
#                     "RR.CH2":19,
#                     "RR.CH2O":20,
#                     "RR.CH3":21,
#                     "RR.CH4":22,
#                     "RR.CO":23,
#                     "RR.CO2":24,
#                     "RR.H":25,
#                     "RR.H2":26,
#                     "RR.H2O":27,
#                     "RR.H2O2":28,
#                     "RR.HCO":29,
#                     "RR.HO2":30,
#                     "RR.O":31,
#                     "RR.O2":32,
#                     "RR.OH":33}

plt.plot(Points0,RRCH4,label='original')
if norm:
    plt.plot(Points0,output[:,4].detach().numpy()*data.sumstd[0][22]+data.summean[0][22],label='predict')
else:
    plt.plot(Points0,output[:,4].detach().numpy(),label='predict')

plt.legend()
plt.show()
