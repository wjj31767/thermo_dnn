from model import ThermoNet
import torch
import torch.utils.data as Data
from dataset_tiny import THERMO
from tqdm import tqdm
import glob
import os
print(torch.__version__)
save_CPP = True
norm = 'rescale'
net = ThermoNet(6,8,4,'dense')
ckpt_list = glob.glob(str('*checkpoint_epoch_*.pth'))
if len(ckpt_list) > 0:
    ckpt_list.sort(key=os.path.getmtime)
    checkpoint = torch.load(ckpt_list[-1])
    net.load_state_dict(checkpoint['model_state'])
    global_train_l = checkpoint['loss']
    print(global_train_l)
data = THERMO('data/',norm)
train_iter = Data.DataLoader(data, 19472, shuffle=False)
for X,y in tqdm(train_iter):
    y_hat = net(X)
    if save_CPP:
        traced_script_module = torch.jit.trace(net, X)
        traced_script_module.save("model.pt")
        break
