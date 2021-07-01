from torch.nn import init
from model_tiny import ThermoNet
import datetime
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as Data
from dataset_tiny import THERMO
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import math
import os
import sys
import glob
def checkpoint_state(model=None, optimizer=None, epoch=None,loss=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None



    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state,'loss':loss}
def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu
def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
if __name__ == '__main__':

    max_ckpt_save_num = 4
    net = ThermoNet(6,8,4,'dense').cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    ckpt_list = glob.glob(str('*checkpoint_epoch_*.pth'))
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        checkpoint = torch.load(ckpt_list[-1])
        net.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        global_train_l = checkpoint['loss']
    else:
        for params in net.parameters():
            init.normal_(params, mean=0, std=0.05)
        global_train_l = sys.maxsize
    train_iter = Data.DataLoader(THERMO('data/','rescale'), 19472, shuffle=True,pin_memory=True)
    print(global_train_l)
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.001
        print("learning rate",param_group['lr'])
    loss =nn.MSELoss()


    scheduler = ReduceLROnPlateau(optimizer,patience=300,verbose=True,factor=0.5)
    # scheduler = StepLR(optimizer,step_size=100,gamma=0.4,verbose=True)
    def evaluate_accuracy(data_iter, net):
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    num_epochs = 500000

    for epoch in range(num_epochs):
        train_l_sum , n = 0.0, 0
        net.train()
        for i,(X, y) in enumerate(train_iter):
            X = X.cuda()
            y = y.cuda()
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            optimizer.zero_grad()
            l.backward()

            optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            n += y.shape[0]
        if train_l_sum<global_train_l:
            ckpt_list = glob.glob(str( 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)

            if ckpt_list.__len__() >= max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])

            ckpt_name =  'checkpoint_epoch_%d' % epoch
            save_checkpoint(
                checkpoint_state(net, optimizer, epoch,train_l_sum), filename=ckpt_name,
            )
            print("save model",epoch+1,train_l_sum,datetime.datetime.now())
            # torch.save({
            #     'model_state_dict': net.state_dict(),
            #     'loss': train_l_sum,
            #     'optimizer': optimizer.state_dict(),
            #     }, 'model.pth')
            global_train_l = train_l_sum
        # scheduler.step(train_l_sum)
        # print('epoch',epoch+1, 'loss',train_l_sum)
