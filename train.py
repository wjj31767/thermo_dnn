
if __name__ == '__main__':
    from torch.nn import init
    from model import ThermoNet
    import datetime
    import torch
    import numpy as np
    from torch import nn
    from torch.nn import functional as F
    import torch.optim as optim
    import torch.utils.data as Data
    from dataset import THERMO
    from tqdm import tqdm
    from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
    import math
    import os
    import sys
    try:
        net = ThermoNet(18,64,16,'linear').cuda()
        optimizer = optim.Adam(net.parameters(), lr=0.000001)

        if os.path.exists('model.pth'):
            checkpoint = torch.load("model.pth")
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            global_train_l = checkpoint['loss']
        else:
            for params in net.parameters():
                init.normal_(params, mean=0, std=0.05)
            global_train_l = sys.maxsize
        train_iter = Data.DataLoader(THERMO('data/',False), 3995, shuffle=True,pin_memory=True)
        print(global_train_l)
        for param_group in optimizer.param_groups:
            print("learning rate",param_group['lr'])
        loss =nn.MSELoss()


        # scheduler = ReduceLROnPlateau(optimizer,patience=1,verbose=True,factor=0.5)
        # scheduler = StepLR(optimizer,step_size=100,gamma=0.4,verbose=True)
        def evaluate_accuracy(data_iter, net):
            acc_sum, n = 0.0, 0
            for X, y in data_iter:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                n += y.shape[0]
            return acc_sum / n

        num_epochs = 50000

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
                print("save model",epoch+1,train_l_sum,datetime.datetime.now())
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'loss': train_l_sum,
                    'optimizer': optimizer.state_dict(),
                    }, 'model.pth')
                global_train_l = train_l_sum
            # scheduler.step()
            # print('epoch %d, loss %f'
            #       % (epoch + 1, train_l_sum))

    except KeyboardInterrupt:
        print("keyinter")
        if train_l_sum < global_train_l:
            print("save model", epoch + 1, train_l_sum, datetime.datetime.now())
            torch.save({
                'model_state_dict': net.state_dict(),
                'loss': train_l_sum,
                'optimizer': optimizer.state_dict(),
            }, 'model.pth')
            global_train_l = train_l_sum
