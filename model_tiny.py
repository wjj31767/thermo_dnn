import torch
import torch.hub as hub
import torch.nn as nn
import torch.nn.functional as F

class ThermoNet(nn.Module):
    """
    Implements LiLaNet model from
    `"Boosting LiDAR-based Semantic Labeling by Cross-Modal Training Data Generation"
    <https://arxiv.org/abs/1804.09915>`_.

    Arguments:
        num_classes (int): number of output classes
    """

    def __init__(self, in_channels,hidden_channels,out_channels, mode='linear'):
        super(ThermoNet, self).__init__()
        self.mode=mode
        if mode=='linear':
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, hidden_channels*2)
            self.fc3 = nn.Linear(hidden_channels * 2, hidden_channels*4)
            self.fc4 = nn.Linear(hidden_channels * 4, hidden_channels*8)
            self.fc5 = nn.Linear(hidden_channels * 8, hidden_channels*16)
            self.fc6 = nn.Linear(hidden_channels * 16, hidden_channels*32)
            self.fc7 = nn.Linear(hidden_channels * 32, hidden_channels*16)
            self.fc8 = nn.Linear(hidden_channels * 16, hidden_channels*8)
            self.fc9 = nn.Linear(hidden_channels * 8, hidden_channels*4)
            self.fc10 = nn.Linear(hidden_channels * 4, hidden_channels*2)
            self.fc11 = nn.Linear(hidden_channels * 2, hidden_channels)
            self.fc12 = nn.Linear(hidden_channels, out_channels)
        elif mode == 'dense':
            self.dense1 = DenseBlock(2,in_channels,hidden_channels)
            self.dense2 = DenseBlock(2,self.dense1.out_channels,hidden_channels)
            self.dense3 = DenseBlock(2,self.dense2.out_channels,hidden_channels)
            self.dense4 = DenseBlock(2,self.dense3.out_channels,hidden_channels)
            self.dense5 = DenseBlock(2,self.dense4.out_channels,hidden_channels)
            # self.dense6 = DenseBlock(2,self.dense5.out_channels,out_channels)
            # self.dense7 = DenseBlock(2, self.dense6.out_channels, out_channels)
            # self.dense8 = DenseBlock(2, self.dense7.out_channels, out_channels)
            # self.dense9 = DenseBlock(2, self.dense8.out_channels, out_channels)
            self.lin = nn.Sequential()
            outdense_channels = self.dense5.out_channels
            i = 0
            while outdense_channels//2>out_channels:
                self.lin.add_module('linear'+str(i),nn.Linear(outdense_channels,outdense_channels//2))
                outdense_channels//=2
                i+=1
            else:
                self.lin.add_module('linear'+str(i),nn.Linear(outdense_channels,out_channels))
        elif mode == 'resnet':
            self.res1 = resnet_block(in_channels,in_channels,2,True)
            self.res2 = resnet_block(in_channels,hidden_channels,2)
            self.res3 = resnet_block(hidden_channels,hidden_channels*2,2)
            self.res4 = resnet_block(hidden_channels*2,hidden_channels*4,2)
            self.res5 = resnet_block(hidden_channels*4,hidden_channels*8,2)
            self.res6 = resnet_block(hidden_channels*8,hidden_channels*16,2)

            self.lin = nn.Sequential()
            outdense_channels = hidden_channels*16
            i = 0
            while outdense_channels // 2 > out_channels:
                self.lin.add_module('linear' + str(i), nn.Linear(outdense_channels, outdense_channels // 2))
                outdense_channels //= 2
                i += 1
            else:
                self.lin.add_module('linear' + str(i), nn.Linear(outdense_channels, out_channels))
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = x.squeeze()

        if self.mode == 'linear':
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = F.leaky_relu(self.fc3(x))
            x = F.leaky_relu(self.fc4(x))
            x = F.leaky_relu(self.fc5(x))
            x = F.leaky_relu(self.fc6(x))
            x = F.leaky_relu(self.fc7(x))
            x = F.leaky_relu(self.fc8(x))
            x = F.leaky_relu(self.fc9(x))
            x = F.leaky_relu(self.fc10(x))
            x = F.leaky_relu(self.fc11(x))
            x = self.fc12(x)
        elif self.mode == 'dense':
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.dense4(x)
            x = self.dense5(x)
            # x = self.dense6(x)
            # x = self.dense7(x)
            # x = self.dense8(x)
            # x = self.dense9(x)

            x = self.lin(x.squeeze())
        elif self.mode == 'resnet':
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            x = self.res5(x)
            x = self.res6(x)
            # x = self.dense7(x)
            # x = self.dense8(x)
            # x = self.dense9(x)

            x = self.lin(x.squeeze())
        return x
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.Linear(in_channels, out_channels),
                        nn.LeakyReLU(),
                        )
    return blk



class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        if use_1x1conv:
            self.conv3 = nn.Linear(in_channels, out_channels)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.conv1(X))
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


if __name__ == '__main__':
    # model = nn.BatchNorm1d(10)
    # X = torch.rand(4, 10,1)
    # print(model(X.squeeze()).shape)
    from torchsummary import summary
    model = ThermoNet(6,8,4,mode = 'resnet').cuda()
    print(summary(model, (6, 1)))
