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
            self.fc4 = nn.Linear(hidden_channels * 4, hidden_channels*2)
            self.fc5 = nn.Linear(hidden_channels*2, hidden_channels)
            self.fc6 = nn.Linear(hidden_channels, out_channels)
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
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        if self.mode == 'linear':

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = self.fc6(x)
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
    blk = nn.Sequential(nn.BatchNorm1d(in_channels),
                        nn.ReLU(),
                        nn.Conv1d(in_channels, out_channels,kernel_size=1))
    return blk




if __name__ == '__main__':
    # model = nn.BatchNorm1d(10)
    # X = torch.rand(4, 10,1)
    # print(model(X.squeeze()).shape)

    model = ThermoNet(18,64,16,mode = 'dense')
    X = torch.rand(4, 18,1)
    Y = model(X)
    print(Y.shape)
    X = torch.rand(4, 18,1)
    print(X[0,0,0])
    net = nn.Conv1d(18,16,kernel_size=1)
    print(net(X).shape)
