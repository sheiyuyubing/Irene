import torch
import torch.nn as nn
import torch.nn.functional as F
from go import *


# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(15, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # self.convDrop1 = nn.Dropout2d(0.3)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        # self.convDrop2 = nn.Dropout2d(0.3)
        self.conv5 = nn.Conv2d(32, 1, 3, padding=1)

        self.bn0 = nn.BatchNorm2d(15)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(1)

        # 全连接层，将特征转换成100类输出。
        self.fc1 = nn.Linear(361,1)

    def forward(self, x):
        blank = x[:, 0]
        x = x.float()
        x = F.relu(self.bn0(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = x.view(-1, 19 * 19)

        # print(x.shape)
        x = torch.cat((x * blank.view(-1, 19 * 19), torch.ones((len(x), 1)).to(x.device) * 1e-50), dim=1)
        # 应用全连接
        # x = self.fc1(x)


        return x



# 快速策略网络
class PlayoutNetwork(nn.Module):
    def __init__(self):
        super(PlayoutNetwork, self).__init__()
        self.conv1 = nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(19 * 19, 19 * 19 + 1)

    def forward(self, x):
        blank = x[:, 0]
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(-1, 19 * 19)
        x = self.linear(x)
        x = torch.cat((x[:, :-1] * blank.view(-1, 19 * 19), x[:, -1:]), dim=1)
        x = F.log_softmax(x, dim=1)
        return x


# 价值网络，输入棋盘 features，输出胜率

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(2 * 19 * 19, 1)

    def forward(self, x):
        x = x.float()  # [B, 15, 19, 19]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)  # [B, 2, 19, 19]
        x = x.view(x.size(0), -1)  # Flatten: [B, 2*19*19]
        x = self.linear(x)         # [B, 1]
        x = x.view(-1)             # [B]
        x = torch.tanh(x)          # 输出范围 [-1, 1]
        return x
