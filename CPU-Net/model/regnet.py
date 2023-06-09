import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 64, (3, 3, 3))
        self.mp3d_1 = nn.MaxPool3d(kernel_size=2)
        self.bn3d_1 = nn.BatchNorm3d(64)

        self.conv3d_2 = nn.Conv3d(64, 64, (3, 3, 3))
        self.mp3d_2 = nn.MaxPool3d(kernel_size=2)
        self.bn3d_2 = nn.BatchNorm3d(64)

        self.conv3d_3 = nn.Conv3d(64, 128, (3, 3, 3))
        self.mp3d_3 = nn.MaxPool3d(kernel_size=2)
        self.bn3d_3 = nn.BatchNorm3d(128)

        self.conv3d_4 = nn.Conv3d(128, 256, (3, 3, 3))
        self.mp3d_4 = nn.MaxPool3d(kernel_size=2)
        self.bn3d_4 = nn.BatchNorm3d(256)

        self.mp3d_5 = nn.AdaptiveAvgPool3d(output_size=1)
        self.fl3d = nn.Flatten()

        self.conv2d_1 = nn.Conv2d(1, 16, 3)
        self.mp2d_1 = nn.MaxPool2d(kernel_size=2)
        self.bn2d_1 = nn.BatchNorm2d(16)

        self.conv2d_2 = nn.Conv2d(16, 32, 3)
        self.mp2d_2 = nn.MaxPool2d(kernel_size=2)
        self.bn2d_2 = nn.BatchNorm2d(32)

        self.conv2d_3 = nn.Conv2d(32, 64, 3)
        self.mp2d_3 = nn.MaxPool2d(kernel_size=2)
        self.bn2d_3 = nn.BatchNorm2d(64)

        self.conv2d_4 = nn.Conv2d(64, 128, 3)
        self.mp2d_4 = nn.MaxPool2d(kernel_size=2)
        self.bn2d_4 = nn.BatchNorm2d(128)

        self.mp2d_5 = nn.AdaptiveAvgPool2d(output_size=1)
        self.fl2d = nn.Flatten()

        self.l1 = nn.Linear(384, 512)
        # self.dp1 = nn.Dropout(p=0.3)
        self.l2 = nn.Linear(512, 6)

    def forward(self, x, y):
        x = F.relu(self.conv3d_1(x))
        x = self.mp3d_1(x)
        x = self.bn3d_1(x)
        x = F.relu(self.conv3d_2(x))
        x = self.mp3d_2(x)
        x = self.bn3d_2(x)
        x = F.relu(self.conv3d_3(x))
        x = self.mp3d_3(x)
        x = self.bn3d_3(x)
        x = F.relu(self.conv3d_4(x))
        x = self.mp3d_4(x)
        x = self.bn3d_4(x)
        x = self.mp3d_5(x)
        x = self.fl3d(x)


        y = F.relu(self.conv2d_1(y))
        y = self.mp2d_1(y)
        y = self.bn2d_1(y)
        y = F.relu(self.conv2d_2(y))
        y = self.mp2d_2(y)
        y = self.bn2d_2(y)
        y = F.relu(self.conv2d_3(y))
        y = self.mp2d_3(y)
        y = self.bn2d_3(y)
        y = F.relu(self.conv2d_4(y))
        y = self.mp2d_4(y)
        y = self.bn2d_4(y)
        y = self.mp2d_5(y)
        y = self.fl2d(y)

        res = torch.cat((x, y), dim=1)
        res = self.l1(res)
        # res = self.dp1(res)
        res = self.l2(res)
        return res

