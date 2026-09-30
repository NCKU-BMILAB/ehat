import sys
import torch.nn as nn
import numpy as np
import torch
import utils

class Net(nn.Module):

    def __init__(self, channel, classes):
        super(Net, self).__init__()

        self.channel = channel

        self.c1 = Down(self.channel, 64, 0)
        self.c2 = Down(64, 128, 0.2)
        self.c3 = Down(128, 256, 0.3)
        self.c4 = Down(256, 512, 0.3)
        self.c5 = Down(512, 1024, 0.4)
        self.b = ConvBlock(1024, 2048 // 2, 0.5)
        self.u1 = Up(2048, 1024 // 2, 0.5)
        self.u2 = Up(1024, 512 // 2, 0.4)
        self.u3 = Up(512, 256 // 2, 0.4)
        self.u4 = Up(256, 128 // 2, 0.3)
        self.u5 = Up(128, 64, 0.2)
        self.last = nn.Conv2d(in_channels = 64, out_channels = classes, kernel_size = 1)

    def forward(self, x):

        x, cat1 = self.c1(x)
        x, cat2 = self.c2(x)
        x, cat3 = self.c3(x)
        x, cat4 = self.c4(x)
        x, cat5 = self.c5(x)
        x = self.b(x)
        x = self.u1(x, cat5)
        x = self.u2(x, cat4)
        x = self.u3(x, cat3)
        x = self.u4(x, cat2)
        x = self.u5(x, cat1)
        x = self.last(x)
        return x

class ConvBlock(nn.Module):

    def __init__(self, Cin, Cout, drop_ratio = 0, mid = None):
        super(ConvBlock, self).__init__()
        if not mid:
            mid = Cout
        self.drop_ratio = drop_ratio
        self.relu= nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_channels = Cin, out_channels = mid, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = mid, out_channels = Cout, kernel_size = 3, padding = 1)
        if self.drop_ratio > 0:
            self.drop = nn.Dropout2d(drop_ratio)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.drop_ratio > 0:
            x = self.drop(x)
        return x

class Down(nn.Module):
    
    def __init__(self, Cin, Cout, drop_ratio):
        super(Down, self).__init__()

        self.drop_ratio = drop_ratio
        self.conv = ConvBlock(Cin, Cout)
        self.maxpool =  nn.MaxPool2d(2)
        if self.drop_ratio > 0:
            self.drop = nn.Dropout2d(drop_ratio)

    def forward(self, x):
        cat = self.conv(x)
        if self.drop_ratio > 0:
            x = self.maxpool(self.drop(cat))
        else:
            x = self.maxpool(cat)
        return x, cat

class Up(nn.Module):
    def __init__(self, Cin, Cout, drop_ratio):
        super(Up, self).__init__()
        
        self.drop_ratio = drop_ratio
        self.relu= nn.ReLU(inplace = True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(Cin, Cout, drop_ratio = 0, mid = Cin // 2)
        if self.drop_ratio > 0:
            self.drop = nn.Dropout2d(drop_ratio)
    
    def forward(self,x1, cat):
        x1 = self.up(x1)
        x1 = torch.cat([cat, x1], dim = 1)
        x1 = self.conv(x1)
        if self.drop_ratio > 0:
            x1 = self.drop(x1)

        return x1