import math
import numpy as np

import torch as T
import torch.nn as L
import torch.nn.init as I
import torch.nn.functional as F


class TrunkBlock(L.Module):
    def __init__(self, feat_in, feat_out):
        super(TrunkBlock, self).__init__()
        self.conv1 = L.Conv2d(feat_in, int(feat_out*1.), kernel_size=3, stride=1, padding=1, dilation=1)
        self.drop1 = L.Dropout2d(p=0.5, inplace=False)
        self.bn1 = L.BatchNorm2d(feat_in, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True)

        I.xavier_normal_(self.conv1.weight, gain=I.calculate_gain('relu'))
        I.constant_(self.conv1.bias, 0.0) # current
        
    def forward(self, x):
        return F.relu(self.conv1(self.drop1(self.bn1(x))))

class PreFilter(L.Module):
    def __init__(self):
        super(PreFilter, self).__init__()
        self.conv1 = L.Sequential(
            L.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            L.ReLU(inplace=True),
            L.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = L.Sequential(
            L.Conv2d(64, 192, kernel_size=5, padding=2),
            L.ReLU(inplace=True)
        )        
        
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        return c1,c2
    
class EncStage3(L.Module):
    def __init__(self, trunk_width=96):
        super(EncStage3, self).__init__()
        self.pre = PreFilter()
        self.conv3  = L.Conv2d(192, 128, kernel_size=3, stride=1, padding=0)
        self.drop1  = L.Dropout2d(p=0.5, inplace=False) ##
        self.bn1    = L.BatchNorm2d(192, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True) ##
        self.pool1  = L.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.tw = int(trunk_width)
        self.conv4  = TrunkBlock(128, 2*self.tw)
        self.conv5  = TrunkBlock(2*self.tw, 2*self.tw)

        I.xavier_normal_(self.conv3.weight, gain=I.calculate_gain('relu'))        
        I.constant_(self.conv3.bias, 0.0)
        
    def forward(self, x):
        c1,c2 = self.pre(x)
        c3 = (F.relu(self.conv3(self.drop1(self.bn1(c2))), inplace=False))
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        return [c1,c2,c3,c4,c5], c5

class Encoder(L.Module):
    def __init__(self, mu, trunk_width):
        super().__init__()
        self.mu = L.Parameter(T.from_numpy(mu), requires_grad=False) #.to(device)
        self.enc = EncStage3(trunk_width) 
    
    def forward(self, x):
        fmaps, h = self.enc(x)
        return x, fmaps, h


        
