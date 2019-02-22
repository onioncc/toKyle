from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from region_loss import RegionLoss
from utils import *
from collections import OrderedDict


class ReLU4(nn.Module):
    def __init__(self,inplace = False):
        super(ReLU4,self).__init__()
    def forward(self, input):
        return F.hardtanh(input, 0,4)

class iSmart2DNN(nn.Module):
    def __init__(self):
        super(iSmart2DNN, self).__init__()
        self.width = int(320)
        self.height = int(160)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                ReLU4(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                ReLU4(inplace=True),
            )
        self.model = nn.Sequential(
            conv_dw( 3,  48, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 48,  96, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 96, 192, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw(192, 384, 1),
            conv_dw(384, 512, 1),
            nn.Conv2d(512, 10, 1, 1,bias=False),
        )
        self.loss = RegionLoss([1.4940052559648322, 2.3598481287086823,4.0113013115312155,5.760873975661669],2)
        self.anchors = self.loss.anchors
        self.num_anchors = self.loss.num_anchors
        self.anchor_step = self.loss.anchor_step
        self._initialize_weights()
    def forward(self, x):
        x = self.model(x)
        return x   
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
