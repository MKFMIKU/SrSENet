import torch.nn as nn
from selayer import SELayer

class SrSEBlock(nn.Module):
    def __init__(self, planes, stride=1):
        super(SrSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.se = SELayer(planes)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out = self.se(out)


        out += residual
        
        return out
