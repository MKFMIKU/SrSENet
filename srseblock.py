import torch.nn as nn
from selayer import SELayer

class SrSEBlock(nn.Module):
    def __init__(self, planes, stride=1, downsample=None):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.se = SELayer(planes)
        #self.relu_out = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        out = self.se(out)

        out += residual
        #out = self.relu_out(out)

        return out
