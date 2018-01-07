import torch
import torch.nn as nn
import numpy as np
import math
from srseblock import SrSEBlock

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convt_I1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=16, stride=8, padding=4, bias=False)

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.relu_input = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F1 = self.make_layer(SEBlock(64), 8)

        self.Transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=16, stride=8, padding=4, bias=False)
        self.relu_transpose = nn.LeakyReLU(0.2, inplace=True) 
        
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, -1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def make_layer(self, block, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        """放大LR"""
        x = self.sub_mean(x)
        convt_I1 = self.convt_I1(x)
    
        out = self.conv_input(x) 

        """利用SE-ResNet进行特征提取"""  
        convt_F1 = self.convt_F1(out)

        """放大提取到的feature map"""
        convt_out = self.relu_transpose(self.Transpose(convt_F1)) 
        convt_R1 = self.convt_R1(convt_out)

        HR = convt_I1 + convt_R1
        return HR
        
class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss
