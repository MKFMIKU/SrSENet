import torch
import torch.nn as nn
import numpy as np
from srseblock import SrSEBlock
import torch.nn.init as init

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

class Net(nn.Module):
    def __init__(self, blocks):
        super(Net, self).__init__()
        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(SrSEBlock(64), blocks)

        self.Transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu_transpose = nn.LeakyReLU(0.2, inplace=True) 
        
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                init.orthogonal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def make_layer(self, block, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        """放大LR"""
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
