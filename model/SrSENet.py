import torch
import torch.nn as nn
import torch.nn.init as init
from model.SrSEBlock import SrSEBlock


class Net(nn.Module):
    def __init__(self, blocks, rate, use_se=True):
        super(Net, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)

        self.conv_res = self._make_layer(SrSEBlock(64, use_se=use_se), blocks)

        self.conv_t = nn.Conv2d(in_channels=64, out_channels=64 * rate ** 2, kernel_size=1, stride=1, padding=1,
                                bias=False)
        self.subpixel = nn.PixelShuffle(rate)
        self.relu = nn.PReLU()

        self.convt_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.conv_res(out)
        out += residual
        out = self.relu(self.subpixel(self.conv_t(out)))
        out = self.convt_out(out)
        return out


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss
