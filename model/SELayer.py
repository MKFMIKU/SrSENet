from torch import nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, 16),
            nn.PReLU(),
            nn.Linear(16, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = self.avg_pool(x).view(b, c)
        excitation = self.fc(squeeze).view(b, c, 1, 1)
        out = x * excitation
        return out
