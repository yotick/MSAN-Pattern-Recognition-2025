import torch
from torch import nn


class Eca_layer(nn.Module):
    def __init__(self, channel, k_size=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.max_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=5, last=nn.ReLU):
        super().__init__()
        if kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        elif kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            last(),
        )

    def forward(self, x):
        return self.main(x)


class SoftAttn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_attn = SpatialAttention(in_channels)
        self.channel_attn = Eca_layer(in_channels)
        self.conv = ConvLayer(in_channels, in_channels, 3)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y

