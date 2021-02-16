# MODIFIED FROM
# https://github.com/google-research/vision_transformer/blob/master/vit_jax/models_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as pjoin
from collections import OrderedDict


def weight_standardize(w, dim, eps):
    """Subtracts mean and divides by standard deviation."""
    w = w - torch.mean(w, dim=dim)
    w = w / (torch.std(w, dim=dim) + eps)
    return w


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = weight_standardize(self.weight, [0, 1, 2], 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(in_channels, out_channels, stride=1, groups=1, bias=False):
    return StdConv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=bias,
                     groups=groups)


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return StdConv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 mid_channels=None,
                 stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or out_channels // 4

        self.gn1 = nn.GroupNorm(32, mid_channels, eps=1e-6)
        self.conv1 = conv1x1(in_channels, mid_channels, bias=False)
        self.gn2 = nn.GroupNorm(32, mid_channels, eps=1e-6)
        self.conv2 = conv3x3(mid_channels, mid_channels, stride,
                             bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv3 = conv1x1(mid_channels, out_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or in_channels != out_channels):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(in_channels,
                                      out_channels,
                                      stride,
                                      bias=False)
            self.gn_proj = nn.GroupNorm(out_channels, out_channels)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.downsample = 16  # four stride=2 conv2d layer

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(
            OrderedDict([('conv',
                          StdConv2d(3,
                                    width,
                                    kernel_size=7,
                                    stride=2,
                                    bias=False,
                                    padding=3)),
                         ('gn', nn.GroupNorm(32, width, eps=1e-6)),
                         ('relu', nn.ReLU(inplace=True)),
                         ('pool',
                          nn.MaxPool2d(kernel_size=3, stride=2, padding=0))]))

        self.body = nn.Sequential(
            OrderedDict([
                ('block1',
                 nn.Sequential(
                     OrderedDict([('unit1',
                                   PreActBottleneck(in_channels=width,
                                                    out_channels=width * 4,
                                                    mid_channels=width))] +
                                 [(f'unit{i:d}',
                                   PreActBottleneck(in_channels=width * 4,
                                                    out_channels=width * 4,
                                                    mid_channels=width))
                                  for i in range(2, block_units[0] + 1)], ))),
                ('block2',
                 nn.Sequential(
                     OrderedDict([('unit1',
                                   PreActBottleneck(in_channels=width * 4,
                                                    out_channels=width * 8,
                                                    mid_channels=width * 2,
                                                    stride=2))] +
                                 [(f'unit{i:d}',
                                   PreActBottleneck(in_channels=width * 8,
                                                    out_channels=width * 8,
                                                    mid_channels=width * 2))
                                  for i in range(2, block_units[1] + 1)], ))),
                ('block3',
                 nn.Sequential(
                     OrderedDict([('unit1',
                                   PreActBottleneck(in_channels=width * 8,
                                                    out_channels=width * 16,
                                                    mid_channels=width * 4,
                                                    stride=2))] +
                                 [(f'unit{i:d}',
                                   PreActBottleneck(in_channels=width * 16,
                                                    out_channels=width * 16,
                                                    mid_channels=width * 4))
                                  for i in range(2, block_units[2] + 1)], ))),
            ]))

    def forward(self, x):
        x = self.root(x)
        x = self.body(x)
        return x


def resnet50():
    return ResNetV2(block_units=(3, 4, 9), width_factor=1)
