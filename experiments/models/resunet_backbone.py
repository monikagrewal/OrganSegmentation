import logging
from multiprocessing.sharedctypes import Value
from typing import Any, Callable, List, Literal, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from toolz.functoolz import pipe
from torch import Tensor
from torchvision import ops


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv3d:
    """3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetConfig():
    """
    This config holds the definition for the different ResNet backbone architectures.
    """

    def __init__(self,
                 architecture: Literal['resnet18', 'resnet34']
                ) -> None:
        if architecture == 'resnet18':
            self.depth = 4
            self.blocks = [2 for _ in range(self.depth)]
            self.inplanes = [64 * 2**i for i in range(self.depth)]
            self.strides = [1 if i==0 else 2 for i in range(self.depth)]
        elif architecture == 'resnet34':
            self.depth = 4
            self.blocks = [3, 4, 6, 3]
            self.inplanes = [64 * 2**i for i in range(self.depth)]
            self.strides = [1 if i==0 else 2 for i in range(self.depth)]
        if not(architecture in ['resnet18', 'resnet34']):
            raise ValueError(f"Specify a valid ResNet architecture instead of architecture = {architecture}")


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        stochastic_decay: float = 0.0
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.stochastic_decay = stochastic_decay

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = pipe(x,
                   self.bn1,
                   self.relu,
                   self.conv1,
                   self.bn2,
                   self.relu,
                   self.conv2,
        )

        out = ops.stochastic_depth(out, p=self.stochastic_decay,
                                mode="batch",
                                training=self.training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResUNet(nn.Module):
    """
    Implementation of U-Net with Residual blocks as described in paper:
	"Road extraction by deep residual U-Net"
    """

    def __init__(
        self,
        backbone: Literal['resnet18', 'resnet34'],
        in_channels: int = 1,
        out_channels: int = 2,
        stochastic_decay: float = 0.0,
    ):
        super(ResUNet, self).__init__()
        self.inplanes = in_channels
        self.outplanes = out_channels
        self.stochastic_decay = stochastic_decay

        self.architecture = ResNetConfig(backbone)

        self._norm_layer = nn.BatchNorm3d

        # Downsampling Path Layers
        self.downblocks = nn.ModuleList()
        for i in range(self.architecture.depth):
            self.downblocks.append(
                self._make_layer(self.architecture.inplanes[i],
                                 self.architecture.blocks[i],
                                 self.architecture.strides[i]
                                )
            )

        # Upsampling Path Layers
        self.upblocks = nn.ModuleList()
        for i in range(1, self.architecture.depth):
            self.inplanes = self.architecture.inplanes[-i] + self.architecture.inplanes[-i-1]
            self.upblocks.append(
                self._make_layer(self.architecture.inplanes[-i-1], self.architecture.blocks[-i-1])
            )

        # Last layer
        self.last_bn = self._norm_layer(self.inplanes)
        self.last_layer = nn.Conv3d(self.inplanes,  self.outplanes, kernel_size=1, stride=1, bias=False)

        # Initialization
        self.apply(self.weight_init)


    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        expansion: int = 1,
    ) -> nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = conv1x1(self.inplanes, planes, stride)

        layers = []
        layers.append(
            BasicBlock(
                self.inplanes, planes, stride, downsample,
                norm_layer=norm_layer,
                stochastic_decay=self.stochastic_decay
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    stochastic_decay=self.stochastic_decay
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Downsampling Path
        out = x
        down_features_list = list()
        for i in range(self.architecture.depth - 1):
            out = self.downblocks[i](out)
            down_features_list.append(out)

        # bottleneck
        out = self.downblocks[-1](out)

        # Upsampling Path
        for i in range(self.architecture.depth - 1):
            _, _, d, h, w = down_features_list[-1-i].shape
            out = F.interpolate(out, size=(d, h, w), mode='trilinear', align_corners=False)
            down_features = down_features_list[-1-i]
            out = torch.cat([down_features, out], dim=1)
            out = self.upblocks[i](out)

        out = self.last_bn(out)
        out = F.relu(out)
        out = self.last_layer(out)

        return out

    def inference(self, x):
        out = self.forward(x)
        return out

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


if __name__ == "__main__":
    model = ResUNet(backbone='resnet18', in_channels=1, out_channels=2).cuda()
    inputs = torch.rand(2, 1, 17, 128, 128).cuda()
    output = model(inputs)
    print(output.shape)
