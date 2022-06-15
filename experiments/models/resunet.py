import logging
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
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
        survival_rate: float = 0.0
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
        self.survival_rate = survival_rate

    def forward(self, x: Tensor) -> Tensor:
        identity = x
 
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.training:
            decay_rate = round(1 - self.survival_rate, 1)
            out = ops.stochastic_depth(out, p=decay_rate, 
                                mode="batch",
                                training=self.training)
        else:
            out = self.survival_rate * out

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
        depth: int = 4,
        blocks_list: list = [1, 1, 1, 1],
        in_channels: int = 1,
        out_channels: int = 5,
        survival_rate: float = 1.0,
    ):
        """
        survival_rate refers to p_L (survival rate of the last block) in the paper
        "Deep networks with stochastic depth"
        """
        super(ResUNet, self).__init__()
        assert depth==len(blocks_list)
        self.depth = depth
        self.inplanes = in_channels
        self.outplanes = out_channels
        self.survival_rate = survival_rate
        
        inplanes_list = [64 * 2**i for i in range(self.depth)]
        stride_list = [1 if i==0 else 2 for i in range(self.depth)]
        L = sum(blocks_list)
        survival_rate_list = [round(1.0 - i/(L-1) * (1.0 - self.survival_rate), 1)
                                for i in range(L)]
        survival_rate_list2 = survival_rate_list[:-blocks_list[-1]]                    

        self._norm_layer = nn.BatchNorm3d

        # Downsampling Path Layers
        self.downblocks = nn.ModuleList()
        for i in range(self.depth):
            survival_rates = [survival_rate_list.pop(0) for i in range(blocks_list[i])]
            self.downblocks.append(
                self._make_layer(inplanes_list[i], blocks_list[i], 
                                stride=stride_list[i], survival_rates=survival_rates)
            )

        # Upsampling Path Layers
        self.upblocks = nn.ModuleList()
        for i in range(1, self.depth):
            survival_rates = [survival_rate_list2.pop() for i in range(blocks_list[i])]
            self.inplanes = inplanes_list[-i] + inplanes_list[-i-1]
            self.upblocks.append(
                self._make_layer(inplanes_list[-i-1], blocks_list[-i-1], 
                                survival_rates=survival_rates)
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
        survival_rates: list = [1.0]
    ) -> nn.Sequential:

        assert len(survival_rates)==blocks

        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = conv1x1(self.inplanes, planes, stride)

        layers = []
        layers.append(
            BasicBlock(
                self.inplanes, planes, stride, downsample,
                norm_layer=norm_layer,
                survival_rate=survival_rates[0]
            )
        )
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    survival_rate=survival_rates[i]
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Downsampling Path
        out = x
        down_features_list = list()
        for i in range(self.depth - 1):
            out = self.downblocks[i](out)
            down_features_list.append(out)

        # bottleneck
        out = self.downblocks[-1](out)

        # Upsampling Path
        for i in range(self.depth - 1):
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
    model = ResUNet(depth=4, in_channels=1, out_channels=2).cuda()
    inputs = torch.rand(2, 1, 17, 128, 128).cuda()
    output = model(inputs)
    print(output.shape)
