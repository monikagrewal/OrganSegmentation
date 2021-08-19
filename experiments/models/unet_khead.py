from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

from .unet import UNet


class KHeadUNet(UNet):
    """
    Implementation of U-Net with K-Head output layers
    """

    def __init__(
        self,
        depth: int = 4,
        width: int = 64,
        growth_rate: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        k_heads: int = 5,
        threeD: bool = True,
    ):
        super().__init__(depth, width, growth_rate, in_channels, out_channels, threeD)
        self.k_heads = k_heads

        # Replace last layer with multiple heads
        last_conv = nn.Conv3d if threeD else nn.Conv2d

        # In channels for last layer should be first of out_channesl
        current_in_channels = self.out_channels[0]
        self.last_layer = nn.ModuleList()
        for _ in range(self.k_heads):
            self.last_layer.append(
                last_conv(current_in_channels, out_channels, kernel_size=1),
            )
        self.last_layer.apply(self.weight_init)

    def forward(self, x):
        # Downsampling Path
        out = x
        down_features_list = list()
        for i in range(self.depth):
            out = self.downblocks[i](out)
            down_features_list.append(out)
            out = self.downsample(out)

        # bottleneck
        out = self.downblocks[-1](out)

        # Upsampling Path
        for i in range(self.depth):
            out = self.deconvblocks[i](out)
            down_features = down_features_list[-1 - i]
            # slice_diff = down_features.shape[2] - out.shape[2]
            # padd the slice dimension on one side to correct dimensions
            # if slice_diff == 1:
            # out = F.pad(out, [0,0,0,0,1,0])

            # pad slice and image dimensions if necessary
            down_shape = torch.tensor(down_features.shape)
            out_shape = torch.tensor(out.shape)
            shape_diff = down_shape - out_shape
            pad_list = [
                padding
                for diff in reversed(shape_diff.numpy())
                for padding in [diff, 0]
            ]
            if max(pad_list) == 1:
                out = F.pad(out, pad_list)

            out = torch.cat([down_features, out], dim=1)
            out = self.upblocks[i](out)

        outs = [layer(out) for layer in self.last_layer]

        # stack on first dimension to get tensor with (BS x K x H x W x D)
        return torch.stack(outs, dim=1)

    def freeze_heads(self, heads_to_freeze: List[int]) -> None:
        """Freeze heads in last layer by index

        Args:
            heads_to_freeze (List[int]): Indices of heads to freeze.
        """
        for k in heads_to_freeze:
            self.last_layer[k].weight.requires_grad = False
            self.last_layer[k].bias.requires_grad = False

        return

    def unfreeze_heads(self) -> None:
        """Unfreeze all heads in the last layer"""
        for k in range(self.k_heads):
            self.last_layer[k].weight.requires_grad = True
            self.last_layer[k].bias.requires_grad = True

        return
