from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.nn.parameter import Parameter

from .unet_khead import KHeadUNet


class KHeadUNetUncertainty(KHeadUNet):
    """
    Implementation of U-Net with K-Head output layers
    """

    def __init__(
        self,
        depth: int = 4,
        width: int = 64,
        **kwargs
    ):
        super().__init__(depth, width, **kwargs)
        # Replace last layer with multiple heads
        last_conv = nn.Conv3d if self.threeD else nn.Conv2d

        # In channels for last layer should be first of out_channesl
        current_in_channels = self.out_channels[0]

        # Data uncertainty
        self.data_uncertainty_layer = last_conv(current_in_channels, 1, kernel_size=1)

    def forward(self, x):
        # randomly decide a head to train
        k_train = torch.randint(0, self.k_heads, (1,1))[0]
        # Freeze all heads except k_train
        self.unfreeze_heads()
        self.freeze_heads([k for k in range(self.k_heads) if k != k_train])

        out = self.unet_forward(x)
        final_out = self.last_layer[k_train](out)
        data_uncertainty = self.data_uncertainty_layer(out)

        return final_out, data_uncertainty


    def inference(self, input):
        out = self.unet_forward(input)
        outs = [layer(out) for layer in self.last_layer]
        
        # stack on first dimension to get tensor with (BS x K x C x H x W x D)
        output = torch.stack(outs, dim=1)
        # calculate mean output as final prediction (BS x C x H x W x D)
        final_out = torch.mean(output, dim=1)

        # data uncertainty
        log_sigma_square = self.data_uncertainty_layer(out)
        data_uncertainty = torch.exp(log_sigma_square)
        
        # model_uncertainty = entropy of the predicted outputs over the k-heads
        model_uncertainty = self.calculate_entropy(outs)

        return final_out, data_uncertainty, model_uncertainty
