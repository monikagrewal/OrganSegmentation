import logging
from distutils.command.config import config
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.nn.parameter import Parameter

from .unet import UNet
from .unet_khead import KHeadUNet


class UNetStudent(UNet):
    """
    Implementation of U-Net with K-Head output layers
    """

    def __init__(
        self,
        teacher_model_name: str,
        teacher_weights_path: str,
        depth: int = 4,
        width: int = 64,
        growth_rate: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        threeD: bool = True,
    ):
        super().__init__(depth, width, growth_rate, in_channels, out_channels, threeD)
        self.teacher_model_name = teacher_model_name
        if self.teacher_model_name=="unet":
            self.teacher = UNet(depth, width, growth_rate, in_channels, out_channels, threeD)
        elif self.teacher_model_name=="khead_unet":
            self.teacher = KHeadUNet(out_channels=out_channels, \
                                    return_uncertainty=True, return_prediction=True)
        else:
            raise ValueError(f"Unknown teacher name: {teacher_model_name}")

        try:
            weights = torch.load(teacher_weights_path)["model"]
            self.teacher.load_state_dict(weights)
        except:
            logging.warning(f"Teacher weights not found at : {teacher_weights_path}"
                "So initializing with random.")
        self.teacher.eval()

    def unet_forward(self, x):
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

        out = self.last_layer(out)

        return out
    
    def forward(self, x):
        final_out = self.unet_forward(x)

        with torch.no_grad():
            if self.teacher_model_name=="unet":
                output = self.teacher.inference(x)
                probs, prediction = torch.max(output, dim=1)
                outputs = (final_out, probs, prediction)
            elif self.teacher_model_name=="khead_unet":
                mean_output, model_uncertainty, _ = self.teacher.inference(x)
                prediction = torch.argmax(mean_output, dim=1)
                outputs = (final_out, model_uncertainty, prediction)

        return outputs
    
    def inference(self, x):
        return self.unet_forward(x)
