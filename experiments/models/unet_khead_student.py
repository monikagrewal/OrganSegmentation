import logging
from distutils.command.config import config
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.nn.parameter import Parameter

from .unet_khead import KHeadUNet


class KHeadUNetStudent(KHeadUNet):
    """
    Implementation of U-Net with K-Head output layers
    """

    def __init__(
        self,
        teacher_weights_path: str,
        depth: int = 4,
        width: int = 64,
        growth_rate: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        k_heads: int = 5,
        threeD: bool = True,
    ):
        super().__init__(depth, width, growth_rate, in_channels, out_channels, k_heads, threeD)
        self.teacher = KHeadUNet(depth, width, growth_rate, in_channels, out_channels, k_heads, threeD,
                                    return_uncertainty=True, return_prediction=True)

        try:
            weights = torch.load(teacher_weights_path)["model"]
            self.teacher.load_state_dict(weights)
        except:
            logging.warning(f"Teacher weights not found at : {teacher_weights_path}"
                "So initializing with random.")
        self.teacher.eval()

    def forward(self, x):
        # randomly decide a head to train
        k_train = torch.randint(0, self.k_heads, (1,1))[0]

        # Freeze all heads except k_train
        self.unfreeze_heads()
        self.freeze_heads([k for k in range(self.k_heads) if k != k_train])

        out = self.unet_forward(x)
        outs = [layer(out) for layer in self.last_layer]
        # pick the unfrozen output for training
        final_out = outs[k_train]

        with torch.no_grad():
            mean_output, model_uncertainty, _ = self.teacher.inference(x)
            prediction = torch.argmax(mean_output, dim=1)

        return (final_out, model_uncertainty, prediction)
