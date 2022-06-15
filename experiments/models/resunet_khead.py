from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.nn.parameter import Parameter

from .resunet import ResUNet


class KHeadResUNet(ResUNet):
    """
    Implementation of ResU-Net with K-Head output layers
    """

    def __init__(
        self,
        k_heads: int = 5,
        return_uncertainty: bool = False,
        return_prediction: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.k_heads = k_heads
        self.return_uncertainty = return_uncertainty
        self.return_prediction = return_prediction
        inplanes_list = [64 * 2**i for i in range(self.depth)]

        # Replace last layer with multiple heads
        # In channels for last layer should be first of out_channesl
        self.last_layer = nn.ModuleList()
        for _ in range(self.k_heads):
            self.last_layer.append(
                nn.Conv3d(inplanes_list[0], self.outplanes, kernel_size=1, stride=1, bias=False),
            )
        self.last_layer.apply(self.weight_init)

    def forward(self, x):
        # randomly decide a head to train
        k_train = torch.randint(0, self.k_heads, (1,1))[0]
        # Freeze all heads except k_train
        self.unfreeze_heads()
        self.freeze_heads([k for k in range(self.k_heads) if k != k_train])

        out = self.resunet_forward(x)
        outs = [layer(out) for layer in self.last_layer]
        # pick the unfrozen output for training
        final_out = outs[k_train]

        outputs = final_out
        if self.return_uncertainty:
            # stack on first dimension to get tensor with (BS x K x C x D x H x W)
            output = torch.stack(outs, dim=1)
            # calculate probs
            output = torch.softmax(output, dim=2)
            # calculate mean probs as final prediction (BS x C x D x H x W)
            mean_out = torch.mean(output, dim=1)
            model_uncertainty = self.calculate_entropy(mean_out)
            outputs = [final_out, model_uncertainty]

        if self.return_prediction:
            output = torch.stack(outs, dim=1)
            mean_output = torch.mean(output, dim=1)
            prediction = torch.argmax(mean_output, dim=1)
            if isinstance(outputs, List):  #if return_uncertainty is True
                outputs.append(prediction)
            else:
                outputs = [final_out, prediction]

        return outputs

    def resunet_forward(self, x):
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
            down_features = down_features_list[-1 - i]
            out = torch.cat([down_features, out], dim=1)
            out = self.upblocks[i](out)

        out = self.last_bn(out)
        out = F.relu(out)
     
        return out

    def freeze_heads(self, heads_to_freeze: List[int]) -> None:
        """Freeze heads in last layer by index

        Args:
            heads_to_freeze (List[int]): Indices of heads to freeze.
        """
        for k in heads_to_freeze:
            self.last_layer[k].weight.requires_grad = False
            if self.last_layer[k].bias is not None:
                self.last_layer[k].bias.requires_grad = False

        return

    def unfreeze_heads(self) -> None:
        """Unfreeze all heads in the last layer"""
        for k in range(self.k_heads):
            self.last_layer[k].weight.requires_grad = True
            if self.last_layer[k].bias is not None:
                self.last_layer[k].bias.requires_grad = True

        return

    def inference(self, input, return_raw=False):
        out = self.resunet_forward(input)
        outs = [layer(out) for layer in self.last_layer]
        
        # stack on first dimension to get tensor with (BS x K x C x D x H x W)
        output = torch.stack(outs, dim=1)
        # calculate probs
        output_softmax = torch.softmax(output, dim=2)
        # calculate mean probs as final prediction (BS x C x D x H x W)
        mean_out = torch.mean(output_softmax, dim=1)
        
        model_uncertainty = self.calculate_entropy(mean_out)
        if return_raw:
            mean_out = torch.mean(output, dim=1)
        return mean_out, model_uncertainty, model_uncertainty

    def calculate_entropy(self, output):
        # model_uncertainty = entropy of the mean probs over the k-heads
        b, c, d, h, w = output.shape
        output = output.data
        entropy = (-1) * torch.sum(output * torch.log(output + 1e-16), dim=1)
        # add channel axis again
        model_uncertainty = entropy.reshape(b, 1, d, h, w)
        return model_uncertainty

