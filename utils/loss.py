import logging
from copy import copy, deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


def convert_idx_to_onehot(label, num_classes):
    label_onehot = torch.eye(num_classes).to(label.device)[label.view(-1)]
    return label_onehot


def convert_seg_output_to_2d(input: torch.Tensor) -> torch.Tensor:
    assert input.dim() > 2
    input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
    input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
    input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
    return input


class PartialAnnotationImputeLoss(nn.Module):
    """
    Weight a criterion based on the uncertainty map

    Args:
        criterion (nn.Module): Criterion to use for the loss
    """

    def __init__(self, weighting_method="uncertainty", **kwargs):
        super(PartialAnnotationImputeLoss, self).__init__()
        self.weighting_method = weighting_method
        self.criterion = nn.CrossEntropyLoss(reduction='none', **kwargs)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor, torch.Tensor): probability outputs without softmax or sigmoid
            &
            model uncertainty map or model variance map,
            target (torch.Tensor): target tensor with each class coded with an integer

        Returns:
            torch.Tensor: Weighted loss
        """
        single_output, uncertainty, prediction = inputs
        assert prediction.shape==target.shape
        mask = mask.reshape(*target.shape)
        uncertainty = uncertainty.reshape(*target.shape)

        target[mask==0] = prediction[mask==0]  #fill with pseudo label
        seg_loss = self.criterion(single_output, target)

        uncertainty[mask==1] = 0  #uncertainty = 0 where annotation is present
        if self.weighting_method=="confidence":
            uncertainty_weight = uncertainty
        elif self.weighting_method=="uncertainty":
            uncertainty_weight = torch.exp(-1 * uncertainty)
        else:
            raise ValueError(f"Unknown weighting method {self.weighting_method}")
        loss = torch.mean(seg_loss * uncertainty_weight)
        return loss


class SoftDiceLoss(nn.Module):
    """
    Inputs:
    Probs = network outputs without softmax or sigmoid,
    targets = not one-hot encoded
    """

    def __init__(
        self, weight: Optional[torch.Tensor] = None, drop_background: bool = False
    ) -> None:
        super(SoftDiceLoss, self).__init__()
        self.weight = weight
        self.drop_background = drop_background
        logging.debug(f"DROP background: {self.drop_background}")

    def forward(self, input, target):
        smooth = 1e-6
        input = convert_seg_output_to_2d(input)
        target = target.view(-1)

        probs = F.softmax(input, dim=1)
        nclasses = probs.shape[1]
        target_one_hot = torch.eye(nclasses, device=probs.device)[target]
        if self.drop_background:
            # drop background
            probs = probs[:, 1:].view(-1, nclasses - 1)
            target_one_hot = target_one_hot[:, 1:].view(-1, nclasses - 1)
            if self.weight is not None:
                self.weight = self.weight[1:].view(1, -1)

        intersection = torch.sum(probs * target_one_hot, dim=0)
        union = torch.sum(probs, dim=0) + torch.sum(target_one_hot, dim=0)
        dice = (2.0 * intersection + smooth) / (union + smooth)

        if self.weight is not None:
            loss = (self.weight * (1 - dice)).mean()
        else:
            loss = (1 - dice).mean()

        return loss
