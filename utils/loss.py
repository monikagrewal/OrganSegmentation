import logging
from copy import copy, deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class PartialAnnotationImputeLoss(nn.Module):
    """
    Weight a criterion based on the uncertainty map

    Args:
        criterion (nn.Module): Criterion to use for the loss
    """

    def __init__(self, **kwargs):
        super(PartialAnnotationImputeLoss, self).__init__()
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
            mask: non_ambiguity_mask (0 = annotation absent, 1 = annotation present)

        Returns:
            torch.Tensor: Weighted loss
        """
        single_output, uncertainty, prediction = inputs
        assert prediction.shape==target.shape
        mask = mask.reshape(*target.shape)
        uncertainty = uncertainty.reshape(*target.shape)

        # impute target with pseudo label where annotation is not present i.e.,
        # non_ambiguity_mask = 0
        target[mask==0] = prediction[mask==0]
        seg_loss = self.criterion(single_output, target)

        uncertainty[mask==1] = 0  #uncertainty = 0 where annotation is present
        uncertainty_weight = torch.exp(-1 * uncertainty)
        loss = torch.mean(seg_loss * uncertainty_weight)
        return loss