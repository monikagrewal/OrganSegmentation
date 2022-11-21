import logging
from copy import copy, deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.config import config


def convert_idx_to_onehot(label, num_classes):
    label_onehot = torch.eye(num_classes).to(label.device)[label.view(-1)]
    return label_onehot


def convert_seg_output_to_2d(input: torch.Tensor) -> torch.Tensor:
    assert input.dim() > 2
    input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
    input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
    input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
    return input


class UncertaintyLoss(nn.Module):
    """
    Learn data uncertainty map along with segmentation loss

    Args:
        criterion (nn.Module): Criterion to use for the loss
    """

    def __init__(self, seg_loss: str = "cross_entropy", reduction="mean", **kwargs):
        super(UncertaintyLoss, self).__init__()
        self.seg_loss_name = seg_loss
        self.reduction = reduction
        if seg_loss == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(reduction='none', **kwargs)
        elif seg_loss == "soft_dice":
            self.criterion = SoftDiceLoss(**kwargs)
        else:
            raise NotImplementedError(f"loss function: {seg_loss} not implemented yet.")

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor, torch.Tensor): probability outputs without softmax or sigmoid
            &
            data uncertainty map,
            target (torch.Tensor): target tensor with each class coded with an integer

        Returns:
            torch.Tensor: Weighted loss
        """
        seg_output, log_variance = inputs

        if self.seg_loss_name=="cross_entropy":
            seg_loss = self.criterion(seg_output, target) #N x d x h x w
            log_variance = torch.clamp(log_variance, -3, 20)
            log_variance = log_variance[:, 0, :, :, :]
            loss = torch.exp((-1) * log_variance) * seg_loss + log_variance
        elif self.seg_loss_name=="soft_dice":
            weighted_seg_output = torch.exp((-1) * log_variance) * seg_output
            loss = self.criterion(weighted_seg_output, target)

        if self.reduction != "none":
            loss = loss.mean()
        return loss


class UncertaintyWeightedLoss(UncertaintyLoss):
    """
    Weight a criterion based on the uncertainty map

    Args:
        criterion (nn.Module): Criterion to use for the loss
    """

    def __init__(self, seg_loss: str = "cross_entropy", reduction="mean", **kwargs):
        super(UncertaintyWeightedLoss, self).__init__(seg_loss, \
                            reduction=reduction, **kwargs)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor
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
        seg_output, model_variance = inputs
        seg_loss = self.criterion(seg_output, target) #N x d x h x w
        model_variance = model_variance.reshape(*seg_loss.shape)
        loss = model_variance * seg_loss

        if self.reduction != "none":
            loss = loss.mean()
        return loss


class UncertaintyWeightedPerClassLoss(UncertaintyLoss):
    """
    Weight a criterion based on the uncertainty map

    Args:
        criterion (nn.Module): Criterion to use for the loss
    """

    def __init__(self, seg_loss: str = "cross_entropy", reduction="mean", **kwargs):
        super(UncertaintyWeightedPerClassLoss, self).__init__(seg_loss, \
                            reduction=reduction, **kwargs)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor
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
        seg_output, model_variance = inputs
        seg_loss = self.criterion(seg_output, target) #N x d x h x w
        model_variance = model_variance.reshape(*seg_loss.shape)

        # calculate class weights
        nclasses = len(config.CLASSES)
        class_weights = []
        for i in range(nclasses):
            weight = model_variance[target==i].mean()
            class_weights.append(weight.item())
        logging.info(f"class weights: {class_weights}")
        # assign class weights
        for i in range(nclasses):
            model_variance[target==i] = class_weights[i]

        loss = model_variance * seg_loss
        if self.reduction != "none":
            loss = loss.mean()
        return loss


class UncertaintyWeightedDoubleLoss(UncertaintyLoss):
    """
    Weight a criterion based on the uncertainty map

    Args:
        criterion (nn.Module): Criterion to use for the loss
    """

    def __init__(self, seg_loss: str = "cross_entropy", reduction="mean", **kwargs):
        super(UncertaintyWeightedDoubleLoss, self).__init__(seg_loss, \
                            reduction=reduction, **kwargs)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor
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
        seg_output, model_variance = inputs
        seg_loss = self.criterion(seg_output, target) #N x d x h x w
        model_variance = model_variance.reshape(*seg_loss.shape)
        model_variance_per_calss = deepcopy(model_variance)

        # calculate class weights
        nclasses = len(config.CLASSES)
        class_weights = []
        for i in range(nclasses):
            weight = model_variance[target==i].mean()
            class_weights.append(weight.item())
        logging.info(f"class weights: {class_weights}")
        # assign class weights
        for i in range(nclasses):
            model_variance_per_calss[target==i] = class_weights[i]

        loss = model_variance * model_variance_per_calss * seg_loss
        if self.reduction != "none":
            loss = loss.mean()
        return loss


class PartialAnnotationLoss(nn.Module):
    """
    Weight a criterion based on the uncertainty map

    Args:
        criterion (nn.Module): Criterion to use for the loss
    """

    def __init__(self, base_loss: str = "UncertaintyWeightedDoubleLoss", \
                    seg_loss: str = "cross_entropy", **kwargs):
        super(PartialAnnotationLoss, self).__init__()

        if base_loss == "uncertainty":
            self.criterion = UncertaintyLoss(seg_loss, reduction="none", **kwargs)

        if base_loss == "uncertainty_weighted":
            self.criterion = UncertaintyWeightedLoss(seg_loss, reduction="none", **kwargs)

        elif base_loss == "uncertainty_weighted_class":
            self.criterion = UncertaintyWeightedPerClassLoss(seg_loss, reduction="none", **kwargs)

        elif base_loss == "uncertainty_weighted_double":
            self.criterion = UncertaintyWeightedDoubleLoss(seg_loss, reduction="none", **kwargs)
        else:
            raise NotImplementedError(f"base_loss = {base_loss} not implemented.")


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
        loss = self.criterion(inputs, target)
        mask = mask.reshape(*loss.shape)
        loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-6)
        return loss


class PartialAnnotationImputeLoss(nn.Module):
    """
    Weight a criterion based on the uncertainty map

    Args:
        criterion (nn.Module): Criterion to use for the loss
    """

    def __init__(self, soft_label=False, **kwargs):
        super(PartialAnnotationImputeLoss, self).__init__()
        print("soft_label: ", soft_label)
        self.soft_label = soft_label
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
        if self.soft_label:
            uncertainty_weight = uncertainty
        else:
            uncertainty_weight = torch.exp(-1 * uncertainty)
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


class MaskedCrossEntropyLoss:
    """
    Naive implementation to handle missing annotation
    """

    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, output, label, classes):
        mask = torch.ones_like(label)
        if len(torch.unique(label)) != len(classes):
            mask = (label != 0).float()

        loss = F.cross_entropy(output, label, reduction="none")
        loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-6)
        return loss


class WeightedCrossEntropyLoss:
    """
    Weights cross-entropy based on inverse class frequency
    currently for 2 classes only
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, output, label):
        device = output.device
        Ntotal = label.view(-1).size()[0]
        weight = (
            torch.tensor(
                [(label == 1).sum() + 1, (label == 0).sum()],
                device=device,
                dtype=torch.float,
            )
            / Ntotal
        )

        loss = F.cross_entropy(output, label, weight=weight)
        return loss


class FocalLossOld(nn.Module):
    def __init__(self, alpha=10, gamma=4.5):
        super(FocalLossOld, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        pt = F.softmax(input, dim=1)
        pt = pt.gather(1, target)
        pt = pt.view(-1)
        pt = self.alpha * torch.exp(-self.gamma * pt)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * pt * logpt
        loss = loss.sum() / (torch.le(pt, 0.5).sum() + 1e2)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=4.5, epsilon=1e-6, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
            # input = input.permute(0,2,3, 1).contiguous().view(-1, 3)
        target = target.view(-1, 1)

        pt = F.softmax(input, dim=1)
        pt = pt.gather(1, target)
        pt = pt.view(-1)

        fl = -(1 - pt).pow(self.gamma) * (pt + self.epsilon).log()
        if self.alpha is not None:
            alpha_t = self.alpha[target.view(-1)]
            fl = fl * alpha_t
        loss = fl.mean()
        return loss
