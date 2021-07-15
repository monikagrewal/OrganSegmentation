from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


def get_criterion() -> nn.Module:
    criterion: nn.Module
    if config.LOSS_FUNCTION == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    elif config.LOSS_FUNCTION == "weighted_cross_entropy":
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(config.CLASS_WEIGHTS, device=config.DEVICE)
        )

    elif config.LOSS_FUNCTION == "focal_loss":
        if config.ALPHA is not None:
            alpha = torch.tensor(config.ALPHA, device=config.DEVICE)
        criterion = FocalLoss(gamma=config.GAMMA, alpha=alpha)

    elif config.LOSS_FUNCTION == "soft_dice":
        criterion = SoftDiceLoss(drop_background=False)

    elif config.LOSS_FUNCTION == "weighted_soft_dice":
        criterion = SoftDiceLoss(
            weight=torch.tensor(config.CLASS_WEIGHTS, device=config.DEVICE)
        )

    else:
        raise ValueError(f"unknown loss function: {config.LOSS_FUNCTION}")

    return criterion


def convert_idx_to_onehot(label, num_classes):
    label_onehot = torch.eye(num_classes).to(label.device)[label.view(-1)]
    return label_onehot


class SoftDiceLoss(nn.Module):
    """
    Inputs:
    Probs = probability outputs after softmax or sigmoid,
        with channels along last dimension
    targets = one-hot encoded target vectors (num_examples, num_classes)
    """

    def __init__(
        self, weight: Optional[torch.Tensor] = None, drop_background: bool = True
    ) -> None:
        super(SoftDiceLoss, self).__init__()
        self.weight = weight
        self.drop_background = drop_background

    def forward(self, input, target):
        smooth = 1e-6
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
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
