import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_idx_to_onehot(label, num_classes):
	label_onehot = torch.eye(num_classes).to(label.device)[label.view(-1)]
	return label_onehot


class SoftDiceLoss(nn.Module):
	"""
	Inputs:
	Probs = probability outputs after softmax or sigmoid, with channels along last dimension
	targets = one-hot encoded target vectors (num_examples, num_classes)
	"""
	def __init__(self, weight=None):
		super(SoftDiceLoss, self).__init__()
		self.weight = weight

	def forward(self, probs, targets, weight=None):
		smooth = 1e-6

		if self.weight is not None:
			weight = self.weight.view(1, -1)
			intersection = torch.dot(probs.view(-1), (targets * weight).view(-1))
		else:
			intersection = torch.dot(probs.view(-1), targets.view(-1))
		
		union = torch.sum(probs) + torch.sum(targets)

		loss = 1 - (2. * intersection + smooth) / (union + smooth)
		return loss


class MaskedCrossEntropyLoss():
	"""
	Naive implementation to handle missing annotation
	"""
	def __init__(self):
		super(MaskedCrossEntropyLoss, self).__init__()

	def forward(self, output, label, classes):   
		mask = torch.ones_like(label)
		if len(torch.unique(label)) != len(classes):
			mask = (label!=0).float()

		loss = F.cross_entropy(output, label, reduction="none")
		loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-6)
		return loss


class WeightedCrossEntropyLoss():
	"""
	Weights cross-entropy based on inverse class frequency
	currently for 2 classes only
	"""
	def __init__(self):
		super(WeightedCrossEntropyLoss, self).__init__()

	def forward(self, output, label):
		device = output.device
		Ntotal = label.view(-1).size()[0]
		weight = torch.tensor([(label==1).sum() + 1, (label==0).sum()], device=device, dtype=torch.float) / Ntotal   

		loss = F.cross_entropy(output, label, weight=weight)
		return loss


class FocalLoss(nn.Module):
	def __init__(self, alpha=10, gamma=4.5):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, input, target):
		if input.dim()>2:
			input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1, 1)

		pt = F.softmax(input, dim=1)
		pt = pt.gather(1, target)
		pt = pt.view(-1)
		pt = self.alpha*torch.exp(-self.gamma*pt)
		logpt = F.log_softmax(input, dim=1)
		logpt = logpt.gather(1, target)
		logpt = logpt.view(-1)

		loss = -1 * pt * logpt
		loss = loss.sum() / (torch.le(pt, 0.5).sum() + 1e2)
		return loss

