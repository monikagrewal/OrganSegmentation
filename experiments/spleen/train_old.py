import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from tensorboard_logger import configure, log_value

import os, argparse
import numpy as np
import json
from skimage.io import imread, imsave
from model_unet import UNet
from etl import *

"""
command line to run the code:
python3 train.py -o "./runs/run1" -d 4 -w 16 -i 128 -n 80 -lr 0.01 -b 1

CUDA_VISIBLE_DEVICES=0, GPU=3
CUDA_VISIBLE_DEVICES=1, GPU=0

"""



parser = argparse.ArgumentParser(description='Train UNet')
parser.add_argument("-out_dir", help="output directory", default="./runs/run_3d_no_balance")
parser.add_argument("-depth", help="network depth", type=int, default=4)
parser.add_argument("-width", help="network width", type=int, default=16)
parser.add_argument("-image_size", help="image size", type=int, default=256)
parser.add_argument("-nepochs", help="number of epochs", type=int, default=100)
parser.add_argument("-lr", help="learning rate", type=float, default=0.01)
parser.add_argument("-batchsize", help="batchsize", type=int, default=2)

def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def parse_input_arguments():
	run_params = parser.parse_args()
	run_params = vars(run_params)
	out_dir = run_params["out_dir"]
	makedir(out_dir)

	json.dump(run_params, open(os.path.join(out_dir, "run_parameters.json"), "w"))
	return run_params


def main():
	run_params = parse_input_arguments()

	out_dir, nepochs, lr, batchsize = run_params["out_dir"], run_params["nepochs"], run_params["lr"], run_params["batchsize"]
	depth, width, image_size = run_params["depth"], run_params["width"], run_params["image_size"]
	
	out_dir_train = os.path.join(out_dir, "train")
	out_dir_val = os.path.join(out_dir, "val")
	out_dir_wts = os.path.join(out_dir, "weights")
	makedir(out_dir_train)
	makedir(out_dir_val)
	makedir(out_dir_wts)
	configure(out_dir, flush_secs=5)

	root_dir = "../Data/Task09_Spleen"
	jsonname = "dataset.json"
	use_cuda = True
	best_dice = 0.65

	train_dataset = SpleenDataset(root_dir, jsonname, image_size=image_size, slice_thickness=3, image_depth=32, is_training=True, augment=True)
	val_dataset = SpleenDataset(root_dir, jsonname, image_size=image_size, slice_thickness=3, image_depth=32, is_training=False, augment=False)
	
	weight = np.array([0.0045128, 0.9954872])# pre-calculated weights
	print("Class Weights: {}".format(weight))
	weight = torch.from_numpy(weight).float()

	model = UNet(depth=depth, width=width, in_channels=1, out_channels=2)
	if use_cuda:
		model.cuda()
		weight = weight.cuda()
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
	criterion = nn.CrossEntropyLoss()

	weights = torch.load(os.path.join(out_dir_wts, "best_model.pth"))["model"]
	model.load_state_dict(weights)

	
	train_steps = 0
	val_steps = 0
	for epoch in range(0, nepochs):
		# training
		model.train()
		train_loss = 0.
		train_acc = 0.
		nbatches = 0
		for nbatches, (image, label) in enumerate(train_dataset.BatchLoader(batchsize=batchsize, use_cuda=use_cuda)):
			optimizer.zero_grad()
			output = model(image)
			output = output.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2)
			loss = criterion(output, label.view(-1))
			loss.backward()
			optimizer.step()

			train_loss += loss.item()

			_, pred = torch.max(output, 1)
			pred = pred.data.cpu().numpy()
			label = label.data.cpu().numpy()
			acc = np.sum(pred == label.flatten()) / len(pred)
			train_acc += acc
			print("Iteration {}: Train Loss: {}, Train Accuracy: {}".format(nbatches, loss.item(), acc))
			log_value("Loss/train_loss", loss.item(), train_steps)
			train_steps += 1


			# if nbatches%100 == 0:
			# 	pred = pred.reshape(label.shape)
			# 	alpha = 0.6
			# 	color = [1, 0, 0]
			# 	for i in range(image.shape[2]):
			# 		im = image.data.cpu().numpy()[0, 0, i, :, :]
			# 		lbl = label[0, 0, i, :, :]
			# 		prob = pred[0, 0, i, :, :]

			# 		im = np.repeat(im.flatten(), 3).reshape(im.shape[0], im.shape[1], 3)
			# 		lbl = np.repeat(lbl.flatten(), 3).reshape(lbl.shape[0], lbl.shape[1], 3)
			# 		prob = np.repeat(prob.flatten(), 3).reshape(prob.shape[0], prob.shape[1], 3)

			# 		im_plus_label = (1-alpha*lbl)*im + alpha*lbl*color
			# 		im_plus_pred = (1-alpha*prob)*im + alpha*prob*color

			# 		out_im = np.concatenate((im, im_plus_label, im_plus_pred), axis=1)
			# 		imsave(os.path.join(out_dir_train, "iter_{}_{}_{}.jpg".format(epoch, nbatches, i)), (out_im*255).astype(np.uint8))

		train_loss = train_loss / float(nbatches+1)
		train_acc = train_acc / float(nbatches+1)

		# validation
		model.eval()
		val_loss = 0.
		tp, tn, fp, fn = 0, 0, 0, 0
		for nbatches, (image, label) in enumerate(val_dataset.BatchLoader(batchsize=batchsize, use_cuda=use_cuda)):
			with torch.no_grad():
				output = model(image)
				output = output.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2)
				loss = criterion(output, label.view(-1))
				val_loss += loss.item()

			_, pred = torch.max(output, 1)
			pred = pred.data.cpu().numpy()
			label = label.data.cpu().numpy()
			pred = pred.reshape(label.shape)

			tp += np.sum(pred * label)
			tn += np.sum((1- pred) * (1 - label))
			fp += np.sum(pred * (1 - label))
			fn += np.sum((1 - pred) * label)

			print("Iteration {}: Validation Loss: {}".format(nbatches, loss.item()))
			log_value("Loss/validation_loss", loss.item(), val_steps)
			val_steps += 1

			if nbatches%3 == 0:
				alpha = 0.6
				color = [1, 0, 0]
				for i in range(image.shape[2]):
					im = image.data.cpu().numpy()[0, 0, i, :, :]
					lbl = label[0, 0, i, :, :]
					prob = pred[0, 0, i, :, :]

					im = np.repeat(im.flatten(), 3).reshape(im.shape[0], im.shape[1], 3)
					lbl = np.repeat(lbl.flatten(), 3).reshape(lbl.shape[0], lbl.shape[1], 3)
					prob = np.repeat(prob.flatten(), 3).reshape(prob.shape[0], prob.shape[1], 3)

					im_plus_label = (1-alpha*lbl)*im + alpha*lbl*color
					im_plus_pred = (1-alpha*prob)*im + alpha*prob*color

					out_im = np.concatenate((im, im_plus_label, im_plus_pred), axis=1)
					imsave(os.path.join(out_dir_val, "iter_{}_{}_{}.jpg".format(epoch, nbatches, i)), (out_im*255).astype(np.uint8))

		val_loss = val_loss / float(nbatches+1)
		accuracy = (tp + tn) / float(tp + tn + fp + fn)
		precision = tp / float(tp + fp)
		recall = tp / float(tp + fn)
		dice = (2*tp) / float(2*tp + fp + fn)

		print("EPOCH {}".format(epoch))
		print("Training Loss: {}, Training Accuracy: {}".format(train_loss, train_acc))
		print("Validation Loss: {}, Accuracy: {}, Precision: {}, Recall: {}, Dice: {}, Best Dice: {}\n".format(val_loss, accuracy, precision, recall, dice, best_dice))
		log_value("Epochwise_Loss/train_loss", train_loss, epoch)
		log_value("Epochwise_Loss/validation_loss", val_loss, epoch)
		log_value("Metric/train_accuracy", train_acc, epoch)
		log_value("Metric/validation_accuracy", accuracy, epoch)
		log_value("Metric/validation_precision", precision, epoch)
		log_value("Metric/validation_recall", recall, epoch)
		log_value("Metric/validation_dice_score", dice, epoch)

		# # saving model
		# weights = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "train_accuracy": train_acc, "validation_accuracy": accuracy}
		# torch.save(weights, os.path.join(out_dir_wts, "model_{}.pth".format(epoch)))

		if dice >= best_dice:
			best_dice = dice
			weights = {"model": model.state_dict(), "epoch": epoch, "dice": dice}
			torch.save(weights, os.path.join(out_dir_wts, "best_model_no_balance.pth"))


if __name__ == '__main__':
	main()


