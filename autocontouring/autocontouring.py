import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import cv2
from pydantic import BaseSettings, validator
from scipy import signal
from torch import nn
from skimage.measure import find_contours

sys.path[0] = str(Path(sys.path[0]).parent)
from experiments.models.unet_khead import KHeadUNet
from experiments.utils.postprocessing_testing import \
	postprocess_segmentation


class Config(BaseSettings):
	# General
	MODE: str = "test"
	DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
	CLASSES: List[str] = ["background", "bowel_bag", "bladder", "hip", "rectum"]

	RANDOM_SEED: int = 0
	IMAGE_DEPTH: int = 32
	WEIGHTS_PATH: str = "/export/scratch2/data/grewal/OAR_segmentation/runs/" +\
		"final_experiments/robust-teacher-folds_05122022_161207/"+\
		"fold0/run0/weights/best_model.pth"
	# for sliding window validation, overlapping slice windows passed to the model.
	# If true, apply gaussian weighting so that the predictions in center of the window
	# have more weight on the final prediction for a voxel
	SLICE_WEIGHTING: bool = True
	POSTPROCESSING: bool = True

	class Config:
		env_file = ".env"
		env_file_encoding = "utf-8"


def main(im_arr: np.array) -> np.array:
	print("generating auto contours ...")
	image = np.expand_dims(np.expand_dims(im_arr, 0), 0)
	image = torch.from_numpy(image)
	min_depth = 2 ** model.depth
	with torch.no_grad():
		nslices = image.shape[2]
		image = image.to(config.DEVICE)

		output = torch.zeros(
			1, len(config.CLASSES), *image.shape[2:]
		)
		model_uncertainty = torch.zeros(
			1, 1, *image.shape[2:]
		)
		slice_overlaps = torch.zeros(1, 1, nslices, 1, 1)
		start = 0
		while start + min_depth <= nslices:
			if start + config.IMAGE_DEPTH >= nslices:
				indices = slice(nslices - config.IMAGE_DEPTH, nslices)
				start = nslices
			else:
				indices = slice(start, start + config.IMAGE_DEPTH)
				start += config.IMAGE_DEPTH // 3

			mini_image = image[:, :, indices, :, :]
			mini_output, _, mini_model_uncertainty = \
									model.inference(mini_image)

			if config.SLICE_WEIGHTING:
				actual_slices = mini_image.shape[2]
				weights = signal.gaussian(actual_slices, std=actual_slices / 6)
				weights = torch.tensor(weights, dtype=torch.float32)

				output[:, :, indices, :, :] += mini_output.to(
					device="cpu", dtype=torch.float32
				) * weights.view(1, 1, actual_slices, 1, 1)

				model_uncertainty[:, :, indices, :, :] += mini_model_uncertainty.to(
					device="cpu", dtype=torch.float32
				) * weights.view(1, 1, actual_slices, 1, 1)

				slice_overlaps[0, 0, indices, 0, 0] += 1
			else:
				output[:, :, indices, :, :] += mini_output.to(
					device="cpu", dtype=torch.float32
				)

				model_uncertainty[:, :, indices, :, :] += mini_model_uncertainty.to(
					device="cpu", dtype=torch.float32
				)

				slice_overlaps[0, 0, indices, 0, 0] += 1

		output = output / slice_overlaps
		output = torch.argmax(output, dim=1).view(*image.shape)
		model_uncertainty = model_uncertainty / slice_overlaps

	output_cpu = output.data.cpu().numpy()
	model_uncertainty = model_uncertainty.data.cpu().numpy()[0, 0]
		
	del image, mini_image, mini_output
	torch.cuda.empty_cache()

	# Postprocessing
	if config.POSTPROCESSING:
		multiple_organ_indici = [
			idx
			for idx, class_name in enumerate(config.CLASSES)
			if class_name == "hip"
		]
		output_cpu = postprocess_segmentation(
			output_cpu[0, 0],  # remove batch and color channel dims
			n_classes=len(config.CLASSES),
			multiple_organ_indici=multiple_organ_indici,
			bg_idx=0,
		)

	return output_cpu, model_uncertainty


def crop_center(im:np.array, crop_width: int=50, crop_height: int=120) \
	-> np.array:
	"""
	im: 2D np array, single slice, 1 channels
	"""
	h, w = im.shape
	im = im[crop_height : h - crop_height, crop_width : w - crop_width]
	return im

parser = argparse.ArgumentParser(description="Organ at Risk model Inference")
parser.add_argument(
	"--env-file",
	dest="env_file",
	type=str,
	default=None,
	help="Set the location of the environment file.",
)

cli_args = parser.parse_args()
config = Config(_env_file=cli_args.env_file) if cli_args.env_file else Config()

model = KHeadUNet(out_channels=len(config.CLASSES)).to(config.DEVICE)
weights = torch.load(config.WEIGHTS_PATH,
	map_location=config.DEVICE)["model"]
model.load_state_dict(weights)
model.eval()

if __name__=="__main__":
	root_path = "../outputs/autocontours_review_processed"
	output_path = "../outputs/schematic"
	os.makedirs(output_path, exist_ok=True)

	patient_id = 3
	patient_path = os.path.join(root_path, "{0:03d}_CT_EBRT".format(patient_id))
	image = np.load(os.path.join(patient_path, "image.npz"))["volume"]
	mask, uncertainty = main(image)

	# # vizualization
	# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
	# for i in range(im_arr.shape[0]):
	# 	im = im_arr[i, :, :]
	# 	im_color = (cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) * 255).astype(np.uint8)
	# 	mask = output_cpu[i, :, :]
	# 	for class_no, _ in enumerate(["bowel_bag", "bladder", "hip", "rectum"]):
	# 		# contours = find_contours((mask==class_no+1).astype(np.float32), fully_connected='low', level=0.99)
	# 		# for contour in contours:
	# 		# 	rr, cc = contour.astype(np.int32)[:, 0], contour.astype(np.int32)[:, 1]
	# 		# 	im_color[rr, cc, :] = colors[class_no]
	# 		rr, cc = np.where(mask==class_no+1)
	# 		im_color[rr, cc, :] = colors[class_no]

	# 	cv2.imwrite(f"./outputs/seg_{i}.png", im_color)

	# vizualization
	print(uncertainty.shape, uncertainty.max(), uncertainty.min())
	slice_idx = 63
	im = uncertainty[slice_idx, :, :]
	im = crop_center(im)
	im = cv2.resize(im, (272, 272))
	print(im.shape)

	filepath = os.path.join(output_path, f"{patient_id}_{slice_idx}_uncertainty.jpg")
	cv2.imwrite(filepath, (im*255).astype(np.uint8))
	

	# u = 0 at hips and rectum because for image generation, 
	# it is assumed that annotations are present for hip and rectum
	uncertainty[mask==3] = 0
	uncertainty[mask==4] = 0
	im = uncertainty[slice_idx, :, :]
	uncertainty_weight = np.exp(-1*im)
	print(f"uncertainty weight max: {uncertainty_weight.max()}, min: {uncertainty_weight.min()}")
	uncertainty_weight = crop_center(uncertainty_weight)
	uncertainty_weight = cv2.resize(uncertainty_weight, (272, 272))
	filepath = os.path.join(output_path, f"{patient_id}_{slice_idx}_uncertainty_weight.jpg")
	cv2.imwrite(filepath, (uncertainty_weight*255).astype(np.uint8))