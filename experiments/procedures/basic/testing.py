import logging
import os

import numpy as np
import torch
from experiments.utils.cache import RuntimeCache
from experiments.utils.metrics import calculate_metrics
from experiments.utils.postprocessing import postprocess_segmentation
from experiments.utils.utilities import log_iteration_metrics
from experiments.utils.visualize import visualize_output
from scipy import signal
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def test(model: nn.Module, dataloader: DataLoader, config):

    metrics = np.zeros((4, len(config.CLASSES)))
    min_depth = 2**model.depth
    model.eval()

    for nbatches, (image, label) in enumerate(dataloader):
        label = label.view(*image.shape).data.cpu().numpy()
        with torch.no_grad():

            nslices = image.shape[2]
            image = image.to(config.DEVICE)

            output = torch.zeros(
                config.BATCHSIZE, len(config.CLASSES), *image.shape[2:]
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
                outputs = model.inference(mini_image)
                if isinstance(outputs, tuple):
                    mini_output = outputs[0]
                else:
                    mini_output = outputs
                if config.SLICE_WEIGHTING:
                    actual_slices = mini_image.shape[2]
                    weights = signal.gaussian(actual_slices, std=actual_slices / 6)
                    weights = torch.tensor(weights, dtype=torch.float32)

                    output[:, :, indices, :, :] += mini_output.to(
                        device="cpu", dtype=torch.float32
                    ) * weights.view(1, 1, actual_slices, 1, 1)
                    slice_overlaps[0, 0, indices, 0, 0] += weights
                else:
                    output[:, :, indices, :, :] += mini_output.to(
                        device="cpu", dtype=torch.float32
                    )
                    slice_overlaps[0, 0, indices, 0, 0] += 1

            output = output / slice_overlaps
            output = torch.argmax(output, dim=1).view(*image.shape)

        image_cpu = image.data.cpu().numpy()
        output_cpu = output.data.cpu().numpy()
        del image, outputs, mini_image, mini_output
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
            # return batch & color channel dims
            output_cpu = np.expand_dims(np.expand_dims(output_cpu, 0), 0)

        im_metrics = calculate_metrics(label, output_cpu, class_names=config.CLASSES)
        metrics = metrics + im_metrics

        visualize_output(
            image_cpu,
            output_cpu,
            label,
            config.out_dir_val,
            class_names=config.CLASSES,
            base_name=f"out_{nbatches}",
        )

    metrics /= nbatches + 1

    accuracy, recall, precision, dice = metrics
    logging.info(
        f"Proper evaluation results:\n"
        f"accuracy = {accuracy}\n"
        f"recall = {recall}\n"
        f"precision = {precision}\n"
        f"dice = {dice}\n"
    )
