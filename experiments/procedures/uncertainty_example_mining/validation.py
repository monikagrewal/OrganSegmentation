import logging
import os

import numpy as np
import torch
from scipy import signal
from sklearn import metrics
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from experiments.config import config
from experiments.utils.cache import RuntimeCache
from experiments.utils.metrics import calculate_metrics
from experiments.utils.postprocessing import postprocess_segmentation
from experiments.utils.utilities import log_iteration_metrics
from experiments.utils.visualize import visualize_uncertainty_validation


def inference(dataset, model, criterion, cache, visualize=True, return_raw=False):
    metrics = np.zeros((4, len(config.CLASSES)))
    min_depth = 2 ** config.MODEL_PARAMS["depth"]
    model.eval()

    losses = []
    for nbatches, (image, label) in enumerate(dataset):
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        image_loss = 0
        with torch.no_grad():
            image = torch.tensor(image).to(config.DEVICE)
            label = torch.tensor(label).to(config.DEVICE)
            nslices = image.shape[2]

            output = torch.zeros(
                config.BATCHSIZE, len(config.CLASSES), *image.shape[2:]
            )
            data_uncertainty = torch.zeros(config.BATCHSIZE, 1, *image.shape[2:])
            model_uncertainty = torch.zeros(config.BATCHSIZE, 1, *image.shape[2:])
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
                mini_label = label[:, indices, :, :]
                (
                    mini_output,
                    mini_data_uncertainty,
                    mini_model_uncertainty,
                ) = model.inference(mini_image, return_raw=return_raw)
                loss = criterion((mini_output, mini_data_uncertainty), mini_label)
                image_loss += loss.item()

                if config.SLICE_WEIGHTING:
                    actual_slices = mini_image.shape[2]
                    weights = signal.gaussian(actual_slices, std=actual_slices / 6)
                    weights = torch.tensor(weights, dtype=torch.float32)

                    output[:, :, indices, :, :] += mini_output.to(
                        device="cpu", dtype=torch.float32
                    ) * weights.view(1, 1, actual_slices, 1, 1)

                    data_uncertainty[:, :, indices, :, :] += mini_data_uncertainty.to(
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

                    data_uncertainty[:, :, indices, :, :] += mini_data_uncertainty.to(
                        device="cpu", dtype=torch.float32
                    )

                    model_uncertainty[:, :, indices, :, :] += mini_model_uncertainty.to(
                        device="cpu", dtype=torch.float32
                    )

                    slice_overlaps[0, 0, indices, 0, 0] += 1

            output = output / slice_overlaps
            output = torch.argmax(output, dim=1).view(*image.shape)
            data_uncertainty = data_uncertainty / slice_overlaps
            model_uncertainty = model_uncertainty / slice_overlaps

        image = image.data.cpu().numpy()
        output = output.data.cpu().numpy()
        data_uncertainty = data_uncertainty.data.cpu().numpy()
        model_uncertainty = model_uncertainty.data.cpu().numpy()
        label = label.view(*image.shape).data.cpu().numpy()

        # Postprocessing
        if config.POSTPROCESSING:
            multiple_organ_indici = [
                idx
                for idx, class_name in enumerate(config.CLASSES)
                if class_name == "hip"
            ]
            output = postprocess_segmentation(
                output[0, 0],  # remove batch and color channel dims
                n_classes=len(config.CLASSES),
                multiple_organ_indici=multiple_organ_indici,
                bg_idx=0,
            )
            # return batch & color channel dims
            output = np.expand_dims(np.expand_dims(output, 0), 0)

        im_metrics = calculate_metrics(label, output, class_names=config.CLASSES)
        metrics = metrics + im_metrics
        losses.append(image_loss)

        # probably visualize
        if visualize:
            visualize_uncertainty_validation(
                image,
                (output, data_uncertainty, model_uncertainty),
                label,
                cache.out_dir_val,
                class_names=config.CLASSES,
                base_name=f"out_{nbatches}",
            )

    metrics /= nbatches + 1
    return metrics, losses


def validate(
    dataset: Dataset,
    model: nn.Module,
    criterion: nn.Module,
    cache: RuntimeCache,
    writer: SummaryWriter,
    visualize: bool = True,
):

    metrics, losses = inference(
        dataset, model, criterion, cache, visualize=visualize, return_raw=False
    )

    # Logging
    accuracy, recall, precision, dice, haussdorf_distance, surface_distance = metrics
    log_iteration_metrics(metrics, steps=cache.epoch, writer=writer, data="validation")
    logging.debug(
        f"Proper evaluation results:\n"
        f"accuracy = {accuracy}\nrecall = {recall}\n"
        f"precision = {precision}\ndice = {dice}\n"
    )
    for class_no, classname in enumerate(config.CLASSES):
        cache.last_epoch_results.update(
            {
                f"recall_{classname}": recall[class_no],
                f"precision_{classname}": precision[class_no],
                f"dice_{classname}": dice[class_no],
            }
        )

    mean_dice = np.mean(dice[1:])
    cache.last_epoch_results.update({"mean_dice": mean_dice})

    # Store model if best in validation
    if mean_dice >= cache.best_mean_dice:
        logging.info(f"Epoch: {cache.epoch}: Best Dice: {mean_dice}")
        cache.best_epoch = cache.epoch
        cache.best_mean_dice = mean_dice
        cache.epochs_no_improvement = 0

        if config.SAVE_MODEL == "best":
            weights = {
                "model": model.state_dict(),
                "epoch": cache.epoch,
                "mean_dice": mean_dice,
            }
            torch.save(weights, os.path.join(cache.out_dir_weights, "best_model.pth"))
    else:
        cache.epochs_no_improvement += 1

    # Store model at end of epoch to get final model (also on failure)
    if config.SAVE_MODEL == "final":
        weights = {
            "model": model.state_dict(),
            "epoch": cache.epoch,
            "mean_dice": mean_dice,
        }
        torch.save(weights, os.path.join(cache.out_dir_weights, "final_model.pth"))

    cache.last_epoch_results.update({"best_epoch": cache.best_epoch})
    cache.all_epoch_results.append(cache.last_epoch_results)
    return cache, losses
