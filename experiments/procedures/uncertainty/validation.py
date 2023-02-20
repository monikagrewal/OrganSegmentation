import logging
import os

import numpy as np
import torch
from scipy import signal
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from experiments.config import config
from experiments.utils.cache import RuntimeCache
from experiments.utils.metrics import calculate_metrics
from experiments.utils.postprocessing import postprocess_segmentation
from experiments.utils.utilities import log_iteration_metrics
from experiments.utils.visualize import visualize_uncertainty_validation


def validate(
    proper_val_dataloader: DataLoader,
    model: nn.Module,
    cache: RuntimeCache,
    writer: SummaryWriter,
):
    metrics = np.zeros((4, len(config.CLASSES)))
    min_depth = 2 ** config.MODEL_PARAMS["depth"]
    model.eval()

    for nbatches, (image, label) in enumerate(proper_val_dataloader):
        with torch.no_grad():
            nslices = image.shape[2]
            image = image.to(config.DEVICE)

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
                (
                    mini_output,
                    mini_data_uncertainty,
                    mini_model_uncertainty,
                ) = model.inference(mini_image)

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

        # probably visualize
        if config.VISUALIZE_OUTPUT in ["val", "all"]:
            visualize_uncertainty_validation(
                image,
                (output, data_uncertainty, model_uncertainty),
                label,
                cache.out_dir_val,
                class_names=config.CLASSES,
                base_name=f"out_{nbatches}",
            )

    metrics /= nbatches + 1

    # Logging
    accuracy, recall, precision, dice, haussdorf_distance, surface_dice = metrics
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
    return cache
