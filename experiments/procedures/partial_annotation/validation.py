import logging
import os
import numpy as np
from sklearn import metrics
import torch
from scipy import signal
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from utils.cache import RuntimeCache
from utils.metrics import calculate_metrics
from utils.postprocessing import postprocess_segmentation
from utils.visualize import visualize_uncertainty_validation
from utils.utilities import log_iteration_metrics


def inference(val_dataloader, model, criterion, cache, visualize=True, return_raw=False):
    metrics = np.zeros((4, len(config.CLASSES)))
    min_depth = 2 ** config.MODEL_PARAMS["depth"]
    model.eval()

    uncertainties = []
    for nbatches, items in enumerate(val_dataloader):
        image = items[0]  #len(items can be 2 or 3 depending on whether or not mask is returned)
        label = items[1]

        image_uncertainty = 0
        with torch.no_grad():
            image = image.to(config.DEVICE)
            label = label.to(config.DEVICE)
            nslices = image.shape[2]

            output = torch.zeros(
                config.BATCHSIZE, len(config.CLASSES), *image.shape[2:],
                device="cpu"
            )
            data_uncertainty = torch.zeros(
                config.BATCHSIZE, 1, *image.shape[2:],
                device="cpu"
            )
            model_uncertainty = torch.zeros(
                config.BATCHSIZE, 1, *image.shape[2:],
                device="cpu"
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
                mini_output, mini_data_uncertainty, mini_model_uncertainty = \
                                        model.inference(mini_image, return_raw=return_raw)
                image_uncertainty += mini_model_uncertainty.mean().item()

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
            data_uncertainty /= slice_overlaps
            model_uncertainty /= slice_overlaps
            image_uncertainty /= slice_overlaps.max().item()
        print(f"{nbatches}: uncertainty = {image_uncertainty}")

        image_cpu = image.data.cpu().numpy()
        output_cpu = output.data.cpu().numpy()
        data_uncertainty_cpu = data_uncertainty.data.cpu().numpy()
        model_uncertainty_cpu = model_uncertainty.data.cpu().numpy()
        label_cpu = label.view(*image.shape).data.cpu().numpy()

        del image, label, mini_image, mini_output, \
            mini_data_uncertainty, mini_model_uncertainty
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

        im_metrics = calculate_metrics(label_cpu, output_cpu, class_names=config.CLASSES)
        metrics = metrics + im_metrics
        uncertainties.append(image_uncertainty)

        # probably visualize
        if visualize:
            visualize_uncertainty_validation(
                image_cpu, (output_cpu, data_uncertainty_cpu, model_uncertainty_cpu), label_cpu,
                cache.out_dir_val,
                class_names=config.CLASSES,
                base_name=f"out_{nbatches}",
            )
    
    metrics /= nbatches + 1
    return metrics, uncertainties



def validate(
    val_dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    cache: RuntimeCache,
    writer: SummaryWriter,
    visualize: bool = True
):

    metrics, uncertainties = inference(val_dataloader, model, criterion, cache, visualize=visualize, return_raw=False)

    # Logging
    accuracy, recall, precision, dice = metrics
    log_iteration_metrics(metrics, steps=cache.epoch, writer=writer, data="validation")
    print(
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
        logging.info(f"Best Dice: {mean_dice}, Epoch: {cache.epoch}")
        cache.best_epoch = cache.epoch
        cache.best_mean_dice = mean_dice
        cache.epochs_no_improvement = 0

        if config.SAVE_MODEL=="best":
            weights = {
                "model": model.state_dict(),
                "epoch": cache.epoch,
                "mean_dice": mean_dice,
            }
            torch.save(weights, os.path.join(cache.out_dir_weights, "best_model.pth"))
    else:
        cache.epochs_no_improvement += 1

    # Store model at end of epoch to get final model (also on failure)
    if config.SAVE_MODEL=="final":
        weights = {
            "model": model.state_dict(),
            "epoch": cache.epoch,
            "mean_dice": mean_dice,
        }
        torch.save(weights, os.path.join(cache.out_dir_weights, "final_model.pth"))
    
    cache.last_epoch_results.update({"best_epoch": cache.best_epoch})
    cache.all_epoch_results.append(cache.last_epoch_results)
    return cache, uncertainties
