import os

import numpy as np
import torch
from scipy import signal
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from models.unet import UNet
from utils.cache import RuntimeCache
from utils.metrics import calculate_metrics
from utils.visualize import visualize_output


def validate(
    val_dataloader: DataLoader,
    model: UNet,
    criterion: nn.Module,
    cache: RuntimeCache,
    writer: SummaryWriter,
):
    # validation
    model.eval()
    val_loss = 0.0
    for nbatches, (image, label) in enumerate(val_dataloader):
        image = image.to(config.DEVICE)
        label = label.to(config.DEVICE)
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, label)
            val_loss += loss.item()

            print("Iteration {}: Validation Loss: {}".format(nbatches, loss.item()))
            writer.add_scalar("Loss/validation_loss", loss.item(), cache.val_steps)
            cache.val_steps += 1

            image = image.data.cpu().numpy()
            label = label.view(*image.shape).data.cpu().numpy()
            output = torch.argmax(output, dim=1).view(*image.shape)
            output = output.data.cpu().numpy()

            accuracy, recall, precision, dice = calculate_metrics(
                label, output, class_names=config.CLASSES
            )
            # log metrics
            for class_no, classname in enumerate(config.CLASSES):
                writer.add_scalar(
                    f"accuracy/val/{classname}", accuracy[class_no], cache.val_steps
                )
                writer.add_scalar(
                    f"recall/val/{classname}", recall[class_no], cache.val_steps
                )
                writer.add_scalar(
                    f"precision/val/{classname}",
                    precision[class_no],
                    cache.val_steps,
                )
                writer.add_scalar(
                    f"dice/val/{classname}", dice[class_no], cache.val_steps
                )

        # probably visualize
        if nbatches % 10 == 0:
            visualize_output(
                image[0, 0, :, :, :],
                label[0, 0, :, :, :],
                output[0, 0, :, :, :],
                config.OUT_DIR_VAL,
                class_names=config.CLASSES,
                base_name="out_{}".format(cache.epoch),
            )

    val_loss = val_loss / float(nbatches + 1)
    cache.last_epoch_results.update({"val_loss": val_loss})

    return val_loss


def proper_validate(
    proper_val_dataloader: DataLoader,
    model: UNet,
    cache: RuntimeCache,
    writer: SummaryWriter,
):
    metrics = np.zeros((4, len(config.CLASSES)))
    min_depth = 2 ** config.MODEL_DEPTH
    model.eval()

    for nbatches, (image, label) in enumerate(proper_val_dataloader):
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
                mini_output = model(mini_image)

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

        image = image.data.cpu().numpy()
        output = output.data.cpu().numpy()
        im_metrics = calculate_metrics(label, output, class_names=config.CLASSES)
        metrics = metrics + im_metrics

        # probably visualize
        visualize_output(
            image[0, 0, :, :, :],
            label[0, 0, :, :, :],
            output[0, 0, :, :, :],
            config.OUT_DIR_VAL,
            class_names=config.CLASSES,
            base_name=f"out_{nbatches}",
        )

    metrics /= nbatches + 1

    # Logging
    accuracy, recall, precision, dice = metrics
    print(
        f"Proper evaluation results:\n"
        f"accuracy = {accuracy}\nrecall = {recall}\n"
        f"precision = {precision}\ndice = {dice}\n"
    )
    for class_no, classname in enumerate(config.CLASSES):
        writer.add_scalar(
            f"sw_validation/recall/{classname}", recall[class_no], cache.epoch
        )
        writer.add_scalar(
            f"sw_validation/precision/{classname}", precision[class_no], cache.epoch
        )
        writer.add_scalar(
            f"sw_validation/dice/{classname}", dice[class_no], cache.epoch
        )
        cache.last_epoch_results.update(
            {
                f"recall_{classname}": recall[class_no],
                f"precision_{classname}": precision[class_no],
                f"dice_{classname}": dice[class_no],
            }
        )

    # Store model if best in proper validation
    mean_dice = np.mean(dice[1:])
    cache.last_epoch_results.update({"mean_dice": mean_dice})
    if mean_dice >= cache.best_mean_dice:
        cache.best_mean_dice = mean_dice
        cache.epochs_no_improvement = 0
        weights = {
            "model": model.state_dict(),
            "epoch": cache.epoch,
            "mean_dice": mean_dice,
        }
        torch.save(weights, os.path.join(config.OUT_DIR_WEIGHTS, "best_model.pth"))
    else:
        cache.epochs_no_improvement += 1
