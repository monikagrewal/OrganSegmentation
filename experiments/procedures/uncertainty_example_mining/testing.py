import logging
import os

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from torch.utils.data import DataLoader

from experiments.config import Config
from experiments.datasets.amc import AMCDataset
from experiments.models.unet import UNet
from experiments.utils.cache import RuntimeCache
from experiments.utils.metrics import calculate_metrics
from experiments.utils.postprocessing import postprocess_segmentation
from experiments.utils.visualize import visualize_uncertainty_validation


def setup_test(out_dir):
    # Reinitialize config
    config = Config.parse_file(os.path.join(out_dir, "run_parameters.json"))

    # apply validation metrics on training set instead of validation set if train=True

    test_dataset = AMCDataset(
        config.DATA_DIR,
        config.META_PATH,
        classes=config.CLASSES,
        is_training=config.TEST_ON_TRAIN_DATA,
        log_path=None,
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=5
    )

    model = UNet(
        depth=config.MODEL_PARAMS["depth"],
        width=config.MODEL_PARAMS["width"],
        in_channels=1,
        out_channels=len(config.CLASSES),
    )

    model.to(config.DEVICE)
    logging.info("Model initialized for testing")

    # load weights
    state_dict = torch.load(
        os.path.join(config.OUT_DIR_WEIGHTS, "best_model.pth"),
        map_location=config.DEVICE,
    )["model"]
    model.load_state_dict(state_dict)
    logging.info("weights loaded")

    test(model, test_dataloader, config)


def test(
    model: nn.Module, test_dataloader: DataLoader, config: Config, cache: RuntimeCache
):
    """
    Sliding window validation of a model (stored in out_dir) on train or validation set.
    This stores the results in a log file, and visualizations of predictions in the
    out_dir directory
    """
    # validation
    metrics = np.zeros((4, len(config.CLASSES)))
    min_depth = 2 ** config.MODEL_PARAMS["depth"]
    model.eval()
    for nbatches, (image, label) in enumerate(test_dataloader):
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
        if config.VISUALIZE_OUTPUT != "none":
            visualize_uncertainty_validation(
                image,
                (output, data_uncertainty, model_uncertainty),
                label,
                cache.out_dir_test,
                class_names=config.CLASSES,
                base_name=f"out_{nbatches}",
            )

    metrics /= nbatches + 1
    accuracy, recall, precision, dice, haussdorf_distance, surface_dice = metrics
    logging.info(
        f"Test results:\n"
        f"accuracy = {accuracy}\nrecall = {recall}\n"
        f"precision = {precision}\ndice = {dice}\n"
    )
    for class_no, classname in enumerate(config.CLASSES):
        cache.test_results.update(
            {
                f"recall_{classname}": recall[class_no],
                f"precision_{classname}": precision[class_no],
                f"dice_{classname}": dice[class_no],
            }
        )

    return metrics
