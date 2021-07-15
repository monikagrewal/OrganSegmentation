import logging
import os

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from torch.utils.data import DataLoader

from config import Config
from data.datasets.amc import AMCDataset
from models.unet import UNet
from utils.metrics import calculate_metrics
from utils.postprocessing import postprocess_segmentation
from utils.visualize import visualize_output


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
        depth=config.MODEL_DEPTH,
        width=config.MODEL_WIDTH,
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


def test(model: nn.Module, test_dataloader: DataLoader, config: Config):
    """
    Sliding window validation of a model (stored in out_dir) on train or validation set.
    This stores the results in a log file, and visualizations of predictions in the
    out_dir directory
    """
    # validation
    metrics = np.zeros((4, len(config.CLASSES)))
    min_depth = 2 ** config.MODEL_DEPTH
    model.eval()
    for nbatches, (image, label) in enumerate(test_dataloader):
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
                    weights = torch.tensor(weights, dtype=torch.float)

                    output[:, :, indices, :, :] += mini_output.to("cpu") * weights.view(
                        1, 1, actual_slices, 1, 1
                    )
                    slice_overlaps[0, 0, indices, 0, 0] += weights
                else:
                    output[:, :, indices, :, :] += mini_output.to("cpu")
                    slice_overlaps[0, 0, indices, 0, 0] += 1

            output = output / slice_overlaps
            output = torch.argmax(output, dim=1).view(*image.shape)

        image = image.data.cpu().numpy()
        output = output.data.cpu().numpy()
        # logging.info(f'Output shape before pp: {output.shape}')
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

        # logging.info(f'Output shape after pp: {output.shape}')
        # logging.info(f'n classes: {val_dataset.classes}')
        im_metrics = calculate_metrics(label, output, class_names=config.CLASSES)
        metrics = metrics + im_metrics

        # probably visualize
        if config.VISUALIZE_OUTPUT in ["test", "all"]:
            visualize_output(
                image[0, 0, :, :, :],
                label[0, 0, :, :, :],
                output[0, 0, :, :, :],
                config.OUT_DIR_TEST,
                class_names=config.CLASSES,
                base_name="out_{}".format(nbatches),
            )

        # if nbatches >= 0:
        #   break

    metrics /= nbatches + 1
    accuracy, recall, precision, dice = metrics
    logging.info(
        f"Test results:\n"
        f"accuracy = {accuracy}\nrecall = {recall}\n"
        f"precision = {precision}\ndice = {dice}\n"
    )

    return metrics
