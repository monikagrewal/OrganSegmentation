import os

import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader

from config import Config
from data.datasets.spleen import SpleenDataset
from model.transform.postprocessing import postprocess_segmentation
from model.unet import UNet
from model.utils.metrics import calculate_metrics
from model.utils.visualize import visualize_output


def test(out_dir, test_on_train=False, postprocess=False):
    """
    Sliding window validation of a model (stored in out_dir) on train or validation set.
    This stores the results in a log file, and visualizations of predictions in the
    out_dir directory
    """
    batchsize = 1

    config = Config.parse_file(os.path.join(out_dir, "run_parameters.json"))
    # filter_label = run_params["filter_label"]
    depth, width, image_depth = (
        config.MODEL_DEPTH,
        config.MODEL_WIDTH,
        config.IMAGE_DEPTH,
    )

    # apply validation metrics on training set instead of validation set if train=True
    if test_on_train:
        out_dir_val = os.path.join(out_dir, "train_final")
    else:
        out_dir_val = os.path.join(out_dir, "test")
    if postprocess:
        out_dir_val = out_dir_val + "_postprocessed"

    os.makedirs(out_dir_val, exist_ok=True)

    val_dataset = SpleenDataset(
        config.DATA_DIR, config.META_PATH, train=test_on_train, log_path=None
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=batchsize, num_workers=5
    )

    model = UNet(
        depth=depth, width=width, in_channels=1, out_channels=len(val_dataset.classes)
    )

    model.to(config.DEVICE)
    print("model initialized")

    # load weights
    state_dict = torch.load(
        os.path.join(config.OUT_DIR_WEIGHTS, "best_model.pth"),
        map_location=config.DEVICE,
    )["model"]
    model.load_state_dict(state_dict)
    print("weights loaded")

    slice_weighting = True
    classes = val_dataset.classes
    # validation
    metrics = np.zeros((4, len(val_dataset.classes)))
    min_depth = 2 ** depth
    model.eval()
    for nbatches, (image, label) in enumerate(val_dataloader):
        label = label.view(*image.shape).data.cpu().numpy()
        with torch.no_grad():
            nslices = image.shape[2]
            image = image.to(config.DEVICE)

            output = torch.zeros(batchsize, len(classes), *image.shape[2:])
            slice_overlaps = torch.zeros(1, 1, nslices, 1, 1)
            start = 0
            while start + min_depth <= nslices:
                if start + image_depth >= nslices:
                    indices = slice(nslices - image_depth, nslices)
                    start = nslices
                else:
                    indices = slice(start, start + image_depth)
                    start += image_depth // 3

                mini_image = image[:, :, indices, :, :]
                mini_output = model(mini_image)
                if slice_weighting:
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
        # print(f'Output shape before pp: {output.shape}')
        if postprocess:
            multiple_organ_indici = [
                idx
                for idx, class_name in enumerate(val_dataset.classes)
                if class_name == "hip"
            ]
            output = postprocess_segmentation(
                output[0, 0],  # remove batch and color channel dims
                n_classes=len(val_dataset.classes),
                multiple_organ_indici=multiple_organ_indici,
                bg_idx=0,
            )
            # return batch & color channel dims
            output = np.expand_dims(np.expand_dims(output, 0), 0)

        # print(f'Output shape after pp: {output.shape}')
        # print(f'n classes: {val_dataset.classes}')
        im_metrics = calculate_metrics(label, output, class_names=val_dataset.classes)
        metrics = metrics + im_metrics

        # probably visualize
        # if nbatches%5==0:
        visualize_output(
            image[0, 0, :, :, :],
            label[0, 0, :, :, :],
            output[0, 0, :, :, :],
            out_dir_val,
            classes=val_dataset.classes,
            base_name="out_{}".format(nbatches),
        )

        # if nbatches >= 0:
        #   break

    metrics /= nbatches + 1
    accuracy, recall, precision, dice = metrics
    print(
        f"Proper evaluation results:\n"
        f"accuracy = {accuracy}\nrecall = {recall}\n"
        f"precision = {precision}\ndice = {dice}\n"
    )

    return metrics
