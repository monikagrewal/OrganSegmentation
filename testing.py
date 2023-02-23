import logging
import numpy as np
import torch
from scipy import signal
from torch import nn

from config import config
from utils.metrics import calculate_metrics
from utils.postprocessing import postprocess_segmentation
from utils.visualize import visualize_output
from datasets.spleen import *
from models.unet_khead_student import KHeadUNetStudent


def inference(
    image: np.array,
    model: nn.Module,
):
    # convert to tensor
    d, h, w = image.shape
    logging.debug(f"image input to inference: {image.shape}")
    logging.debug(f"image max: {image.max()}, min: {image.min()}")
    image = torch.from_numpy(image.copy()).float().to(config.DEVICE).view(1, 1, d, h, w)
    min_depth = 2**model.depth
    
    with torch.no_grad():
        nslices = image.shape[2]

        output = torch.zeros(
            1, len(config.CLASSES), *image.shape[2:]
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

        output_cpu = output.data.cpu().numpy()
        del image, outputs, mini_image, mini_output
        torch.cuda.empty_cache()

        # Postprocessing
        if config.POSTPROCESSING:
            output_cpu = postprocess_segmentation(
                output_cpu[0, 0],  # remove batch and color channel dims
                n_classes=len(config.CLASSES),
                bg_idx=0,
            )
            # return batch & color channel dims
            output_cpu = np.expand_dims(np.expand_dims(output_cpu, 0), 0)

    return output_cpu


def setup_test():
    # load dataset
    image, label = SpleenDataset.read_nifti_images(config.FILEPATH, labelpath=config.LABELPATH)

    # load model
    model = KHeadUNetStudent(**config.MODEL_PARAMS).to(config.DEVICE)
    logging.info("Model initialized for testing")

    # load weights
    state_dict = torch.load(
        config.WEIGHTS_PATH,
        map_location=config.DEVICE,
    )["model"]

    model.load_state_dict(state_dict)
    logging.info("Weights loaded")
    model.eval()

    output = inference(image, model)

    if config.LABELPATH:
        im_metrics = calculate_metrics(label, output, class_names=config.CLASSES)
        (
            accuracy,
            recall,
            precision,
            dice,
            haussdorf_distance,
            surface_dice,
        ) = im_metrics

        mean_dice = np.mean(dice[1:])
        logging.info(f"class names: {config.CLASSES[1:]}")
        logging.info(f"accuracy = {accuracy[1:]}")
        logging.info(f"recall = {recall[1:]}")
        logging.info(f"precision = {precision[1:]}")
        logging.info(f"dice = {dice[1:]}")
        logging.info(f"hd = {haussdorf_distance[1:]}")
        logging.info(f"sd = {surface_dice[1:]}")
        logging.info(f"mean dice = {mean_dice[1:]}")

        # probably visualize
        if config.VISUALIZE_OUTPUT in ["test", "all"]:
            logging.info("visualizing...")
            visualize_output(
                image,
                output,
                label,
                config.OUT_DIR,
                class_names=config.CLASSES,
                base_name=f"out",
            )


if __name__=="__main__":
    pass
