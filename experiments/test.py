import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from experiments.config import config
from experiments.utils.cache import RuntimeCache
from experiments.utils.metrics import calculate_metrics
from experiments.utils.postprocessing_testing import postprocess_segmentation
from experiments.utils.utilities import log_iteration_metrics
from experiments.utils.visualize import visualize_output

idx2class = {
    0: "background",
    1: "bowel_bag",
    2: "bladder",
    3: "hip",
    4: "rectum",
}


def main(
    dataloader: DataLoader,
    model: nn.Module,
    cache: RuntimeCache,
    writer: SummaryWriter,
):
    metrics = np.zeros((6, len(config.CLASSES)))
    min_depth = 2**model.depth
    model.eval()

    for nbatches, (image, label) in enumerate(dataloader):
        cache.test_results = {}

        label = label.view(*image.shape).data.cpu().numpy()
        present_labels = np.unique(label)
        present_labels = "|".join([idx2class[idx] for idx in present_labels][1:])
        cache.test_results.update({"present labels": present_labels})
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
        (
            accuracy,
            recall,
            precision,
            dice,
            haussdorf_distance,
            surface_distance,
        ) = im_metrics
        for class_no, classname in enumerate(config.CLASSES):
            cache.test_results.update(
                {
                    f"recall_{classname}": recall[class_no],
                    f"precision_{classname}": precision[class_no],
                    f"dice_{classname}": dice[class_no],
                    f"hd_{classname}": haussdorf_distance[class_no],
                    f"sd_{classname}": surface_distance[class_no],
                }
            )

        mean_dice = np.mean(dice[1:])
        cache.test_results.update({"mean_dice": mean_dice})
        cache.all_test_results.append(cache.test_results)
        logging.info(f"Scan {nbatches}:")
        logging.info(f"accuracy = {accuracy}")
        logging.info(f"recall = {recall}")
        logging.info(f"precision = {precision}")
        logging.info(f"dice = {dice}")
        logging.info(f"hd = {haussdorf_distance}")
        logging.info(f"sd = {surface_distance}")

        log_iteration_metrics(im_metrics, steps=nbatches, writer=writer, data="test")
        metrics = metrics + im_metrics

        # probably visualize
        if config.VISUALIZE_OUTPUT in ["test", "all"]:
            print("visualizing...")
            visualize_output(
                image_cpu,
                output_cpu,
                label,
                cache.out_dir_test,
                class_names=config.CLASSES,
                base_name=f"out_{nbatches}",
            )

    metrics /= nbatches + 1

    accuracy, recall, precision, dice, haussdorf_distance, surface_distance = metrics
    mean_dice = np.mean(dice[1:])
    logging.info(f"Total Results:")
    logging.info(f"accuracy = {accuracy}")
    logging.info(f"recall = {recall}")
    logging.info(f"precision = {precision}")
    logging.info(f"dice = {dice}")
    logging.info(f"hd = {haussdorf_distance}")
    logging.info(f"sd = {surface_distance}")

    logging.info(f"mean dice: {mean_dice}")

    # Store test results
    results_df = pd.DataFrame(cache.all_test_results)
    results_df.to_csv(os.path.join(cache.out_dir_test, "test_results.csv"), index=False)
    writer.close()

    return None
