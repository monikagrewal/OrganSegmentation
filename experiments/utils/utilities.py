import os
from typing import Dict, List

from torch.utils.tensorboard import SummaryWriter

from experiments.config import config
from experiments.utils.cache import RuntimeCache
from experiments.utils.metrics import calculate_metrics


def create_subfolders(
    root_folder: str, foldernames: Dict, cache: object = None
) -> None:

    for name, value in foldernames.items():
        folderpath = os.path.join(root_folder, value)
        os.makedirs(folderpath, exist_ok=True)
        if cache is not None:
            cache.__setattr__(name, folderpath)


def log_iteration_metrics(
    metrics, steps: int, writer: SummaryWriter, data: str = "train"
) -> None:

    (
        accuracy,
        recall,
        precision,
        dice,
        haussdorf_distance,
        surface_distance,
    ) = metrics
    # log metrics
    for class_no, classname in enumerate(config.CLASSES):
        writer.add_scalar(
            f"accuracy/{data}/{classname}",
            accuracy[class_no],
            steps,
        )
        writer.add_scalar(f"recall/{data}/{classname}", recall[class_no], steps)
        writer.add_scalar(
            f"precision/{data}/{classname}",
            precision[class_no],
            steps,
        )
        writer.add_scalar(f"dice/{data}/{classname}", dice[class_no], steps)
