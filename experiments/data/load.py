from typing import Dict, List, Tuple

from torch.utils.data import DataLoader

from config import config
from data.datasets.amc import AMCDataset
from utils.augmentation import Compose


def get_datasets(
    classes: List[str], transform_pipelines: Dict[str, Compose]
) -> Tuple[AMCDataset, AMCDataset]:
    train_dataset = AMCDataset(
        config.DATA_DIR,
        config.META_PATH,
        classes=classes,
        is_training=True,
        transform=transform_pipelines.get("train"),
        log_path=None,
    )

    # Validation dataset without cropping to do complete sliding window validation
    val_dataset = AMCDataset(
        config.DATA_DIR,
        config.META_PATH,
        classes=classes,
        is_training=False,
        transform=transform_pipelines.get("validation"),
        log_path=None,
    )

    return train_dataset, val_dataset


def get_dataloaders(
    classes: List[str], transform_pipelines: Dict[str, Compose]
) -> Dict[str, DataLoader]:
    train_dataset, val_dataset = get_datasets(classes, transform_pipelines)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=config.BATCHSIZE, num_workers=3
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=3
    )

    return {
        "train": train_dataloader,
        "val": val_dataloader,
    }