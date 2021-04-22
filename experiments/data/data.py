from torch.utils.data import DataLoader

from config import config
from data.datasets.spleen import SpleenDataset
from data.datasets.amc import AMCDataset
from model.transform.transformers import Compose


def get_datasets(
    transform_pipelines: dict[str, Compose]
) -> tuple[SpleenDataset, SpleenDataset]:
    train_dataset = AMCDataset(
        config.DATA_DIR,
        config.META_PATH,
        is_training=True,
        transform=transform_pipelines.get("train"),
        log_path=None,
    )
    val_dataset = AMCDataset(
        config.DATA_DIR,
        config.META_PATH,
        is_training=False,
        transform=transform_pipelines.get("validation"),
        log_path=None,
    )

    # dataset without cropping to do complete sliding window validation over the scan
    proper_val_dataset = AMCDataset(
        config.DATA_DIR,
        config.META_PATH,
        is_training=False,
        transform=transform_pipelines.get("valation_sliding"),
        log_path=None,
    )

    return train_dataset, val_dataset, proper_val_dataset


def get_dataloaders(transform_pipelines: dict[str, Compose]):
    train_dataset, val_dataset, proper_val_dataset = get_datasets(transform_pipelines)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=config.BATCHSIZE, num_workers=3
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=3
    )
    proper_val_dataloader = DataLoader(
        proper_val_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=3
    )

    return train_dataloader, val_dataloader, proper_val_dataloader
