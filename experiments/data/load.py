from typing import Dict, List, Tuple, Union

from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold

from config import config
from data.datasets.amc import AMCDataset
from utils.augmentation import Compose


def get_datasets(
    nfolds: int,
    classes: List[str],
    transform_pipelines: Dict[str, Compose],
    random_seed: Union[None, int],
) -> List[Dict[str, AMCDataset]]:
    """
    Assumption: this function will be used only during training

    nfolds = 0 (all data in train),
            1 (hold out validation set with 80:20 split)
            >=2 (N/nfolds splits, where N is total data)
    """

    full_dataset = AMCDataset(
        config.DATA_DIR,
        config.META_PATH,
        config.SLICE_ANNOT_CSV_PATH,
        classes=classes,
        transform=transform_pipelines.get("train"),
        log_path=None,
    )

    N = len(full_dataset)
    if nfolds == 0:
        datasets_list = [
            {"train": full_dataset, "val": None}
        ]
    elif nfolds == 1:
        indices = np.arange(N)
        np.random.shuffle(indices)
        ntrain = int(N * 0.80)
        train_indices = indices[:ntrain]
        val_indices = indices[ntrain:]

        train_dataset = deepcopy(full_dataset).partition(train_indices)
        val_dataset = deepcopy(full_dataset).partition(val_indices)
        val_dataset.transform = transform_pipelines.get("validation")

        datasets_list = [
            {"train": train_dataset, "val": val_dataset}
        ]
    elif nfolds >= 2:
        datasets_list = []
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_seed)
        for train_indices, val_indices in kf.split(full_dataset):
            train_dataset = deepcopy(full_dataset).partition(train_indices)
            val_dataset = deepcopy(full_dataset).partition(val_indices)
            val_dataset.transform = transform_pipelines.get("validation")
            
            datasets_list.append({"train": train_dataset, "val": val_dataset})
    
    return datasets_list


def get_dataloaders(datasets: Dict[str, DataLoader]
) -> Dict[str, DataLoader]:

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

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
