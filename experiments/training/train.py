import json
import logging
import os
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from data.load import get_dataloaders, get_datasets
from models import unet, unet_khead
from training.procedures import basic, uncertainty
from training.test import test
from utils.augmentation import get_augmentation_pipelines
from utils.cache import RuntimeCache
from utils.loss import get_criterion
from utils.utilities import create_subfolders


def get_model() -> nn.Module:
    if config.MODEL == "unet":
        return unet.UNet(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_unet":
        return unet_khead.KHeadUNet(**config.MODEL_PARAMS)
    else:
        raise ValueError(f"unknown model: {config.MODEL}")


def get_training_procedure() -> Callable:
    if config.TRAIN_PROCEDURE == "basic":
        return basic.train
    elif config.TRAIN_PROCEDURE == "uncertainty":
        return uncertainty.train
    else:
        raise ValueError(f"unknown training procedure: {config.MODEL}")


def setup_train():
    # Intermediate results storage to pass to other functions to reduce parameters
    cache = RuntimeCache()

    # Print & Store config
    logging.info(config.dict())
    with open(os.path.join(config.OUT_DIR, "run_parameters.json"), "w") as file:
        json.dump(config.dict(), file, indent=4)

    # Set seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False

    # Load datasets
    augmentation_pipelines = get_augmentation_pipelines()
    datasets_list = get_datasets(
        config.NFOLDS,
        config.CLASSES,
        augmentation_pipelines,
        random_seed=config.RANDOM_SEED,
    )

    for i_fold, datasets in enumerate(datasets_list):
        # Create fold folder
        fold_dir = os.path.join(config.OUT_DIR, f"fold{i_fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Set seed again for dataloader reproducibility (probably unnecessary)
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        # Initialize dataloaders
        dataloaders = get_dataloaders(datasets)

        for i_run in range(config.NRUNS):
            # Log
            ntrain, nval = len(datasets["train"]), len(datasets["val"])
            logging.info(
                f"Run: {i_run}, Fold: {i_fold}\n"
                f"Total train dataset: {ntrain}, "
                f"Total validation dataset: {nval}"
            )

            # Create run folder
            run_dir = os.path.join(fold_dir, f"run{i_run}")
            os.makedirs(run_dir, exist_ok=True)

            # Logging of training progress
            writer = SummaryWriter(run_dir)

            #  Create subfolders
            foldernames = config.FOLDERNAMES
            create_subfolders(run_dir, foldernames, cache=cache)

            # Change seed for each run
            torch.manual_seed(config.RANDOM_SEED + i_run)
            np.random.seed(config.RANDOM_SEED + i_run)

            # Initialize parameters
            model = get_model()
            model.to(config.DEVICE)

            criterion = get_criterion()
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.LR,
                weight_decay=config.WEIGHT_DECAY,
                eps=0.001,
            )

            # Mixed precision training scaler
            scaler = torch.cuda.amp.GradScaler()

            # Training
            train_func = get_training_procedure()
            train_func(model, criterion, optimizer, scaler, dataloaders, cache, writer)

            # Testing with best model
            state_dict = torch.load(
                os.path.join(cache.OUT_DIR_WEIGHTS, "best_model.pth"),
                map_location=config.DEVICE,
            )["model"]
            model.load_state_dict(state_dict)
            logging.info("weights loaded")

            test_dataset = deepcopy(datasets["val"])
            test_dataset.transform = None
            test_dataloader = DataLoader(
                test_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=3
            )
            test(model, test_dataloader, config)
