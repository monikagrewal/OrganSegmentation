import json
import logging
import os
from copy import deepcopy

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from data.load import get_dataloaders, get_datasets
from models.unet import UNet
from training.procedures import basic
from training.test import test
from utils.augmentation import get_augmentation_pipelines
from utils.cache import RuntimeCache
from utils.loss import get_criterion
from utils.utilities import create_subfolders


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
        # initialize dataloaders
        dataloaders = get_dataloaders(datasets)

        for i_run in range(config.NRUNS):
            # log
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
            writer = SummaryWriter(config.OUT_DIR)

            #  Create subfolders
            foldernames = config.FOLDERNAMES
            create_subfolders(run_dir, foldernames, cache=cache)

            # Change seed for each run
            torch.manual_seed(config.RANDOM_SEED + i_run)
            np.random.seed(config.RANDOM_SEED + i_run)

            # Initialize parameters
            model = UNet(
                depth=config.MODEL_DEPTH,
                width=config.MODEL_WIDTH,
                in_channels=1,
                out_channels=len(config.CLASSES),
            )
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
            basic.train(model, criterion, optimizer, scaler, dataloaders, cache, writer)

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
