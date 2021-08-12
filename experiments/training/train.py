import json
import logging
import os

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from config import config
from data.load import get_dataloaders
from models import unet
from training.procedures import basic
from utils.augmentation import get_augmentation_pipelines
from utils.cache import RuntimeCache
from utils.loss import get_criterion


def setup_train():
    # Print & Store config
    logging.info(config.dict())
    with open(os.path.join(config.OUT_DIR, "run_parameters.json"), "w") as file:
        json.dump(config.dict(), file, indent=4)

    # Load datasets
    augmentation_pipelines = get_augmentation_pipelines()
    dataloaders = get_dataloaders(config.CLASSES, augmentation_pipelines)

    # Intermediate results storage to pass to other functions to reduce parameters
    writer = SummaryWriter(config.OUT_DIR)
    cache = RuntimeCache()

    # Initialize parameters
    model = unet.UNet(
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
