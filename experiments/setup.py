import json
import logging
import os
from copy import deepcopy
from typing import Callable, Dict, List, Union

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from config import Config, config
from datasets.amc import *
from models import unet, unet_khead, unet_khead_uncertainty
from procedures.basic import training as basic_training,\
                            validation as basic_validation,\
                            testing as basic_testing
from procedures.uncertainty import training as uncertainty_training,\
                                    validation as uncertainty_validation,\
                                    testing as uncertainty_testing
from procedures.uncertainty_example_mining import training as mining_training,\
                                                validation as mining_validation,\
                                                    testing as mining_testing
from utils.augmentation import *
from utils.cache import RuntimeCache
from utils.loss import *


def get_augmentation_pipelines() -> Dict[str, Compose]:
    # Random augmentations
    transform_any = ComposeAnyOf([])
    if config.AUGMENTATION_BRIGHTNESS:
        logging.info(
            "Adding random brightness augmentation with params: "
            f"{config.AUGMENTATION_BRIGHTNESS}"
        )
        transform_any.transforms.append(
            RandomBrightness(**config.AUGMENTATION_BRIGHTNESS)
        )
    if config.AUGMENTATION_CONTRAST:
        logging.info(
            "Adding random contrast augmentation with params: "
            f"{config.AUGMENTATION_CONTRAST}"
        )
        transform_any.transforms.append(RandomContrast(**config.AUGMENTATION_CONTRAST))
    if config.AUGMENTATION_ROTATE3D:
        logging.info(
            "Adding random rotate3d augmentation with params: "
            f"{config.AUGMENTATION_ROTATE3D}"
        )
        transform_any.transforms.append(RandomRotate3D(**config.AUGMENTATION_ROTATE3D))

    # Training pipeline
    transform_train = Compose(
        [
            transform_any,
            CropDepthwise(crop_size=config.IMAGE_DEPTH, crop_mode="random"),
        ]
    )

    # Validation pipelines
    transform_val_sliding_window = Compose(
        [
            # CustomResize(output_size=image_size),
            # CropInplane(crop_size=crop_inplane, crop_mode='center'),
        ]
    )

    # temporary addition to test inplance scaling
    if config.IMAGE_SCALE_INPLANE is not None:
        transform_train.transforms.append(
            CustomResize(scale=config.IMAGE_SCALE_INPLANE)
        )
        transform_val_sliding_window.transforms.append(
            CustomResize(scale=config.IMAGE_SCALE_INPLANE)
        )

    return {
        "train": transform_train,
        "validation": transform_val_sliding_window,
    }


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

    # No folds, return full dataset
    if nfolds is None:
        indices = np.arange(N)
        train_dataset = deepcopy(full_dataset).partition(indices)
        val_dataset = deepcopy(full_dataset).partition([])
        datasets_list = [
            {
                "train": train_dataset,
                "val": val_dataset,
            }
        ]

    # Basic single holdout validation
    elif nfolds == 1:
        indices = np.arange(N)
        np.random.shuffle(indices)
        ntrain = int(N * 0.80)
        train_indices = indices[:ntrain]
        val_indices = indices[ntrain:]

        train_dataset = deepcopy(full_dataset).partition(train_indices)
        val_dataset = deepcopy(full_dataset).partition(val_indices)
        val_dataset.transform = transform_pipelines.get("validation")

        datasets_list = [{"train": train_dataset, "val": val_dataset}]

    # K-Fold
    elif nfolds >= 2:
        datasets_list = []
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_seed)
        for train_indices, val_indices in kf.split(full_dataset):
            train_dataset = deepcopy(full_dataset).partition(train_indices)
            val_dataset = deepcopy(full_dataset).partition(val_indices)
            val_dataset.transform = transform_pipelines.get("validation")

            datasets_list.append({"train": train_dataset, "val": val_dataset})

    return datasets_list


def get_dataloaders(datasets: Dict[str, Dataset]) -> Dict[str, DataLoader]:

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


def get_model() -> nn.Module:
    if config.MODEL == "unet":
        return unet.UNet(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_unet":
        return unet_khead.KHeadUNet(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_unet_uncertainty":
        return unet_khead_uncertainty.KHeadUNetUncertainty(**config.MODEL_PARAMS)
    else:
        raise ValueError(f"unknown model: {config.MODEL}")


def get_criterion() -> nn.Module:
    criterion: nn.Module
    if config.LOSS_FUNCTION == "cross_entropy":
        criterion = nn.CrossEntropyLoss(**config.LOSS_FUNCTION_ARGS)

    elif config.LOSS_FUNCTION == "soft_dice":
        raise Warning(f"loss function soft_dice not tested yet.")
        criterion = SoftDiceLoss(**config.LOSS_FUNCTION_ARGS)

    elif config.LOSS_FUNCTION == "uncertainty":
        criterion = UncertaintyLoss(**config.LOSS_FUNCTION_ARGS)

    elif config.LOSS_FUNCTION == "uncertainty_weighted":
        criterion = UncertaintyWeightedLoss(**config.LOSS_FUNCTION_ARGS)

    elif config.LOSS_FUNCTION == "uncertainty_weighted_class":
        criterion = UncertaintyWeightedPerClassLoss(**config.LOSS_FUNCTION_ARGS)

    else:
        raise NotImplementedError(f"loss function: {config.LOSS_FUNCTION} not implemented yet.")

    return criterion


def get_lr_scheduler() -> Callable:
    scheduler = optim.lr_scheduler
    if config.LR_SCHEDULER == "step_lr":
        scheduler = optim.lr_scheduler.StepLR
    elif config.LR_SCHEDULER == "cyclic_lr":
        scheduler = optim.lr_scheduler.CyclicLR
    elif config.LR_SCHEDULER == "multi_step_lr":
        scheduler = optim.lr_scheduler.MultiStepLR
    elif config.LR_SCHEDULER == "cosine_annealing_lr":
        scheduler = optim.lr_scheduler.CosineAnnealingLR
    elif config.LR_SCHEDULER == "cosine_annealing_restart":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts
    else:
        raise ValueError(f"Unknown lr scheduler: {config.LR_SCHEDULER}")
    return scheduler


def get_training_procedures() -> \
    List[Callable]:
    if config.TRAIN_PROCEDURE == "basic":
        train = basic_training.train
        validate = basic_validation.validate
        test = basic_testing.test
    elif config.TRAIN_PROCEDURE == "uncertainty":
        train = uncertainty_training.train
        validate = uncertainty_validation.validate
        test = uncertainty_testing.test
    elif config.TRAIN_PROCEDURE == "uncertainty_example_mining":
        train = mining_training.train
        validate = mining_validation.validate
        test = mining_testing.test
    else:
        raise ValueError(f"Unknown TRAIN_PROCEDURE: {config.TRAIN_PROCEDURE}")
    return train, validate, test


def setup_train():
    # Print & Store config
    print(config)
    with open(os.path.join(config.OUT_DIR, "run_parameters.json"), "w") as file:
        json.dump(config.dict(), file, indent=4)

    # get train procedures
    train, validate, test = get_training_procedures()

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
            print(
                f"Run: {i_run}, Fold: {i_fold}\n"
                f"Total train dataset: {ntrain}, "
                f"Total validation dataset: {nval}"
            )

            # Create run folder and set-up run dir
            run_dir = os.path.join(fold_dir, f"run{i_run}")
            os.makedirs(run_dir, exist_ok=True)

            # Intermediate results storage to pass to other functions to reduce parameters
            cache = RuntimeCache()
            #  Create subfolders
            foldernames = config.FOLDERNAMES
            cache.create_subfolders(run_dir, foldernames)
            # Logging of training progress
            writer = SummaryWriter(run_dir)

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
            lr_scheduler_fn = get_lr_scheduler()
            lr_scheduler = lr_scheduler_fn(optimizer, **config.LR_SCHEDULER_ARGS)

            # Mixed precision training scaler
            scaler = torch.cuda.amp.GradScaler()

            # Training
            train(model, criterion, optimizer, lr_scheduler, scaler, dataloaders, cache, writer)

            # Testing with best model
            if config.SAVE_MODEL != "none":
                model_filename = f"{config.SAVE_MODEL}_model.pth"
                state_dict = torch.load(
                    os.path.join(cache.out_dir_weights, model_filename),
                    map_location=config.DEVICE,
                )["model"]
                model.load_state_dict(state_dict)
                if config.DEBUG:
                    logging.info("weights loaded")

                test_dataset = deepcopy(datasets["val"])
                test_dataset.transform = None
                test_dataloader = DataLoader(
                    test_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=3
                )
                test(model, test_dataloader, config, cache)

            # Delete cache in the end. Q. is it necessary?
            del cache


def setup_test(out_dir):
    # Reinitialize config
    config = Config.parse_file(os.path.join(out_dir, "run_parameters.json"))

    # apply validation metrics on training set instead of validation set if train=True

    test_dataset = AMCDataset(
        config.DATA_DIR,
        config.META_PATH,
        classes=config.CLASSES,
        is_training=config.TEST_ON_TRAIN_DATA,
        log_path=None,
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=5
    )

    model = unet.UNet(
        depth=config.MODEL_PARAMS["depth"],
        width=config.MODEL_PARAMS["width"],
        in_channels=1,
        out_channels=len(config.CLASSES),
    )

    model.to(config.DEVICE)
    logging.info("Model initialized for testing")

    # load weights
    state_dict = torch.load(
        os.path.join(config.OUT_DIR_WEIGHTS, "best_model.pth"),
        map_location=config.DEVICE,
    )["model"]
    model.load_state_dict(state_dict)
    logging.info("weights loaded")

    test(model, test_dataloader, config)