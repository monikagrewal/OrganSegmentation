import json
import logging
import os
from copy import deepcopy
from typing import Callable, Dict, List, Union

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter

from experiments.config import Config, config
from experiments.datasets.amc import *
from experiments.models import (
    resunet,
    resunet_khead,
    unet,
    unet_khead,
    unet_khead_student,
    unet_khead_uncertainty,
)
from experiments.procedures.basic import testing as basic_testing
from experiments.procedures.basic import training as basic_training
from experiments.procedures.basic import validation as basic_validation
from experiments.procedures.partial_annotation import testing as partial_testing
from experiments.procedures.partial_annotation import training as partial_training
from experiments.procedures.partial_annotation import validation as partial_validation
from experiments.procedures.uncertainty import testing as uncertainty_testing
from experiments.procedures.uncertainty import training as uncertainty_training
from experiments.procedures.uncertainty import validation as uncertainty_validation
from experiments.procedures.uncertainty_example_mining import testing as mining_testing
from experiments.procedures.uncertainty_example_mining import (
    training as mining_training,
)
from experiments.procedures.uncertainty_example_mining import (
    validation as mining_validation,
)
from experiments.utils.augmentation import *
from experiments.utils.cache import RuntimeCache
from experiments.utils.loss import *


def get_augmentation_pipelines() -> Dict[str, Compose]:
    # Random augmentations
    transform_any = ComposeAnyOf([])
    if config.AUGMENTATION_BRIGHTNESS:
        logging.debug(
            "Adding random brightness augmentation with params: "
            f"{config.AUGMENTATION_BRIGHTNESS}"
        )
        transform_any.transforms.append(
            RandomBrightness(**config.AUGMENTATION_BRIGHTNESS)
        )
    if config.AUGMENTATION_CONTRAST:
        logging.debug(
            "Adding random contrast augmentation with params: "
            f"{config.AUGMENTATION_CONTRAST}"
        )
        transform_any.transforms.append(RandomContrast(**config.AUGMENTATION_CONTRAST))
    if config.AUGMENTATION_ROTATE3D:
        logging.debug(
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
    nfolds: int, classes: List[str], transform_pipelines: Dict[str, Compose]
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
    if nfolds == 0:
        logging.info(f"NFolds = {nfolds}: Full dataset")
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
        logging.info(f"NFolds = {1}: Single holdout")
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
        logging.info(f"NFolds = {nfolds}: K-Fold")
        datasets_list = []
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=config.RANDOM_SEED)
        for train_indices, val_indices in kf.split(full_dataset):
            train_dataset = deepcopy(full_dataset).partition(train_indices)
            val_dataset = deepcopy(full_dataset).partition(val_indices)
            val_dataset.transform = transform_pipelines.get("validation")

            datasets_list.append({"train": train_dataset, "val": val_dataset})

    if config.DATASET_NAME == "AMCDatasetPartialAnnotation":
        logging.info(f"Including partial annotated")
        META_PATH = "data_preparation/meta/dataset_train_21-08-2020.csv"
        partial_dataset = AMCDatasetPartialAnnotation(
            config.DATA_DIR,
            META_PATH,
            config.SLICE_ANNOT_CSV_PATH,
            classes=classes,
            transform=transform_pipelines.get("train"),
            log_path=None,
        )
        for i, item in enumerate(datasets_list):
            new_partial_dataset = deepcopy(partial_dataset).add_samples(
                datasets_list[i]["train"]
            )
            datasets_list[i]["train"] = new_partial_dataset

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
    elif config.MODEL == "resunet":
        return resunet.ResUNet(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_unet":
        return unet_khead.KHeadUNet(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_resunet":
        return resunet_khead.KHeadResUNet(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_unet_uncertainty":
        return unet_khead_uncertainty.KHeadUNetUncertainty(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_unet_student":
        return unet_khead_student.KHeadUNetStudent(**config.MODEL_PARAMS)
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

    elif config.LOSS_FUNCTION == "uncertainty_weighted_double":
        criterion = UncertaintyWeightedDoubleLoss(**config.LOSS_FUNCTION_ARGS)

    elif config.LOSS_FUNCTION == "partial_annotation":
        criterion = PartialAnnotationLoss(**config.LOSS_FUNCTION_ARGS)

    elif config.LOSS_FUNCTION == "partial_annotation_impute":
        criterion = PartialAnnotationImputeLoss(**config.LOSS_FUNCTION_ARGS)

    else:
        raise NotImplementedError(
            f"loss function: {config.LOSS_FUNCTION} not implemented yet."
        )

    return criterion


def get_optimizer(model: nn.Module) -> Callable:
    if config.OPTIMIZER == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
            **config.OPTIMIZER_PARAMS,
        )
    elif config.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
            eps=0.001,
            **config.OPTIMIZER_PARAMS,
        )
    return optimizer


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


def get_training_procedures() -> List[Callable]:
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
    elif config.TRAIN_PROCEDURE == "partial_annotation":
        train = partial_training.train
        validate = partial_validation.validate
        test = partial_testing.test
    else:
        raise ValueError(f"Unknown TRAIN_PROCEDURE: {config.TRAIN_PROCEDURE}")
    return train, validate, test


def setup_train():

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
    datasets_list = get_datasets(config.NFOLDS, config.CLASSES, augmentation_pipelines)

    for i_fold, datasets in enumerate(datasets_list):
        # Create fold folder
        fold_dir = os.path.join(config.OUT_DIR, f"fold{i_fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Set seed again for dataloader reproducibility (probably unnecessary)
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        dataloaders = get_dataloaders(datasets)

        for i_run in range(config.NRUNS):
            ntrain, nval = len(datasets["train"]), len(datasets["val"])
            logging.info(f"Run: {i_run}, Fold: {i_fold}")
            logging.info(f"Total train dataset: {ntrain}")
            logging.info(f"Total validation dataset: {nval}")

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
            optimizer = get_optimizer(model)
            lr_scheduler_fn = get_lr_scheduler()
            lr_scheduler = lr_scheduler_fn(optimizer, **config.LR_SCHEDULER_ARGS)

            # Mixed precision training scaler
            scaler = torch.cuda.amp.GradScaler()

            # Training
            train(
                model,
                criterion,
                optimizer,
                lr_scheduler,
                scaler,
                dataloaders,
                cache,
                writer,
            )

            # Delete cache in the end. Q. is it necessary?
            del cache


def setup_test(out_dir):

    # Reinitialize config
    config = Config.parse_file(os.path.join(out_dir, "run_parameters.json"))

    # Set seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False

    test_dataset = AMCDataset(
        config.DATA_DIR_TEST,  # Should take test data dir
        config.META_PATH_TEST,
        config.SLICE_ANNOT_CSV_PATH_TEST,
        classes=config.CLASSES,
        transform=None,
        log_path=None,
    )

    dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=5
    )

    if config.MODEL == "unet":
        model = unet.UNet(**config.MODEL_PARAMS)
    elif config.MODEL == "resunet":
        model = resunet.ResUNet(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_unet":
        model = unet_khead.KHeadUNet(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_resunet":
        model = resunet_khead.KHeadResUNet(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_unet_uncertainty":
        model = unet_khead_uncertainty.KHeadUNetUncertainty(**config.MODEL_PARAMS)
    elif config.MODEL == "khead_unet_student":
        model = unet_khead_student.KHeadUNetStudent(**config.MODEL_PARAMS)
    else:
        raise ValueError(f"unknown model: {config.MODEL}")

    model.to(config.DEVICE)
    logging.info("Model initialized for testing")

    # load weights
    weights_dir = os.path.join(
        out_dir, "fold0/run0", config.FOLDERNAMES["out_dir_weights"], "final_model.pth"
    )

    state_dict = torch.load(
        weights_dir,
        map_location=config.DEVICE,
    )["model"]

    model.load_state_dict(state_dict)
    logging.info("Weights loaded")

    _, __, test = get_training_procedures()

    test(model, dataloader, config)
