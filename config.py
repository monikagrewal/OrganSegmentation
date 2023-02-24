import logging
import os
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseSettings, validator

from cli import cli_args


class Config(BaseSettings):
    # General
    EXPERIMENT_NAME: str = "test_experiment"
    MODE: str = "train"
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    CLASSES: List[str] = ["background", "bowel_bag", "bladder", "hip", "rectum"]
    DEBUG: bool = False

    RANDOM_SEED: int = 20220903
    NRUNS: int = 1
    NFOLDS: Optional[int] = 1

    # Data
    DATASET_NAME: Literal["SpleenDataset"] = "SpleenDataset" # noqa
    DATA_DIR: str = "../Data/Task09_Spleen"  # noqa
    DATASET_PARAMS: Dict[str, Any] = {}
    DATASET_TYPE: Literal["fully_annotated", "partially_annotated"] = "fully_annotated"

    # Unet
    MODEL: Literal[
        "unet",
        "khead_unet",
        "khead_unet_student",
    ] = "unet"
    LOAD_WEIGHTS: bool = False
    WEIGHTS_PATH: str = ""
    IMAGE_DEPTH: int = 32

    # Model params can be added in env file based on chosen model
    MODEL_PARAMS: Dict[str, Any] = {}

    @validator("MODEL_PARAMS")
    def set_dynamic_model_params(cls, v, values):
        v["out_channels"] = len(values["CLASSES"])
        return v

    # Preprocessing
    AUGMENTATION_BRIGHTNESS: dict = dict(p=0.5, rel_addition_range=(-0.2, 0.2))
    AUGMENTATION_CONTRAST: dict = dict(p=0.5, contrast_mult_range=(0.8, 1.2))
    AUGMENTATION_ROTATE3D: dict = dict(
        p=0.3, x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10)
    )
    AUGMENTATION_LIST: Dict[str, dict] = {}

    # Training
    TRAIN_PROCEDURE: Literal[
        "basic", "uncertainty", "partial_annotation"
    ] = "basic"

    @validator("TRAIN_PROCEDURE")
    def check_train_procedure(cls, v, values):
        model = values["MODEL"]
        if "uncertainty" in v and model == "unet":
            raise ValueError(f"TRAIN_PROCEDURE = {v} not valid for MODEL = {model}")

        return_uncertainty = values["MODEL_PARAMS"].get("return_uncertainty", False)
        if v == "basic" and return_uncertainty:
            raise ValueError(f"TRAIN_PROCEDURE = 'basic' not valid for MODEL = {model}")
        return v

    NEPOCHS: int = 100
    BATCHSIZE: int = 1
    ACCUMULATE_BATCHES: int = 1
    OPTIMIZER: Literal["SGD", "Adam"] = "Adam"
    OPTIMIZER_PARAMS: dict = {}
    LR: float = 1e-3
    LR_SCHEDULER: Literal["step_lr"] = "step_lr"
    LR_SCHEDULER_ARGS: Dict = {"step_size": 33, "gamma": 0.1}
    WEIGHT_DECAY: float = 1e-4

    LOSS_FUNCTION: Literal[
        "cross_entropy",
        "partial_annotation_impute",
    ] = "cross_entropy"

    @validator("LOSS_FUNCTION")
    def check_loss_function(cls, v, values):
        if "partial" in values["TRAIN_PROCEDURE"] and "partial" not in v:
            raise ValueError(
                f"LOSS_FUNCTION = {v} not valid for TRAIN_PROCEDURE = 'partial_annotation'"
            )
        return v

    LOSS_FUNCTION_ARGS: Dict = dict()

    # for sliding window validation, overlapping slice windows passed to the model.
    # If true, apply gaussian weighting so that the predictions in center of the window
    # have more weight on the final prediction for a voxel
    SLICE_WEIGHTING: bool = True
    POSTPROCESSING: bool = True

    # Where to perform visualization
    VISUALIZE_OUTPUT: Literal["none", "val", "test", "all"] = "none"
    SAVE_MODEL: Literal["none", "best", "final"] = "best"

    # Folders for logging
    # Base folders
    OUT_DIR: str = ""

    @validator("OUT_DIR")
    def set_out_dir(cls, v, values):
        """Dynamically create based on experiment name"""
        t0 = datetime.now()
        t0_str = datetime.strftime(t0, "%d%m%Y_%H%M%S")
        value = f"runs/{values['EXPERIMENT_NAME']}_{t0_str}"
        return value

    # Subdirectories
    FOLDERNAMES: Dict[str, str] = {
        "out_dir_train": "train",
        "out_dir_val": "val",
        "out_dir_weights": "weights",
        "out_dir_epoch_results": "epoch_results",
    }

    @validator("FOLDERNAMES")
    def add_test_foldername(cls, v, values):
        """Dynamically create test foldername based on other fields"""
        out_dir_test = "test" + (
            "_postprocess" if values["POSTPROCESSING"] else ""
        )

        v["out_dir_test"] = out_dir_test
        return v


class TestConfig(BaseSettings):
    # General
    DEBUG: bool = True
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # model
    CLASSES: List[str] = ["background", "bowel_bag", "bladder", "hip", "rectum"]
    MODEL_PARAMS: Dict[str, Any] = {}
    @validator("MODEL_PARAMS")
    def set_dynamic_model_params(cls, v, values):
        v["out_channels"] = len(values["CLASSES"])
        return v

    IMAGE_DEPTH: int = 32
    WEIGHTS_PATH = "./pretrained_models/basic-teacher-robust-student-f0r0.pth"

    # Data
    FILEPATH = "../Data/Task09_Spleen/imagesTr/spleen_3.nii.gz"
    LABELPATH = "../Data/Task09_Spleen/labelsTr/spleen_3.nii.gz"
    SLICE_WEIGHTING: bool = True
    POSTPROCESSING: bool = True

    # Whether to perform visualization
    VISUALIZE_OUTPUT: bool = False
    OUT_DIR: str = "./outputs/"



def get_config(env_file=cli_args.env_file, test_env_file=cli_args.test_env_file):

    if not (env_file or test_env_file):
        print("No env_file supplied. " "Creating default config")
        return Config()
    elif env_file and not test_env_file:
        env_path = Path(env_file).expanduser()
        if env_path.is_file():
            print("Creating config based on file")
            return Config(_env_file=env_file)
        else:
            print(
                "env_file supplied but does not resolve to a file. "
                "Creating default config"
            )
            return Config()
    elif not env_file and test_env_file:
        env_path = Path(test_env_file).expanduser()
        if env_path.is_file():
            print("Creating test config based on file")
            return TestConfig(_env_file=env_file)
        else:
            print(
                "test_env_file supplied but does not resolve to a file. "
                "Exiting..."
            )
            return None
    elif env_file and test_env_file:
        print("Both env file and test env file supplied. Exiting...")
        return None


config = get_config()
