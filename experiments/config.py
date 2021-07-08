import os
from typing import Optional

import torch
from pydantic import BaseSettings, validator

from cli import cli_args


class Config(BaseSettings):
    # General
    EXPERIMENT_NAME: str = "all_classes"
    MODE: str = "train"
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    CLASSES: list[str] = ["background", "bowel_bag", "bladder", "hip", "rectum"]

    # Data
    DATA_DIR: str = "/export/scratch2/grewal/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split_preprocessed_21-08-2020"  # noqa
    META_PATH: str = "../data_preparation/meta/dataset_train_21-08-2020_slice_annot.csv"

    # Unet
    LOAD_WEIGHTS: bool = False
    MODEL_DEPTH: int = 4  # network depth
    MODEL_WIDTH: int = 64  # network width
    IMAGE_DEPTH: int = 32

    # Preprocessing
    GAMMA: int = 1
    ALPHA: Optional[int] = None
    IMAGE_SCALE_INPLANE: Optional[int] = None
    AUGMENTATION_BRIGHTNESS: dict = dict(p=0.5, rel_addition_range=(-0.2, 0.2))
    AUGMENTATION_CONTRAST: dict = dict(p=0.5, contrast_mult_range=(0.8, 1.2))
    AUGMENTATION_ROTATE3D: dict = dict(
        p=0.3, x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10)
    )

    # Training
    NEPOCHS: int = 100
    BATCHSIZE: int = 1
    ACCUMULATE_BATCHES: int = 1
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    LOSS_FUNCTION: str = "soft_dice"
    CLASS_WEIGHTS: Optional[list[int]] = None
    CLASS_SAMPLE_FREQS: list[int] = [1, 1, 1, 1, 1]  # sample freq weight per class

    # for sliding window validation, overlapping slice windows passed to the model.
    # If true, apply gaussian weighting so that the predictions in center of the window
    # have more weight on the final prediction for a voxel
    SLICE_WEIGHTING: bool = True
    POSTPROCESSING: bool = True

    # Testing
    TEST_ON_TRAIN_DATA: bool = False

    # Folders for logging
    # Base fodlers
    OUT_DIR: str = ""

    @validator("OUT_DIR")
    def set_out_dir(cls, v, values):
        """Dynamically create based on experiment name"""
        return f"../runs/{values['EXPERIMENT_NAME']}"

    # Subdirectories
    OUT_DIR_TRAIN: str = ""
    OUT_DIR_VAL: str = ""
    OUT_DIR_WEIGHTS: str = ""
    OUT_DIR_EPOCH_RESULTS: str = ""
    OUT_DIR_TEST: str = ""

    @validator(
        "OUT_DIR_TRAIN",
        "OUT_DIR_VAL",
        "OUT_DIR_WEIGHTS",
        "OUT_DIR_EPOCH_RESULTS",
        "OUT_DIR_TEST",
    )
    def create_folders(cls, v, values, field):
        """Dynamically create based on experiment name"""
        suffix = field.name.split("_", 2)[-1].lower()

        # Edge case for test folder
        if suffix == "test":
            suffix = ("training" if values["TEST_ON_TRAIN_DATA"] else "test") + (
                "_postprocess" if values["POSTPROCESSING"] else ""
            )

        folder = os.path.join(values["OUT_DIR"], suffix)

        # Also create folder if it don't exist
        os.makedirs(folder, exist_ok=True)
        return folder

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Config(_env_file=cli_args.env_file) if cli_args.env_file else Config()
