import os
from typing import List, Dict, Literal, Optional, Union

import torch
from pydantic import BaseSettings, validator

from cli import cli_args


class Config(BaseSettings):
    # General
    RANDOM_SEED: int = 0
    NRUNS: int = 2
    NFOLDS: Union[None, int] = 5
    @validator("NFOLDS")
    def check(cls, v):
        if v==0:
            raise ValueError("NFOLDS = 0 means nothing")
        return v

    EXPERIMENT_NAME: str = "all_classes"
    MODE: str = "train"
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    CLASSES: List[str] = ["background", "bowel_bag", "bladder", "hip", "rectum"]

    # Data
    DATA_DIR: str = "/export/scratch2/grewal/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split_preprocessed_21-08-2020"  # noqa
    META_PATH: str = "../data_preparation/meta/dataset_train_21-08-2020_slice_annot.csv"
    SLICE_ANNOT_CSV_PATH: str = "../data_preparation/meta/dataset_train_21-08-2020_slice_annot.csv"

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
    CLASS_WEIGHTS: Optional[List[int]] = None
    CLASS_SAMPLE_FREQS: List[int] = [1, 1, 1, 1, 1]  # sample freq weight per class

    # for sliding window validation, overlapping slice windows passed to the model.
    # If true, apply gaussian weighting so that the predictions in center of the window
    # have more weight on the final prediction for a voxel
    SLICE_WEIGHTING: bool = True
    POSTPROCESSING: bool = True

    # Testing
    TEST_ON_TRAIN_DATA: bool = False

    # WHere to perform visualization
    VISUALIZE_OUTPUT: Literal[None, "test", "all"] = None

    # Folders for logging
    # Base fodlers
    OUT_DIR: str = ""

    @validator("OUT_DIR")
    def set_out_dir(cls, v, values):
        """Dynamically create based on experiment name"""
        value = f"../runs/{values['EXPERIMENT_NAME']}"
        os.makedirs(value, exist_ok=True)
        return value

    # Subdirectories
    FOLDERNAMES: dict = {}

    @validator(
        "FOLDERNAMES"
    )
    def create_foldernames(cls, v, values, field):
        """Dynamically create test foldername based on other fields"""
        out_dir_test = ("training" if values["TEST_ON_TRAIN_DATA"] else "test") + (
            "_postprocess" if values["POSTPROCESSING"] else ""
        )

        value = {"out_dir_train": "train",
                "out_dir_val": "val", 
                "out_dir_weights": "weights",
                "out_dir_epoch_results": "epoch_results",
                "out_dir_test": out_dir_test 
            }
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Config(_env_file=cli_args.env_file) if cli_args.env_file else Config()
