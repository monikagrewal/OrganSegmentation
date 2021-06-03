import os
from typing import Optional

import torch
from pydantic import BaseSettings


class Config(BaseSettings):
    # General
    MODE: str = "train"
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Data
    CLASSES: list[str] = ["background", "spleen"]
    DATA_DIR: str = "/export/scratch3/bvdp/data/Task09_Spleen"
    META_PATH: str = "dataset.json"
    # ROOT_DIR = "/export/scratch2/bvdp/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split_preprocessed_21-08-2020/"  # noqa
    # META_PATH = "/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_21-08-2020_slice_annot.csv"  # noqa

    # Unet
    LOAD_WEIGHTS: bool = False
    MODEL_DEPTH: int = 4  # network depth
    MODEL_WIDTH: int = 64  # network width
    IMAGE_DEPTH: int = 32

    # Preprocessing
    GAMMA: bool = 1
    ALPHA: bool = None
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
    POSTPROCESSING: bool = False

    # Testing
    TEST_ON_TRAIN_DATA: bool = False

    # Logging Directories
    OUT_DIR: str = "../runs/tmp"
    OUT_DIR_TRAIN: str = os.path.join(OUT_DIR, "train")
    OUT_DIR_VAL: str = os.path.join(OUT_DIR, "val")
    OUT_DIR_PROPER_VAL: str = os.path.join(OUT_DIR, "proper_val")
    OUT_DIR_WEIGHTS: str = os.path.join(OUT_DIR, "weights")
    OUT_DIR_EPOCH_RESULTS: str = os.path.join(OUT_DIR, "epoch_results")

    OUT_DIR_TEST: str = os.path.join(
        OUT_DIR,
        ("training" if TEST_ON_TRAIN_DATA else "test")
        + ("_postprocess" if POSTPROCESSING else ""),
    )


config = Config()
