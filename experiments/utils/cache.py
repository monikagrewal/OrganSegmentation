import os
from dataclasses import dataclass, field
from typing import Dict, List, Union

from numpy import number

# Type stub
NumberDict = Dict[str, Union[float, int, number]]


@dataclass
class RuntimeCache:
    """Class for keeping track of intermediate runtime parameters and results."""

    # Defaults
    out_dir_train: str = ""
    out_dir_val: str = ""
    out_dir_weights: str = ""
    out_dir_epoch_results: str = ""
    out_dir_test: str = ""

    epoch: int = 0
    epochs_no_improvement: int = 0
    best_mean_dice: float = 0.0
    best_loss: float = 100.0
    best_epoch: int = 0
    last_epoch_results: NumberDict = field(default_factory=dict)
    all_epoch_results: List[NumberDict] = field(default_factory=list)
    test_results: NumberDict = field(default_factory=dict)
    train_steps: int = 0
    val_steps: int = 0

    def create_subfolders(self: object,
                        root_folder: str, 
                        foldernames: Dict
                        ) -> None:

        for name, value in foldernames.items():
            folderpath = os.path.join(root_folder, value)
            os.makedirs(folderpath, exist_ok=True)
            self.__setattr__(name, folderpath)
