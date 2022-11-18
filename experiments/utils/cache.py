import os
from dataclasses import dataclass, field
from typing import Dict, List, Union

from numpy import number

# Type stub
NumberDict = Dict[str, Union[float, int, number]]


@dataclass
class RuntimeCache:
    """Class for keeping track of intermediate runtime parameters and results."""

    def __init__(self, mode:str="train") -> None:
        # Defaults
        if mode=="train":
            self.out_dir_train: str = ""
            self.out_dir_val: str = ""
            self.out_dir_weights: str = ""
            self.out_dir_epoch_results: str = ""

            self.epoch: int = 0
            self.epochs_no_improvement: int = 0
            self.best_mean_dice: float = 0.0
            self.best_loss: float = 100.0
            self.best_epoch: int = 0
            self.last_epoch_results: NumberDict = field(default_factory=dict)
            self.all_epoch_results: List[NumberDict] = field(default_factory=list)
            self.train_steps: int = 0
            self.val_steps: int = 0
        elif mode=="test":
            self.out_dir_test: str = ""
            self.test_results: NumberDict = {}
            self.all_test_results: List[NumberDict] = []
        else:
            raise ValueError("Unknown mode.")

    def create_subfolders(self: object,
                        root_folder: str, 
                        foldernames: Dict
                        ) -> None:

        for name, value in foldernames.items():
            folderpath = os.path.join(root_folder, value)
            os.makedirs(folderpath, exist_ok=True)
            self.__setattr__(name, folderpath)
    
    def set_subfolder_names(self: object,
                        root_folder: str, 
                        foldernames: Dict
                        ) -> None:

        for name, value in foldernames.items():
            folderpath = os.path.join(root_folder, value)
            self.__setattr__(name, folderpath)
