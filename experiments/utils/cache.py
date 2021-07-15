from dataclasses import dataclass, field
from typing import Dict, List, Union

from numpy import number

# Type stub
NumberDict = Dict[str, Union[float, int, number]]


@dataclass
class RuntimeCache:
    """Class for keeping track of intermediate runtime parameters and results."""

    # Defaults
    epoch: int = 0
    epochs_no_improvement: int = 0
    best_mean_dice: float = 0.0
    best_loss: float = 100.0
    last_epoch_results: NumberDict = field(default_factory=dict)
    all_epoch_results: List[NumberDict] = field(default_factory=list)
    train_steps: int = 0
    val_steps: int = 0
