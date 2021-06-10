from dataclasses import dataclass, field


@dataclass
class RuntimeCache:
    """Class for keeping track of intermediate runtime parameters and results."""

    # Defaults
    epoch: int = 0
    epochs_no_improvement: int = 0
    best_mean_dice: float = 0.0
    best_loss: float = 100.0
    last_epoch_results: dict[str, float] = field(default_factory=dict)
    all_epoch_results: list[dict[str, float]] = field(default_factory=list)
    train_steps: int = 0
    val_steps: int = 0
