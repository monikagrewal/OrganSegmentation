import argparse
import json
import logging
import os

from config import config
from model.test import test
from model.train import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organ at Risk model training / testing"
    )
    parser.add_argument(
        "--test",
        dest="test",
        action="store_true",
        help="Flag to set mode to testing instead of training",
    )
    args = parser.parse_args()

    if not args.test:
        os.makedirs(config.OUT_DIR_TRAIN, exist_ok=True)
        os.makedirs(config.OUT_DIR_VAL, exist_ok=True)
        os.makedirs(config.OUT_DIR_PROPER_VAL, exist_ok=True)
        os.makedirs(config.OUT_DIR_WEIGHTS, exist_ok=True)
        os.makedirs(config.OUT_DIR_EPOCH_RESULTS, exist_ok=True)

        json.dump(
            config.dict(),
            open(os.path.join(config.OUT_DIR, "run_parameters.json"), "w"),
        )
        train()
    else:
        test()


if __name__ == "__main__":
    main()
