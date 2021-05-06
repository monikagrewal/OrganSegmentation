import argparse

from config import config
from training.test import setup_test
from training.train import setup_train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organ at Risk model training / testing"
    )
    parser.add_argument(
        "--only-test",
        dest="test",
        action="store_true",
        help="Flag to set mode to testing instead of training",
    )
    args = parser.parse_args()

    if not args.test:
        # Run both training and test procedures
        setup_train()
        setup_test(out_dir=config.OUT_DIR)
    else:
        # Only test procedure
        setup_test(out_dir=config.OUT_DIR)


if __name__ == "__main__":
    main()
