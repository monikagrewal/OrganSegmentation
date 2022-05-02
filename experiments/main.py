import logging

from cli import cli_args
from config import config
from setup import setup_train, setup_test

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(f"{config.OUT_DIR}/info.log"),
        logging.StreamHandler(),
    ],
)


if __name__ == "__main__":
    if not cli_args.only_test:
        # Run both training and test procedures
        setup_train()
    else:
        # Only test procedure, make sure to include correct env file via CLI
        setup_test(out_dir=config.OUT_DIR)
