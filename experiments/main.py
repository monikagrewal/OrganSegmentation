from config import config
from training.test import setup_test
from training.train import setup_train
from cli import cli_args

if __name__ == "__main__":
    if not cli_args.test:
        # Run both training and test procedures
        setup_train()
        setup_test(out_dir=config.OUT_DIR)
    else:
        # Only test procedure
        setup_test(out_dir=config.OUT_DIR)
