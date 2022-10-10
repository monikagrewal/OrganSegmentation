import logging
from datetime import datetime

from experiments.cli import cli_args
from experiments.config import config
from experiments.setup import setup_test, setup_train

# if someone tried to log something before basicConfig is called, Python creates a default handler that
# goes to the console and will ignore further basicConfig calls. Remove the handler if there is one.
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(f"{config.OUT_DIR}/info.log"),
        logging.StreamHandler(),
    ],
)


if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logging.info(f"Start time: {start_time}")
        if cli_args.env_file:
            # Train model and test on validation dataset
            logging.info("Training model")
            setup_train()
        elif cli_args.out_dir:
            # Run model on test dataset
            logging.info("Testing model")
            setup_test(out_dir=cli_args.out_dir)
        else:
            logging.warning("No env file supplied or out dir specified.")
        end_time = datetime.now()
        logging.info(f"End time: {end_time}. Duration: {end_time - start_time}")
    except Exception as e:
        logging.warning(e)
        raise e
