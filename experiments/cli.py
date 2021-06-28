import argparse

# Add argparse in seperate file for clutter free code (and avoid circular imports)
parser = argparse.ArgumentParser(description="Organ at Risk model training / testing")
parser.add_argument(
    "--only-test",
    dest="test",
    action="store_true",
    help="Flag to set mode to testing instead of training.",
)
parser.add_argument(
    "--env-file",
    dest="env_file",
    type=str,
    default=None,
    help="Set the location of the environment file.",
)

# CLI arguments are currently only used in config file
cli_args = parser.parse_args()
