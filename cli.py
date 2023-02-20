import argparse

# Add argparse in seperate file for clutter free code (and avoid circular imports)
parser = argparse.ArgumentParser(description="Organ at Risk model training / testing")
parser.add_argument(
    "--env-file",
    dest="env_file",
    type=str,
    default=None,
    help="Set the location of the environment file.",
)
parser.add_argument(
    "--test-env-file",
    dest="test_env_file",
    type=str,
    default=None,
    help="Set the location of the environment file for test config.",
)

cli_args = parser.parse_args()
