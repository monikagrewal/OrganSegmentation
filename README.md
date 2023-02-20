# Description

This repository contains all the codes related to data preparation, training, and testing for Organs At Risk segmentation.

# Set up

This repository uses `poetry` as a package manager. To set up the environment run the following commands:

- ``curl -sSL https://install.python-poetry.org | python3 -`` to install `poetry`
- ``poetry config virtualenvs.in-project true`` to have poetry install the .venv in this folder
- ``poetry install`` in this working directory to setup your virtual environment
- activate the environment
- git clone `https://github.com/deepmind/surface-distance`
- run `pip install ./surface_distance`

For more info see the Poetry docs: https://python-poetry.org/docs/cli/
