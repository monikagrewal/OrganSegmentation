# Description

This repository contains the source code used in the paper [**Clinically Acceptable Segmentation of Organs at Risk in Cervical Cancer Radiation Treatment from Clinically Available Annotations**](https://arxiv.org/abs/2302.10661).

# Set up

The repository uses `poetry` as a package manager. To set up the environment run the following commands:

- ``curl -sSL https://install.python-poetry.org | python3 -`` to install `poetry`
- ``poetry config virtualenvs.in-project true`` to have poetry install the .venv in this folder
- ``poetry install`` in this working directory to setup your virtual environment
- activate the environment

For more info see the Poetry docs: https://python-poetry.org/docs/cli/

The repository uses the repository `https://github.com/deepmind/surface-distance` for the calculation of surface Dice, and Hausdorff distance. For setup, run the following commands:
- git clone `https://github.com/deepmind/surface-distance`
- run `pip install ./surface_distance`

# Usage

## Use as off-the-shelf method for direct inference
- Either modify the definition (--test-env-file) file in `./definitions/test_on_single_ct` or create a new definition file

- cd to the repository and activate the environment:
    ```
    cd <path-to-repository>
    source .venv/bin/activate
    ```
- Run `python -m main --test-env-file <path-to-test-env-file>`

## Semi-supervised learning (with teacher-student setup, annotation imputation, and uncertainty-guided loss) with your data

- Preprocess data and define a datainfo file (`.csv`, `.json`, or similar format) that describes paths for images, and labels
- Define a custom dataset class in folder datasets. You may use the classes defined in `spleen.py` as template. Basically, define a class for fully annotated dataset and for partially annotated dataset. In each class, implement a __getitem__ method that uses the information provided in datainfo file to load one image and corresponding label at a time.
- Adapt `config.py` -> `DATASET_NAME` and `setup.py` -> `get_datasets` according to the implemented dataset class
- Define a definition file (only specifying the settings that are different from the defaults in `config.py`). Examples of definition files are in `./definitions/`.
- Run `python -m main --env-file <path-to-definition-file>`
- The outputs of training are saved in the folder `./runs/`

## Define a custom model, custom augmentation, and everything else

The main script of the project is `main.py`, which uses `cli.py` to parse command line arguments, `config.py` to load hyperparameters based on the command line arguments, and passes them to `setup.py`. In `setup.py`, the dataset, model, training function are loaded based on the `config` and run. 

