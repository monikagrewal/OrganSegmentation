# Description

This repository contains all the codes related to data preparation, training, and testing for Organs At Risk segmentation.

# Set up

This repository uses `poetry` as a package manager. To set up the environment run the following commands:

- ``pip install poetry --user`` to install `poetry`
- ``poetry config virtualenvs.in-project true`` to have poetry install the .venv in this folder
- ``poetry install`` in this working directory to setup your virtual environment

For more info see the Poetry docs: https://python-poetry.org/docs/cli/

# To do list

### TODO - Geert

- [x] Replace print statements with logging
- [x] Check SummaryWriter
- [x] Read `parameters.json` in test procedure, because defaults may change over time.
- [x] Visualization on/off flag

### TODO - Monika

- [x] Replace data_preparation to include all files for binary segmentation
- [x] training --> test.py --> modify setup_test in context on KFold
- [x] run Kfold experiment for binary (more data) vs. multi-class (less data) segmentation comparison
- [x] analyze results of binary vs. multi-class segmentation

### TODO - March 8, 2022

- [x] Model uncertainty for multi class output (variance is not enough), KL divergence?, entropy? ENTROPY implemented
- [x] Visualize combined uncertainty map
- [x] Check loss shooting: clipped log sigma square
- [x] Upsample foreground patches
- [] Look into using dice loss with uncertainty prediction - DISCARDED
- [x] Analyse uncertainty maps

### Analysis - May 3, 2022

- LR = 0.001 has higher performance than LR = 0.01 in combination with uncertainty weighted loss
- uncertainty based example mining increases loss suddenly and by the time the networks learns all the hard example, next round of example mining starts. So loss keeps wiggling at the frequency of example mining. PROBABLE REASON: too high selection pressure
- example mining start after EPOCH = 100 without decreasing LR makes loss overshoot (in both cases: initial LR = 0.01 and 0.001)

### Next experiments

- Baseline k-head UNet with fixed LR = 0.001
- Baseline + uncertainty per class weighted loss
- Baseline + uncertainty per class weighted loss + example mining with selection pressure 2 + mining start point EPOCH = 10 + mining frequency 10
- Baseline + uncertainty per class weighted loss + example mining with selection pressure 2 + mining start point EPOCH = 100 + mining frequency 10

- Baseline + uncertainty per class weighted loss + example mining with selection pressure 2 + mining start point EPOCH = 10 + mining frequency 5
- if mining start point 100 is better than 10, then LR SCHEDULER with step size = 100

### Analysis - May 11, 2022

- Uncertainty weighting increases recall (close to 1) at a cost of precision
- Example mining does not work
- Uncrtainty weighting double is slightly better than uncertainty weighting per class, also slightly better than no uncertainty weighting
- step lr smoother than fixed lr, slightly better too

### TODO - May 11, 2022

- [x] Run uncertainty double with step lr
- [x] Use example mining to add data with missing annotations, write inference differently in validation

### Analysis and TODO - May 16, 2022

- Example mining to add data with missing annotation has poorer performance than baseline
- step lr scheduler with uncertainty double gives smoother performance curves, but not better
- [] Dustin: check existing implementation of partial annotation, log which examples are being mined and if they have class imbalance CANCELLED
- [x] Monika: implement uncertainty based data imputation and training more on examples where model is less uncertain
  - Basically, if annotation present: train more on high uncertainty voxels, if annotation absent: train more on low uncertainty voxels after pseudo labelling

### TODO - June 7, 2022

- Make ResU-Net more memory efficient - Dustin
- Implement ResU-Net with resnet34 backbone - Dustin
- Implement ResU-Net stochastic depth student
- Train ResU-Net stochastic depth student with khead UNet teacher, later with khead ResU-Net teacher
