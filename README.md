# Description
This repository contains all the codes related to data preparation, training, and testing for Organs At Risk segmentation.


# TODO - Geert
- [X] Replace print statements with logging
- [X] Check SummaryWriter
- [X] Read `parameters.json` in test procedure, because defaults may change over time.
- [x] Visualization on/off flag

# TODO - Monika
- [X] Replace data_preparation to include all files for binary segmentation
- [X] training --> test.py --> modify setup_test in context on KFold
- [X] run Kfold experiment for binary (more data) vs. multi-class (less data) segmentation comparison
- [X] analyze results of binary vs. multi-class segmentation

## TODO - March 8, 2022
- [X] Model uncertainty for multi class output (variance is not enough), KL divergence?, entropy? ENTROPY implemented
- [X] Visualize combined uncertainty map
- [X] Check loss shooting: clipped log sigma square
- [X] Upsample foreground patches
- [] Look into using dice loss with uncertainty prediction - DISCARDED
- [X] Analyse uncertainty maps

## Analysis - May 3, 2022
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

## Analysis - May 11, 2022
- Uncertainty weighting increases recall (close to 1) at a cost of precision
- Example mining does not work
- Uncrtainty weighting double is slightly better than uncertainty weighting per class, also slightly better than no uncertainty weighting
- step lr smoother than fixed lr, slightly better too

## TODO - May 11, 2022
- [X] Run uncertainty double with step lr
- [X] Use example mining to add data with missing annotations, write inference differently in validation