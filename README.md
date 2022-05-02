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
- [] Visualize combined uncertainty map
- [X] Check loss shooting: clipped log sigma square
- [] Upsample foreground patches
- [] Look into using dice loss with uncertainty prediction
- [X] Analyse uncertainty maps