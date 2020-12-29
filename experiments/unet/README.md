# Experiment scripts

This directory contains scripts to train a model, train multiple models, and test multiple models.

# Training a model

`train.py` is a script to train a single model and store the results:
* Weights of the best performing model on validation set
* csv with the evaluation metrics on each epoch
* Visualizations of predictions

The `main` function contains hardcoded paths to the preprocessed data directory, and the meta dataframe, which are used to initialize the PyTorch Datasets and Dataloaders.

The python library sacred is used to setup parameters. `train.py` contains a default configuration of parameters that is used when `train.py` is called without any extra options. Sacred allows you to override them when calling the script. For example: 

`python train.py with nepochs=200 lr=0.01 out_dir="./runs/experiment_2"`

to override the `nepochs`, `lr`, and `out_dir` parameters and keep the rest at default. See Sacred documentation for options. 

`device` is a parameter that can be set to run a model on a specific GPU, but because there always seems to be some memory leaking to gpu0 regardless of the device that is set, we tend to keep the device at default 0, and use the environment variable `CUDA_VISIBLE_DEVICES=x` to set a specific GPU. That way Pytorch has no knowledge of the other GPU's and is not able to leak memory. So for example, if you want to run the model above on GPU2, keep device at default 0, and run the script like this:

`CUDA_VISIBLE_DEVICES=2 python main.py with nepochs=200 lr=0.01 out_dir="./runs/experiment_2"`

Pytorch will now only see GPU2, and it will be device 0. 

# Setting up an experiment with multiple models

`create_experiment_list_bash.py` is used to create a text file with bash commands to run each experiment (each line is a bash command for a single experiment).

Examples of experiment definitions of previous experiments are in the `experiment_definitions` directory. 

We then take the very simple approach of running subsets of the experiments in this list using bash. For example, to run the first 10 experiments in one of the experiment definitions on GPU3:

`sed -n 1,10p experiment_definitions/experiments_bash_20-10-2020_with_lr_no_weight_decay.txt  | CUDA_VISIBLE_DEVICES=3 bash`

`sed` will extract the first 10 lines, and pass them to bash with the `CUDA_VISIBLE_DEVICES` environment variabele set to make sure it runs on GPU3. 

We then repeat this process with different subsets of the experiments on different GPUs to divide the runs of an experiment on multiple GPUs. A bit of manual work, but it works...

# Testing models

`test.py` contains code to perform a sliding window validation of a stored model. The basic settings should give equivalent results to the ones that result from the training script. But there are some options to test on the training set instead (for inspecting overfitting), and to perform postprocessing (for each organ select the largest connected region (or 2 regions in the case of hip) in the prediction mask and discard noisy disconnected predictions). 

The bottom of the script defines a list `experiments` where you define a set of output directories of training runs. The script will go over them one by one, load the best model for each run, and evaluate it based on the settings.


# Notebooks
The `notebooks` directory contains two notebooks:
* `2020-09-24_analyse_results.ipynb` to load in the results of an experiment and inspect/analyse them
* `2020-09-10_determine_class_weights.ipynb` to do some analysis of the class frequencies, and come up with some class weight parameters for one of our experiments 