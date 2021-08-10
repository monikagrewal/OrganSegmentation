import json
import logging
import os
from typing import Dict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from data.load import get_datasets, get_dataloaders
from models.unet import UNet
from training.validate import validate
from training.test import test
from utils.augmentation import get_augmentation_pipelines
from utils.cache import RuntimeCache
from utils.loss import get_criterion
from utils.metrics import calculate_metrics
from utils.visualize import visualize_output
from utils.utilities import create_subfolders


def setup_train():
    # Intermediate results storage to pass to other functions to reduce parameters
    cache = RuntimeCache()

    # Print & Store config
    logging.info(config.dict())
    with open(os.path.join(config.OUT_DIR, "run_parameters.json"), "w") as file:
        json.dump(config.dict(), file, indent=4)

    # Set seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False

    # Load datasets
    augmentation_pipelines = get_augmentation_pipelines()
    datasets_list = get_datasets(config.NFOLDS, config.CLASSES,\
                                    augmentation_pipelines,
                                    random_seed=config.RANDOM_SEED)


    for i_fold, datasets in enumerate(datasets_list):
        # Create fold folder
        fold_dir = os.path.join(config.OUT_DIR, f"fold{i_fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Set seed again for dataloader reproducibility (probably unnecessary)
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        # initialize dataloaders
        dataloaders = get_dataloaders(datasets)

        for i_run in range(config.NRUNS):
            # log
            ntrain, nval = len(datasets["train"]), len(datasets["val"])
            logging.info(f"\nRun: {i_run}, Fold: {i_fold}\n \
                Total train dataset: {ntrain} \
                 Total validation dataset: {nval}")

            # Create run folder
            run_dir = os.path.join(fold_dir, f"run{i_run}")
            os.makedirs(run_dir, exist_ok=True)

            # Logging of training progress
            writer = SummaryWriter(config.OUT_DIR)
            
            #  Create subfolders
            foldernames = config.FOLDERNAMES
            create_subfolders(run_dir, foldernames, cache=cache)

            # Change seed for each run
            torch.manual_seed(config.RANDOM_SEED + i_run)
            np.random.seed(config.RANDOM_SEED + i_run)

            # Initialize parameters
            model = UNet(
                depth=config.MODEL_DEPTH,
                width=config.MODEL_WIDTH,
                in_channels=1,
                out_channels=len(config.CLASSES),
            )
            model.to(config.DEVICE)

            criterion = get_criterion()
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.LR,
                weight_decay=config.WEIGHT_DECAY,
                eps=0.001,
            )

            # Mixed precision training scaler
            scaler = torch.cuda.amp.GradScaler()

            # Training
            train(model, criterion, optimizer, scaler, dataloaders, cache, writer)

            # Testing with best model
            state_dict = torch.load(
                os.path.join(cache.OUT_DIR_WEIGHTS, "best_model.pth"),
                map_location=config.DEVICE,
            )["model"]
            model.load_state_dict(state_dict)
            logging.info("weights loaded")

            test_dataset = deepcopy(datasets["val"])
            test_dataset.transform = None
            test_dataloader = DataLoader(
                test_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=3
            )
            test(model, test_dataloader, config)


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    dataloaders: Dict[str, DataLoader],
    cache: RuntimeCache,
    writer: SummaryWriter,
) -> None:
    # Load weights if needed
    if config.LOAD_WEIGHTS:
        weights = torch.load(
            os.path.join(cache.out_dir_weights, "best_model.pth"),
            map_location=config.DEVICE,
        )["model"]
        model.load_state_dict(weights)

    for epoch in range(0, config.NEPOCHS):
        cache.epoch += 1
        cache.last_epoch_results = {"epoch": epoch}
        # Traning step
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        nbatches = 0
        # accumulate gradients over multiple batches (equivalent to bigger batchsize, but without memory issues)  # noqa
        # Note: depending on the reduction method in the loss function, this might need to be divided by the number  # noqa
        #   of accumulation iterations to be truly equivalent to training with bigger batchsize  # noqa
        accumulated_batches = 0
        for nbatches, (image, label) in enumerate(dataloaders["train"]):
            logging.info(f"Image shape: {image.shape}")
            image = image.to(config.DEVICE)
            label = label.to(config.DEVICE)

            with torch.cuda.amp.autocast():
                output = model(image)
                loss = criterion(output, label)

            # to make sure accumulated loss equals average loss in batch
            # and won't depend on accumulation batch size
            loss = loss / config.ACCUMULATE_BATCHES
            scaler.scale(loss).backward()

            if ((nbatches + 1) % config.ACCUMULATE_BATCHES) == 0:
                accumulated_batches += 1
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                train_loss += loss.item()
                logging.info(
                    "Iteration {}: Train Loss: {}".format(nbatches, loss.item())
                )
                writer.add_scalar("Loss/train_loss", loss.item(), cache.train_steps)
                cache.train_steps += 1

                # if not a full batch left, break out of epoch to prevent wasted
                # computation on sample that won't add up to a full batch and
                # therefore won't result in a step
                if (
                    len(dataloaders["train"]) - (nbatches + 1)
                    < config.ACCUMULATE_BATCHES
                ):
                    break

            # Train results
            if (nbatches % config.ACCUMULATE_BATCHES * 3) == 0 or nbatches == len(
                dataloaders["train"]
            ) - 1:
                with torch.no_grad():
                    image = image.data.cpu().numpy()
                    label = label.view(*image.shape).data.cpu().numpy()
                    output = torch.argmax(output, dim=1).view(*image.shape)
                    output = output.data.cpu().numpy()

                    write_train_results(image, label, output, cache, writer)

        train_loss = train_loss / float(accumulated_batches)
        train_acc = train_acc / float(accumulated_batches)
        cache.last_epoch_results.update({"train_loss": train_loss})

        # VALIDATION
        val_dice = validate(dataloaders["val"], model, cache, writer)
        # Store model if best in validation
        if val_dice >= cache.best_mean_dice:
            cache.best_mean_dice = val_dice
            cache.epochs_no_improvement = 0
            weights = {
                "model": model.state_dict(),
                "epoch": cache.epoch,
                "mean_dice": val_dice,
            }
            torch.save(weights, os.path.join(cache.out_dir_weights, "best_model.pth"))
        else:
            cache.epochs_no_improvement += 1

        # Store model at end of epoch to get final model (also on failure)
        weights = {
            "model": model.state_dict(),
            "epoch": cache.epoch,
            "mean_dice": val_dice,
        }
        torch.save(weights, os.path.join(cache.out_dir_weights, "final_model.pth"))

        logging.info(
            f"EPOCH {epoch} = Train Loss: {train_loss}, Validation DICE: {val_dice}\n"
        )
        writer.add_scalar("epoch_loss/train_loss", train_loss, epoch)

        cache.all_epoch_results.append(cache.last_epoch_results)

    # TODO: Validation on Training to get training DICE

    # Store all epoch results
    results_df = pd.DataFrame(cache.all_epoch_results)
    results_df.to_csv(
        os.path.join(cache.out_dir_epoch_results, "epoch_results.csv"), index=False
    )

    writer.close()


def write_train_results(
    image, label, prediction, cache: RuntimeCache, writer: SummaryWriter
) -> None:

    # calculate metrics and probably visualize prediction
    accuracy, recall, precision, dice = calculate_metrics(
        label, prediction, class_names=config.CLASSES
    )

    if config.VISUALIZE_OUTPUT == "all":
        visualize_output(
            image[0, 0, :, :, :],
            label[0, 0, :, :, :],
            prediction[0, 0, :, :, :],
            cache.out_dir_train,
            class_names=config.CLASSES,
            base_name="out_{}".format(cache.epoch),
        )

    # log metrics
    for class_no, classname in enumerate(config.CLASSES):
        writer.add_scalar(
            f"accuracy/train/{classname}",
            accuracy[class_no],
            cache.train_steps,
        )
        writer.add_scalar(
            f"recall/train/{classname}", recall[class_no], cache.train_steps
        )
        writer.add_scalar(
            f"precision/train/{classname}",
            precision[class_no],
            cache.train_steps,
        )
        writer.add_scalar(f"dice/train/{classname}", dice[class_no], cache.train_steps)
