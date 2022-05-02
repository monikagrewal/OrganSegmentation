import json
import logging
import os
import random
from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd
import torch
from config import config
from procedures.uncertainty_example_mining.validation import (inference,
                                                              validate)
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.augmentation import *
from utils.cache import RuntimeCache
from utils.metrics import calculate_metrics
from utils.utilities import log_iteration_metrics
from utils.visualize import visualize_uncertainty_training


def hard_example_sampler(indices_by_ranks, number_of_samples, selection_pressure):
    N = len(indices_by_ranks)
    x = np.arange(0, N)
    x_exp = np.exp(-x * np.log(selection_pressure) / N)

    return random.choices(indices_by_ranks, x_exp, k=number_of_samples)


START_EPOCH_EXAMPLE_MINING = 100  # Should be a multiple of EXAMPLE_MINING_FREQ
EXAMPLE_MINING_FREQ = 10


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: lr_scheduler,
    scaler: GradScaler,
    dataloaders: Dict[str, DataLoader],
    cache: RuntimeCache,
    writer: SummaryWriter,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)

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
        nbatches = 0

        train_dataset_copy = deepcopy(dataloaders["train"].dataset)
        train_dataset_copy.transform = dataloaders["val"].dataset.transform
        if (
            (epoch != 0)
            and (epoch != (config.NEPOCHS - 1))
            and (epoch >= START_EPOCH_EXAMPLE_MINING)
        ):
            if epoch % EXAMPLE_MINING_FREQ == 0:
                _, losses = inference(
                    train_dataset_copy,
                    model,
                    criterion,
                    cache,
                    visualize=False,
                    return_raw=True,
                )
                # Starting from START_EPOCH_EXAMPLE_MINING we rank the indices by
                # the loss and then sample them. This sampling stays constant for
                # EXAMPLE_MINING_FREQ epochs
                indices_by_ranks = np.argsort(np.array(losses))[::-1]
                indices_sampled = hard_example_sampler(
                    indices_by_ranks,
                    len(dataloaders["train"].dataset),
                    selection_pressure=10,
                )
        else:
            indices_by_ranks = np.arange(len(train_dataset_copy))
            np.random.shuffle(indices_by_ranks)

        # accumulate gradients over multiple batches (equivalent to bigger batchsize, but without memory issues)  # noqa
        # Note: depending on the reduction method in the loss function, this might need to be divided by the number  # noqa
        #   of accumulation iterations to be truly equivalent to training with bigger batchsize  # noqa
        accumulated_batches = 0
        for idx in indices_sampled:
            image, label = dataloaders["train"].dataset[idx]
            if config.DEBUG:
                logging.info(
                    f"Image shape: {image.shape}, image max: {image.max()}, min: {image.min()}"
                )
                logging.info("labels: ", torch.unique(label))
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            image = torch.tensor(image).to(config.DEVICE)
            label = torch.tensor(label).to(config.DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(image)
                loss = criterion(outputs, label)

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
                print("Iteration {}: Train Loss: {}".format(nbatches, loss.item()))
                if config.DEBUG:
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
            if (nbatches % (config.ACCUMULATE_BATCHES * 10)) == 0 or nbatches == len(
                dataloaders["train"]
            ) - 1:
                with torch.no_grad():
                    image = image.data.cpu().numpy()
                    prediction = torch.argmax(outputs[0], dim=1).view(*image.shape)
                    prediction = prediction.data.cpu().numpy()
                    uncertainty_map = outputs[1].data.cpu().numpy()
                    label = label.view(*prediction.shape).data.cpu().numpy()
                    # calculate metrics
                    metrics = calculate_metrics(
                        label, prediction, class_names=config.CLASSES
                    )
                    log_iteration_metrics(metrics, cache.train_steps, writer, "train")
                    if config.VISUALIZE_OUTPUT == "all":
                        visualize_uncertainty_training(
                            image,
                            (prediction, uncertainty_map),
                            label,
                            cache.out_dir_train,
                            class_names=config.CLASSES,
                            base_name="out_{}".format(cache.epoch),
                        )
            nbatches += 1

        # change learning rate according to scheduler
        scheduler.step()

        # logging
        train_loss = train_loss / float(accumulated_batches)
        writer.add_scalar("epoch_loss/train_loss", train_loss, epoch)
        cache.last_epoch_results.update({"train_loss": train_loss})

        # VALIDATION
        if config.VISUALIZE_OUTPUT in ["val", "all"]:
            visualize = True
        else:
            visualize = False
        cache, _ = validate(
            dataloaders["val"].dataset, model, criterion, cache, writer, visualize
        )
        val_dice = cache.last_epoch_results["mean_dice"]
        print(
            f"EPOCH {epoch} = Train Loss: {train_loss}, Validation DICE: {val_dice}\n"
        )

    # TODO: Validation on Training to get training DICE

    # Store all epoch results
    results_df = pd.DataFrame(cache.all_epoch_results)
    results_df.to_csv(
        os.path.join(cache.out_dir_epoch_results, "epoch_results.csv"), index=False
    )

    writer.close()
