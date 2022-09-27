import json
import logging
import os
from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd
import torch
from experiments.config import config
from experiments.procedures.basic.validation import validate
from experiments.utils.cache import RuntimeCache
from experiments.utils.metrics import calculate_metrics
from experiments.utils.utilities import log_iteration_metrics
from experiments.utils.visualize import visualize_output
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


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

    # Load weights if needed
    if config.LOAD_WEIGHTS:
        weights = torch.load(
            os.path.join(cache.out_dir_weights, "best_model.pth"),
            map_location=config.DEVICE,
        )["model"]
        model.load_state_dict(weights)

    logging.info(f"Running for {config.NEPOCHS} epochs")
    for epoch in range(0, config.NEPOCHS):
        logging.info(f"Epoch: {epoch}")
        cache.epoch += 1
        cache.last_epoch_results = {"epoch": epoch}
        # Training step
        model.train()
        train_loss = 0.0
        nbatches = 0
        # accumulate gradients over multiple batches (equivalent to bigger batchsize, but without memory issues)  # noqa
        # Note: depending on the reduction method in the loss function, this might need to be divided by the number  # noqa
        #   of accumulation iterations to be truly equivalent to training with bigger batchsize  # noqa
        accumulated_batches = 0
        for nbatches, (image, label) in enumerate(dataloaders["train"]):
            logging.debug(f"Epoch: {epoch}, Iteration: {nbatches}")
            logging.debug(
                f"Image shape: {image.shape}, image max: {image.max()}, min: {image.min()}"
            )
            logging.debug(f"labels: {torch.unique(label)}")
            image = image.to(config.DEVICE)
            label = label.to(config.DEVICE)

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
                logging.debug(f"Iteration {nbatches}: Train Loss: {loss.item()}")
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
                    prediction = torch.argmax(outputs, dim=1).view(*image.shape)
                    prediction = prediction.data.cpu().numpy()
                    label = label.view(*prediction.shape).data.cpu().numpy()
                    # calculate metrics
                    metrics = calculate_metrics(
                        label, prediction, class_names=config.CLASSES
                    )
                    log_iteration_metrics(metrics, cache.train_steps, writer, "train")
                    if config.VISUALIZE_OUTPUT == "all":
                        visualize_output(
                            image,
                            prediction,
                            label,
                            cache.out_dir_train,
                            class_names=config.CLASSES,
                            base_name="out_{}".format(cache.epoch),
                        )

            del image, label, outputs
            torch.cuda.empty_cache()

        # change learning rate according to scheduler
        scheduler.step()

        # logging
        train_loss = train_loss / float(accumulated_batches)
        writer.add_scalar("epoch_loss/train_loss", train_loss, epoch)
        cache.last_epoch_results.update({"train_loss": train_loss})

        # VALIDATION
        # TODO: Take model saving out of validation code
        cache = validate(dataloaders["val"], model, cache, writer)

        val_dice = cache.last_epoch_results.get("mean_dice")
        if val_dice:
            logging.debug(
                f"EPOCH {epoch} = Train Loss: {train_loss}, Validation DICE: {val_dice}\n"
            )

    # TODO: Validation on Training to get training DICE

    logging.info("")
    # Store all epoch results
    results_df = pd.DataFrame(cache.all_epoch_results)
    results_df.to_csv(
        os.path.join(cache.out_dir_epoch_results, "epoch_results.csv"), index=False
    )

    writer.close()
