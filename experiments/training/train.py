import json
import os

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
from typing import Dict
from config import config
from data.load import get_dataloaders
from models.unet import UNet
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from training.validate import validate
from utils.augmentation import get_augmentation_pipelines
from utils.cache import RuntimeCache
from utils.loss import get_criterion
from utils.metrics import calculate_metrics
from utils.visualize import visualize_output


def setup_train():
    # Print & Store config
    print(config.dict())
    with open(os.path.join(config.OUT_DIR, "run_parameters.json"), "w") as file:
        json.dump(config.dict(), file, indent=4)

    # Load datasets
    augmentation_pipelines = get_augmentation_pipelines()
    dataloaders = get_dataloaders(config.CLASSES, augmentation_pipelines)

    # Intermediate results storage to pass to other functions to reduce parameters
    writer = SummaryWriter(config.OUT_DIR)
    cache = RuntimeCache()

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


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
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
            os.path.join(config.OUT_DIR_WEIGHTS, "best_model.pth"),
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
            print("Image shape: ", image.shape)
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
                print("Iteration {}: Train Loss: {}".format(nbatches, loss.item()))
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
            torch.save(weights, os.path.join(config.OUT_DIR_WEIGHTS, "best_model.pth"))
        else:
            cache.epochs_no_improvement += 1

        # Store model at end of epoch to get final model (also on failure)
        weights = {
            "model": model.state_dict(),
            "epoch": cache.epoch,
            "mean_dice": val_dice,
        }
        torch.save(weights, os.path.join(config.OUT_DIR_WEIGHTS, "final_model.pth"))

        print(
            f"EPOCH {epoch} = Train Loss: {train_loss}, Validation DICE: {val_dice}\n"
        )
        writer.add_scalar("epoch_loss/train_loss", train_loss, epoch)

    # TODO: Validation on Training to get training DICE

    # Store all epoch results
    results_df = pd.DataFrame(cache.all_epoch_results)
    results_df.to_csv(
        os.path.join(config.OUT_DIR_EPOCH_RESULTS, "epoch_results.csv"), index=False
    )

    writer.close()


def write_train_results(
    image, label, prediction, cache: RuntimeCache, writer: SummaryWriter
) -> None:

    # calculate metrics and probably visualize prediction
    accuracy, recall, precision, dice = calculate_metrics(
        label, prediction, class_names=config.CLASSES
    )

    visualize_output(
        image[0, 0, :, :, :],
        label[0, 0, :, :, :],
        prediction[0, 0, :, :, :],
        config.OUT_DIR_TRAIN,
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
