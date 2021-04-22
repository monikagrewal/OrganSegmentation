import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from config import config
from data.data import get_dataloaders, get_datasets
from model.loss.custom import FocalLoss, SoftDiceLoss
from model.transform.augmentation import get_transform_pipelines
from model.unet import UNet
from model.utils.cache import RuntimeCache
from model.utils.metrics import calculate_metrics
from model.utils.visualize import visualize_output
from model.validate import proper_validate, validate


def get_criterion() -> nn.Module:
    if config.LOSS_FUNCTION == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    elif config.LOSS_FUNCTION == "weighted_cross_entropy":
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(config.CLASS_WEIGHTS, device=config.DEVICE)
        )

    elif config.LOSS_FUNCTION == "focal_loss":
        if config.ALPHA is not None:
            alpha = torch.tensor(config.ALPHA, device=config.DEVICE)
        criterion = FocalLoss(gamma=config.GAMMA, alpha=alpha)

    elif config.LOSS_FUNCTION == "soft_dice":
        criterion = SoftDiceLoss(drop_background=False)

    elif config.LOSS_FUNCTION == "weighted_soft_dice":
        criterion = SoftDiceLoss(
            weight=torch.tensor(config.CLASS_WEIGHTS, device=config.DEVICE)
        )

    else:
        raise ValueError(f"unknown loss function: {config.LOSS_FUNCTION}")

    return criterion


def train() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    transform_pipelines = get_transform_pipelines()
    train_dataloader, val_dataloader, proper_val_dataloader = get_dataloaders(
        transform_pipelines
    )

    # Intermediate results storage to pass to other functions to reduce parameters
    writer = SummaryWriter(config.OUT_DIR)
    cache = RuntimeCache()

    # Tensorboard
    criterion = get_criterion()
    model = UNet(
        depth=config.MODEL_DEPTH,
        width=config.MODEL_WIDTH,
        in_channels=1,
        out_channels=len(config.CLASSES),
    )
    model.to(config.DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        eps=0.001,
    )

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
        for nbatches, (image, label) in enumerate(train_dataloader):
            print("Image shape: ", image.shape)
            image = image.to(config.DEVICE)
            label = label.to(config.DEVICE)
            output = model(image)
            loss = criterion(output, label)

            # to make sure accumulated loss equals average loss in batch
            # and won't depend on accumulation batch size
            loss = loss / config.ACCUMULATE_BATCHES
            loss.backward()

            if ((nbatches + 1) % config.ACCUMULATE_BATCHES) == 0:
                accumulated_batches += 1
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                print("Iteration {}: Train Loss: {}".format(nbatches, loss.item()))
                writer.add_scalar("Loss/train_loss", loss.item(), cache.train_steps)
                cache.train_steps += 1

                # if not a full batch left, break out of epoch to prevent wasted
                # computation on sample that won't add up to a full batch and
                # therefore won't result in a step
                if len(train_dataloader) - (nbatches + 1) < config.ACCUMULATE_BATCHES:
                    break

            # Train results
            if (nbatches % config.ACCUMULATE_BATCHES * 3) == 0 or nbatches == len(
                train_dataloader
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
        val_loss = validate(val_dataloader, model, criterion, cache)
        print(
            f"EPOCH {epoch} = Train Loss: {train_loss}, Validation Loss: {val_loss}\n"
        )

        # if new best model based on val_loss, save the weights
        if val_loss <= cache.best_loss:
            cache.best_loss = val_loss
            weights = {
                "model": model.state_dict(),
                "epoch": epoch,
                "loss": val_loss,
            }
            torch.save(
                weights, os.path.join(config.OUT_DIR_WEIGHTS, "best_model_loss.pth")
            )

        # Finalize epoch
        writer.add_scalar("epoch_loss/train_loss", train_loss, epoch)
        writer.add_scalar("epoch_loss/val_loss", val_loss, epoch)

        # PROPER VALIDATION
        if (
            config.PROPER_EVAL_EVERY_EPOCHS is not None
            and (epoch + 1) % config.PROPER_EVAL_EVERY_EPOCHS == 0
        ):
            proper_validate(proper_val_dataloader, model, cache, writer)

        # Early stopping
        cache.all_epoch_results.append(cache.last_epoch_results)
        if (
            config.EARLY_STOPPING_PATIENCE is not None
            and cache.epochs_no_improvement >= config.EARLY_STOPPING_PATIENCE
        ):
            print(
                f"{cache.epochs_no_improvement} epochs without improvement. Stopping!"
            )
            break

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
