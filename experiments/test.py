import json
import logging
import os

import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter

from config import Config
from datasets.amc import *
from models import unet, unet_khead_student
from utils.augmentation import *
from utils.loss import *


def calculate_metrics(label, output, classes=None):
    """."""
    epsilon = 1e-6

    if classes is not None:
        classes = list(range(len(classes)))

    accuracy = round(np.sum(output == label) / float(output.size), 2)
    accuracy = accuracy * np.ones(len(classes))

    cm = confusion_matrix(label.reshape(-1), output.reshape(-1), labels=classes)
    total_true = np.sum(cm, axis=1).astype(np.float32)
    total_pred = np.sum(cm, axis=0).astype(np.float32)
    tp = np.diag(cm)

    recall = tp / (epsilon + total_true)
    precision = tp / (epsilon + total_pred)
    dice = (2 * tp) / (epsilon + total_true + total_pred)

    recall = np.round(recall, 2)
    precision = np.round(precision, 2)
    dice = np.round(dice, 2)

    metrics = np.asarray([accuracy, recall, precision, dice])

    logging.info("accuracy = {}, recall = {}, precision = {}, dice = {}".format(accuracy, recall, precision, dice))
    return metrics


def setup_test(out_dir):
    # Reinitialize config
    config = Config.parse_file(os.path.join(out_dir, "run_parameters.json"))

    test_dataset = AMCDataset(
        config.DATA_DIR,
        config.META_PATH,
        classes=config.CLASSES,
        slice_annot_csv_path=config.SLICE_ANNOT_CSV_PATH,
        log_path=None,
    )

    dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=5
    )

    model = unet.UNet(
        depth=config.MODEL_PARAMS["depth"],
        width=config.MODEL_PARAMS["width"],
        in_channels=1,
        out_channels=len(config.CLASSES),
    )

    model.to(config.DEVICE)
    logging.info("Model initialized for testing")

    state_dict = torch.load(
        os.path.join(config.OUT_DIR_WEIGHTS, "best_model.pth"),
        map_location=config.DEVICE,
    )["model"]

    model.load_state_dict(state_dict)

    model.eval()
    logging.info("Weights loaded")

    metrics = np.zeros((4, len(config.CLASSES)))
    for nbatches, (image, label) in enumerate(dataloader):
        image = image.to(config.DEVICE)
        label = label.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            output = model(image)
            metrics_batch = calculate_metrics(label=label, output=output, classes=config.CLASSES)

        metrics += metrics_batch

    metrics /= nbatches + 1


    accuracy, recall, precision, dice = metrics
    logging.info(
        f"Test results:\n"
        f"accuracy = {accuracy}\nrecall = {recall}\n"
        f"precision = {precision}\ndice = {dice}\n"
    )

    return metrics
