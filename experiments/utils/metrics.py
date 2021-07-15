import logging

import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_metrics(label, output, class_names=None):
    """
    Inputs:
    output (numpy tensor): prediction
    label (numpy tensor): ground truth
    classes (list): class names

    """
    if class_names is not None:
        class_names = list(range(len(class_names)))

    accuracy = round(np.sum(output == label) / float(output.size), 2)
    accuracy = accuracy * np.ones(len(class_names))
    epsilon = 1e-6
    cm = confusion_matrix(label.reshape(-1), output.reshape(-1), labels=class_names)
    total_true = np.sum(cm, axis=1).astype(np.float32)
    total_pred = np.sum(cm, axis=0).astype(np.float32)
    tp = np.diag(cm)
    recall = tp / (epsilon + total_true)
    precision = tp / (epsilon + total_pred)
    dice = (2 * tp) / (epsilon + total_true + total_pred)

    recall = [round(item, 2) for item in recall]
    precision = [round(item, 2) for item in precision]
    dice = [round(item, 2) for item in dice]

    logging.info(
        "accuracy = {}, recall = {}, precision = {}, dice = {}".format(
            accuracy, recall, precision, dice
        )
    )
    return accuracy, recall, precision, dice
