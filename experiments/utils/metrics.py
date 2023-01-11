import logging

import numpy as np
from sklearn.metrics import confusion_matrix
from surface_distance.metrics import (
    compute_robust_hausdorff,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)


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

    haussdorf_distance = []
    surface_dice = []
    for c in class_names:
        output_c = (output[0, 0] == c).astype(bool)
        label_c = (label[0, 0] == c).astype(bool)

        surface_distances_raw = compute_surface_distances(
            output_c, label_c, (2.5, 2.5, 2.5)
        )
        haussdorf_distance_class = compute_robust_hausdorff(surface_distances_raw, 95)
        surface_dice_class = compute_surface_dice_at_tolerance(
            surface_distances_raw, 2.5
        )

        haussdorf_distance.append(round(haussdorf_distance_class, 3))
        surface_dice.append(round(surface_dice_class, 3))

    return accuracy, recall, precision, dice, haussdorf_distance, surface_dice
