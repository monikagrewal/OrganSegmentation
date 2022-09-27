import os

import cv2
import numpy as np
import torch
from skimage.io import imsave

COLORS = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
    3: (0, 0, 1),
    4: (1, 1, 0),
    5: (0, 1, 1),
    6: (1, 0, 1),
    7: (1, 0.5, 0),
    8: (0, 1, 0.5),
    9: (0.5, 0, 1),
    10: (0.5, 1, 0),
    11: (0, 0.5, 1),
    12: (1, 0, 0.5),
}

def visualize_output(image, prediction, label, out_dir, class_names=None, base_name="im"):
    """
    Inputs:
    image (5D numpy array, N x C x D x H x W): input image
    prediction (5D numpy array, N x C x D x H x W, integer values corresponding to class): prediction
    label (1D numpy array, integer values corresponding to class): ground truth
    out_dir (string): output directory
    classes (list): class names

    """
    image = image[0, 0, :, :, :]
    label = label[0, 0, :, :, :]
    prediction = prediction[0, 0, :, :, :]

    alpha = 0.6

    nslices, shp, _ = image.shape
    imlist = list()
    count = 0
    for slice_no in range(nslices):
        im = cv2.cvtColor(image[slice_no], cv2.COLOR_GRAY2RGB)
        lbl = np.zeros_like(im)
        pred = np.zeros_like(im)
        mask_lbl = (label[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        mask_pred = (prediction[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        for class_no in range(1, len(class_names)):
            lbl[label[slice_no] == class_no] = COLORS[class_no]
            pred[prediction[slice_no] == class_no] = COLORS[class_no]

        im_lbl = (1 - alpha * mask_lbl) * im + alpha * lbl
        im_pred = (1 - alpha * mask_pred) * im + alpha * pred
        im = np.concatenate((im, im_lbl, im_pred), axis=1)
        imlist.append(im)

        if len(imlist) == 4:
            im = np.concatenate(imlist, axis=0)
            imsave(
                os.path.join(out_dir, "{}_{}.jpg".format(base_name, count)),
                (im * 255).astype(np.uint8),
            )
            imlist = list()
            count += 1
    return None


def visualize_uncertainty_training(image, outputs, label, out_dir, class_names=None, base_name="im"):
    """
    Inputs:
    image (5D numpy array, N x C x D x H x W): input image
    outputs (1. 5D numpy array, N x C x D x H x W, integer values corresponding to class): prediction
            (2. 5D numpy array, N x 1 x D x H x W): uncertainty map
    label (5D numpy array, integer values corresponding to class): ground truth
    out_dir (string): output directory
    classes (list): class names

    """
    image = image[0, 0, :, :, :]
    label = label[0, 0, :, :, :]
    prediction, uncertainty_map = outputs
    lb, ub = uncertainty_map.min(), uncertainty_map.max()
    logging.debug(f"log sigma max: {ub}, min: {lb}")
    # convert log sigma to uncertainty
    uncertainty_map = np.exp(uncertainty_map).astype(np.float32)
    lb, ub = uncertainty_map.min(), uncertainty_map.max()
    logging.debug(f"uncertainty max: {ub}, min: {lb}")
    uncertainty_map = (uncertainty_map - lb + 1e-6) / (ub - lb + 1e-6)

    prediction = prediction[0, 0, :, :, :]
    uncertainty_map = uncertainty_map[0, 0, :, :, :]

    alpha = 0.6

    nslices, shp, _ = image.shape
    imlist = list()
    count = 0
    for slice_no in range(nslices):
        im = cv2.cvtColor(image[slice_no], cv2.COLOR_GRAY2RGB)
        uncertainty = cv2.cvtColor(uncertainty_map[slice_no], cv2.COLOR_GRAY2RGB)
        lbl = np.zeros_like(im)
        pred = np.zeros_like(im)
        mask_lbl = (label[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        mask_pred = (prediction[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        for class_no in range(1, len(class_names)):
            lbl[label[slice_no] == class_no] = COLORS[class_no]
            pred[prediction[slice_no] == class_no] = COLORS[class_no]

        im_lbl = (1 - alpha * mask_lbl) * im + alpha * lbl
        im_pred = (1 - alpha * mask_pred) * im + alpha * pred
        im_uncertainty = (1 - alpha) * im + alpha * uncertainty
        im = np.concatenate((im, im_lbl, im_pred, uncertainty), axis=1)
        imlist.append(im)

        if len(imlist) == 4:
            im = np.concatenate(imlist, axis=0)
            imsave(
                os.path.join(out_dir, "{}_{}.jpg".format(base_name, count)),
                (im * 255).astype(np.uint8),
            )
            imlist = list()
            count += 1
    return None


def visualize_uncertainty_validation(image, outputs, label, out_dir, class_names=None, base_name="im"):
    """
    Inputs:
    image (5D numpy array, N x C x D x H x W): input image
    outputs (1. 5D numpy array, N x C x D x H x W, integer values corresponding to class): prediction
            (2. 5D numpy array, N x 1 x D x H x W): data uncertainty map
            (3. 5D numpy array, N x 1 x D x H x W): model uncertainty map
    label (5D numpy array, integer values corresponding to class): ground truth
    out_dir (string): output directory
    classes (list): class names

    """
    def normalize(x):
        lb, ub = x.min(), x.max()
        logging.debug(f"uncertainty max: {ub}, min: {lb}")
        x = (x - lb + 1e-6) / (ub - lb + 1e-6)
        return x

    image = image[0, 0, :, :, :]
    label = label[0, 0, :, :, :]
    prediction, data_uncertainty, model_uncertainty = outputs

    # data_uncertainty = normalize(data_uncertainty)
    lb, ub = data_uncertainty.min(), data_uncertainty.max()
    logging.debug(f"data uncertainty max: {ub}, min: {lb}")
    # model_uncertainty = normalize(model_uncertainty)
    lb, ub = model_uncertainty.min(), model_uncertainty.max()
    logging.debug(f"model uncertainty max: {ub}, min: {lb}")

    data_uncertainty = data_uncertainty[0, 0, :, :, :]
    model_uncertainty = model_uncertainty[0, 0, :, :, :]
    prediction = prediction[0, 0, :, :, :]

    alpha = 0.6

    nslices, shp, _ = image.shape
    imlist = list()
    count = 0
    for slice_no in range(nslices):
        im = cv2.cvtColor(image[slice_no], cv2.COLOR_GRAY2RGB)
        d_uncertainty = cv2.cvtColor(data_uncertainty[slice_no], cv2.COLOR_GRAY2RGB)
        m_uncertainty = cv2.cvtColor(model_uncertainty[slice_no], cv2.COLOR_GRAY2RGB)
        lbl = np.zeros_like(im)
        pred = np.zeros_like(im)
        mask_lbl = (label[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        mask_pred = (prediction[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        for class_no in range(1, len(class_names)):
            lbl[label[slice_no] == class_no] = COLORS[class_no]
            pred[prediction[slice_no] == class_no] = COLORS[class_no]

        im_lbl = (1 - alpha * mask_lbl) * im + alpha * lbl
        im_pred = (1 - alpha * mask_pred) * im + alpha * pred
        im = np.concatenate((im, im_lbl, im_pred, d_uncertainty, m_uncertainty,\
                             d_uncertainty + m_uncertainty), axis=1)
        imlist.append(im)

        if len(imlist) == 4:
            im = np.concatenate(imlist, axis=0)
            imsave(
                os.path.join(out_dir, "{}_{}.jpg".format(base_name, count)),
                (im * 255).astype(np.uint8),
            )
            imlist = list()
            count += 1
    return None
