import os

import cv2
import numpy as np
from skimage.io import imsave


def visualize_output(image, label, output, out_dir, class_names=None, base_name="im"):
    """
    Inputs:
    image (3D numpy array, slices along first axis): input image
    label (3D numpy array, slices along first axis, integer values corresponding to class): ground truth
    output (3D numpy array, slices along first axis, integer values corresponding to class): prediction
    out_dir (string): output directory
    classes (list): class names

    """

    alpha = 0.6
    colors = {
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

    nslices, shp, _ = image.shape
    imlist = list()
    count = 0
    for slice_no in range(nslices):
        im = cv2.cvtColor(image[slice_no], cv2.COLOR_GRAY2RGB)
        lbl = np.zeros_like(im)
        pred = np.zeros_like(im)
        mask_lbl = (label[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        mask_pred = (output[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        for class_no in range(1, len(class_names)):
            lbl[label[slice_no] == class_no] = colors[class_no]
            pred[output[slice_no] == class_no] = colors[class_no]

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
