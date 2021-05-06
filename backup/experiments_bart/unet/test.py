import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from torch.utils.data import Dataset, DataLoader

import os, argparse
import numpy as np
import cv2
import json
from skimage.io import imread, imsave
from sklearn.metrics import confusion_matrix
from model_unet import UNet
from torch_AMCDataset import AMCDataset
import sys
from scipy import signal

sys.path.append("..")
from utils import custom_transforms, custom_losses, postprocessing



def parse_input_arguments(out_dir):
    run_params = json.load(open(os.path.join(out_dir, "run_parameters.json"), "r"))
    return run_params


def calculate_metrics(label, output, classes=None):
    """
    Inputs:
    output (numpy tensor): prediction
    label (numpy tensor): ground truth
    classes (list): class names    

    """
    if classes is not None:
        classes = list(range(len(classes)))

    accuracy = round(np.sum(output == label) / float(output.size), 2)
    accuracy = accuracy * np.ones(len(classes))
    epsilon = 1e-6
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

    print("accuracy = {}, recall = {}, precision = {}, dice = {}".format(accuracy, recall, precision, dice))
    return metrics     



def visualize_output(image, label, output, out_dir, classes=None, base_name="im"):
    """
    Inputs:
    image (3D numpy array, slices along first axis): input image
    label (3D numpy array, slices along first axis, integer values corresponding to class): ground truth
    output (3D numpy array, slices along first axis, integer values corresponding to class): prediction
    out_dir (string): output directory
    classes (list): class names

    """

    alpha = 0.6
    colors = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1),
                4: (1, 1, 0), 5: (0, 1, 1), 6: (1, 0, 1),
                7: (1, 0.5, 0), 8: (0, 1, 0.5), 9: (0.5, 0, 1),
                10: (0.5, 1, 0), 11: (0, 0.5, 1), 12: (1, 0, 0.5)}

    nslices, shp, _ = image.shape
    imlist = list()
    count = 0
    for slice_no in range(nslices):
        im = cv2.cvtColor(image[slice_no], cv2.COLOR_GRAY2RGB)
        lbl = np.zeros_like(im)
        pred = np.zeros_like(im)
        mask_lbl = (label[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        mask_pred = (output[slice_no] != 0).astype(np.float32).reshape(shp, shp, 1)
        for class_no in range(1, len(classes)):
            lbl[label[slice_no]==class_no] = colors[class_no]
            pred[output[slice_no]==class_no] = colors[class_no]

        im_lbl = (1 - alpha*mask_lbl) * im + alpha*lbl
        im_pred = (1 - alpha*mask_pred) * im + alpha*pred
        im = np.concatenate((im, im_lbl, im_pred), axis=1)
        imlist.append(im)

        if len(imlist) == 4:
            im = np.concatenate(imlist, axis=0)
            imsave(os.path.join(out_dir, "{}_{}.jpg".format(base_name, count)), (im*255).astype(np.uint8))
            imlist = list()
            count += 1
    return None


def main(out_dir, test_on_train=False, postprocess=False):
    """
    Sliding window validation of a model (stored in out_dir) on train or validation set. This stores the results in a log file, and 
    visualizations of predictions in the out_dir directory
    """
    device = "cuda:0"
    batchsize = 1   

    run_params = parse_input_arguments(out_dir)
    # filter_label = run_params["filter_label"]
    # depth, width, image_size, image_depth = run_params["depth"], run_params["width"], run_params["image_size"], run_params["image_depth"]
    depth, width, image_depth = run_params["depth"], run_params["width"], run_params["image_depth"]
        
    out_dir_wts = os.path.join(out_dir, "weights")

    # apply validation metrics on training set instead of validation set if train=True
    if test_on_train:
        out_dir_val = os.path.join(out_dir, "train_final")
    else:
        out_dir_val = os.path.join(out_dir, "test")
    if postprocess:
        out_dir_val = out_dir_val + "_postprocessed"

    os.makedirs(out_dir_val, exist_ok=True)

    root_dir = '/export/scratch2/bvdp/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split_preprocessed_21-08-2020/'    
    meta_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_21-08-2020_slice_annot.csv'

    val_dataset = AMCDataset(root_dir, meta_path, is_training=test_on_train, log_path=None)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batchsize, num_workers=5)

    model = UNet(depth=depth, width=width, in_channels=1, out_channels=len(val_dataset.classes))

    model.to(device)
    print("model initialized")

    # load weights
    state_dict = torch.load(os.path.join(out_dir_wts, "best_model.pth"), map_location=device)["model"]
    model.load_state_dict(state_dict)
    print("weights loaded")

    slice_weighting = True
    classes = val_dataset.classes
    # validation
    metrics = np.zeros((4, len(val_dataset.classes)))
    min_depth = 2**depth
    model.eval()
    for nbatches, (image, label) in enumerate(val_dataloader):
        label = label.view(*image.shape).data.cpu().numpy()
        with torch.no_grad():
            nslices = image.shape[2]
            image = image.to(device)

            output = torch.zeros(batchsize, len(classes), *image.shape[2:])
            slice_overlaps = torch.zeros(1,1,nslices,1,1)
            start = 0
            while start+min_depth <= nslices:
                if start + image_depth >= nslices:
                    indices = slice(nslices-image_depth, nslices)
                    start = nslices
                else:
                    indices = slice(start, start + image_depth)
                    start += image_depth // 3
                
                mini_image = image[:, :, indices, :, :]
                mini_output = model(mini_image)
                if slice_weighting:
                    actual_slices = mini_image.shape[2]
                    weights = signal.gaussian(actual_slices, std=actual_slices/6)
                    weights = torch.tensor(weights, dtype=torch.float)

                    output[:,:, indices, :,:] += mini_output.to('cpu')*weights.view(1,1,actual_slices,1,1) 
                    slice_overlaps[0,0,indices,0,0] += weights
                else:
                    output[:,:, indices, :,:] += mini_output.to('cpu')
                    slice_overlaps[0,0,indices,0,0] +=  1 
                    
            output = output / slice_overlaps
            output = torch.argmax(output, dim=1).view(*image.shape)  

        image = image.data.cpu().numpy()
        output = output.data.cpu().numpy()
        # print(f'Output shape before pp: {output.shape}')
        if postprocess:
            multiple_organ_indici = [idx for idx, class_name in enumerate(val_dataset.classes) if class_name == 'hip']
            output = postprocessing.postprocess_segmentation(
                output[0,0], # remove batch and color channel dims
                n_classes=len(val_dataset.classes),
                multiple_organ_indici=multiple_organ_indici,
                bg_idx=0)
            # return batch & color channel dims
            output = np.expand_dims(np.expand_dims(output, 0), 0)

        # print(f'Output shape after pp: {output.shape}')
        # print(f'n classes: {val_dataset.classes}')
        im_metrics = calculate_metrics(label, output, classes=val_dataset.classes)
        metrics = metrics + im_metrics

        # probably visualize
        # if nbatches%5==0:
        visualize_output(image[0, 0, :, :, :], label[0, 0, :, :, :], output[0, 0, :, :, :],
             out_dir_val, classes=val_dataset.classes, base_name="out_{}".format(nbatches))

        # if nbatches >= 0:
        #   break

    metrics /= nbatches + 1
    results = f"accuracy = {metrics[0]}\nrecall = {metrics[1]}\nprecision = {metrics[2]}\ndice = {metrics[3]}\n"
    print(results)
    return run_params, results


if __name__ == '__main__':
    # list of experiment directories to perform the validation on
    experiments = [
        # "./runs/experiment_7/focal_loss_gamma_2",
        # "./runs/experiment_26/weighted_cross_entropy_0.52_0.85_1.09_1.28_1.26",
        # "./runs/experiment_1/cross_entropy",
        # "./runs_augmentation/experiment_7/soft_dice",
        # "./runs_augmentation/experiment_10/soft_dice",
        # "./runs_augmentation/experiment_11/soft_dice",
        "./runs_augmentation_v2/experiment_0/soft_dice",
        "./runs_augmentation_v2/experiment_1/soft_dice",
        "./runs_augmentation_v2/experiment_2/soft_dice",
    ]

    # test on trianing set to inspect overfitting behavior
    test_on_train = False
    # postprocess predictions to take only the largest connected region of predictions (or 2 regiosn in case of hip) and discard 
    # noisy predictions of an organ that are completely seperate from the main region
    postprocess = False

    log_name = "best_augmentation_models_v2"
    if test_on_train:
        log_name = log_name + '_on_train'
    if postprocess:
        log_name = log_name + '_with_postprocessing'
    with open(f"logs/{log_name}.txt", "w") as f: 
        for out_dir in experiments:
            run_params, results = main(out_dir, test_on_train=test_on_train, postprocess=postprocess)
            run_params = ["{} : {}".format(key, val) for key, val in run_params.items()]
            run_params = ", ".join(run_params)
            f.write(f"\n{run_params}\n")
            f.write(results)
            f.write("\n")


