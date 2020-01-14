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
    device = "cuda:2"
    batchsize = 1   

    run_params = parse_input_arguments(out_dir)
    filter_label = run_params["filter_label"]
    depth, width, image_size, image_depth = run_params["depth"], run_params["width"], run_params["image_size"], run_params["image_depth"]
        
    out_dir_wts = os.path.join(out_dir, "weights")

    # apply validation metrics on training set instead of validation set if train=True
    if test_on_train:
        out_dir_val = os.path.join(out_dir, "train_final")
    else:
        out_dir_val = os.path.join(out_dir, "test")
    if postprocess:
        out_dir_val = out_dir_val + "_postprocessed"

    os.makedirs(out_dir_val, exist_ok=True)

    # root_dir = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/'
    # # meta_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/src/meta/dataset_train.csv'
    # meta_path = "/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/{}.csv".format("_".join(filter_label))
    # label_mapping_path = '/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/label_mapping_train.json'

    # root_dir = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/'
    
    root_dir = '/export/scratch3/bvdp/segmentation/data/MODIR_data_train_2019-12-17/'
    meta_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_2019-12-17.csv'
    label_mapping_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/label_mapping_train_2019-12-17.json'

    # meta_path = "/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/{}.csv".format("_".join(filter_label))
    # label_mapping_path = '/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/label_mapping_train.json'

    transform_val = custom_transforms.Compose([
        # custom_transforms.CropInplane(crop_size=384, crop_mode='center')
        custom_transforms.CustomResize(output_size=image_size),
        # custom_transforms.CropInplane(crop_size=384, crop_mode='center')
        # custom_transforms.CropInplane(crop_size=128, crop_mode='center')
        custom_transforms.CropInplane(crop_size=192, crop_mode='center')
        ])

    val_dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=image_size,
     is_training=test_on_train, transform=transform_val, filter_label=filter_label)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batchsize, num_workers=5)

    model = UNet(depth=depth, width=width, in_channels=1, out_channels=len(val_dataset.classes))
    model.to(device)
    print("model initialized")

    # load weights
    state_dict = torch.load(os.path.join(out_dir_wts, "best_model.pth"), map_location=device)["model"]
    model.load_state_dict(state_dict)
    print("weights loaded")

    # validation
    metrics = np.zeros((4, len(val_dataset.classes)))
    min_depth = 2**depth
    model.eval()
    for nbatches, (image, label) in enumerate(val_dataloader):
        label = label.view(*image.shape).data.cpu().numpy()
        with torch.no_grad():
            nslices = image.shape[2]
            image = image.to(device)

            output = []
            start = 0
            while start+min_depth <= nslices:
                if start + image_depth + min_depth >= nslices:
                    indices = slice(start, nslices)
                    start = nslices
                else:
                    indices = slice(start, start + image_depth)
                    start += image_depth
                
                mini_image = image[:, :, indices, :, :]
                mini_output = model(mini_image)
                output.append(mini_output)

            output = torch.cat(output, dim=2)
            output = torch.argmax(output, dim=1).view(*image.shape)

        image = image.data.cpu().numpy()
        output = output.data.cpu().numpy()
        # print(f'Output shape before pp: {output.shape}')
        if postprocess:
            output = postprocessing.postprocess_segmentation(
                output[0,0], # remove batch and color channel dims
                n_classes=len(val_dataset.classes),
                bg_idx=0)
            # return batch & color channel dims
            output = np.expand_dims(np.expand_dims(output, 0), 0)

        # print(f'Output shape after pp: {output.shape}')
        # print(f'n classes: {val_dataset.classes}')
        im_metrics = calculate_metrics(label, output, classes=val_dataset.classes)
        metrics = metrics + im_metrics

        # probably visualize
        if nbatches%5==0:
            visualize_output(image[0, 0, :, :, :], label[0, 0, :, :, :], output[0, 0, :, :, :],
             out_dir_val, classes=val_dataset.classes, base_name="out_{}".format(nbatches))

        # if nbatches >= 0:
        #   break

    metrics /= nbatches + 1
    results = f"accuracy = {metrics[0]}\nrecall = {metrics[1]}\nprecision = {metrics[2]}\ndice = {metrics[3]}\n"
    print(results)
    return run_params, results


if __name__ == '__main__':
    # experiments = ["./runs/bowel_bag/cross_entropy",
    #               "./runs/bladder/cross_entropy",
    #               "./runs/hip/cross_entropy",
    #               "./runs/rectum/cross_entropy"
    #   ]
    experiments = [
        # "./runs/multiclass_ce_newdata_imagesize_256_width_16/cross_entropy",
        # "./runs/multiclass_ce_newdata_imagesize_256_width_32/cross_entropy",
        # "./runs/multiclass_ce_newdata_imagesize_256_width_64/cross_entropy",
        # "./runs/multiclass_ce_newdata_imagesize_512_width_24/cross_entropy"
        # "./runs/downsample_256_img_depth_16_unet_width_64/cross_entropy",
        "./runs/downsample_256_img_depth_48_unet_width_64/cross_entropy",
        # "./runs/downsample_171_img_depth_48_unet_width_64/cross_entropy",
        # "./runs/downsample_128_no_crop/cross_entropy"
        # "./runs/downsample_171_img_depth_48_unet_width_64_unet_depth_5/cross_entropy"
        # "./runs/multiclass_ce_no_elastic_newdata/cross_entropy",
        # "./runs/multiclass_ce_loss_no_elastic/cross_entropy",
        # "./runs/multiclass_ce_loss_weighted_no_elastic/weighted_cross_entropy_0.049_0.13_0.21_0.29_0.32",
        # "./runs/multiclass_no_elastic_inverse_class_weights/focal_loss_gamma_2.0_alpha_0.0003_0.013_0.09_0.36_0.53"
    ]

    test_on_train = False
    postprocess = True

    log_name = "test_log_downsample_171"
    if test_on_train:
        log_name = log_name + '_on_train'
    if postprocess:
        log_name = log_name + '_with_postprocessing'
    f = open(f"logs/{log_name}.txt", "w")
    for out_dir in experiments:
        run_params, results = main(out_dir, test_on_train=test_on_train, postprocess=postprocess)
        run_params = ["{} : {}".format(key, val) for key, val in run_params.items()]
        run_params = ", ".join(run_params)
        f.write(f"\n{run_params}\n")
        f.write(results)
        f.write("\n")

    f.close()


