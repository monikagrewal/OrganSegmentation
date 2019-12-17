import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp

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
from utils import custom_transforms, custom_losses

"""
all labels:
['anal_canal', 'bowel_bag', 'bladder', 'hip', 'rectum', 'sigmoid', 'spinal_cord']
"""

parser = argparse.ArgumentParser(description='Train UNet')
parser.add_argument("-filter_label", help="list of labels to filter",
                        nargs='+', default=['bowel_bag', 'bladder', 'hip', 'rectum'])
parser.add_argument("-out_dir", help="output directory", type=str, default="./runs/tmp")    
parser.add_argument("-device", help="GPU number", type=int, default=0)
parser.add_argument("-load_weights", help="load weights", type=str, default='False')
parser.add_argument("-depth", help="network depth", type=int, default=4)
parser.add_argument("-width", help="network width", type=int, default=16)
parser.add_argument("-image_size", help="image size", type=int, default=512)
parser.add_argument("-crop_sizes", help="crop sizes", nargs='+', type=int, default=[16,256,256])
parser.add_argument("-image_depth", help="image depth", type=int, default=48)
parser.add_argument("-nepochs", help="number of epochs", type=int, default=50)
parser.add_argument("-lr", help="learning rate", type=float, default=0.01)
parser.add_argument("-batchsize", help="batchsize", type=int, default=1)
parser.add_argument("-accumulate_batches", help="batchsize", type=int, default=16)
parser.add_argument("-loss_function", help="loss function", default='cross_entropy')
parser.add_argument("-class_weights", nargs='+', type=float, help="class weights", default=None)
parser.add_argument("-class_sample_freqs", nargs='+', type=float, help="sample freq weight per class", default=[3,1,1,1,1])
parser.add_argument("-gamma", help="loss function", type=float, default='1') 
parser.add_argument("-alpha", help="loss function", type=float, nargs='+', default=None)

def parse_input_arguments():
    run_params = parser.parse_args()
    run_params = vars(run_params)
    run_params["load_weights"] = True if run_params["load_weights"] in ['True', 'true', '1'] else False
    # out_dir_base = run_params["out_dir"]


    return run_params



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


def main():
    run_params = parse_input_arguments()
    print(run_params)
    
    filter_label = run_params["filter_label"]
    out_dir, nepochs, lr, batchsize = run_params["out_dir"], run_params["nepochs"], run_params["lr"], run_params["batchsize"]
    depth, width, image_size, crop_sizes, image_depth = run_params["depth"], run_params["width"], run_params["image_size"], run_params["crop_sizes"], run_params["image_depth"]
    accumulate_batches = run_params["accumulate_batches"]
    class_sample_freqs = run_params["class_sample_freqs"]

    os.makedirs(run_params["out_dir"], exist_ok=True)

    # root_dir = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/'
    root_dir = 'modir_newdata_dicom'
    # meta_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/src/meta/dataset_train.csv'
    # meta_path = "/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/{}.csv".format("_".join(filter_label))
    meta_path = "/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_2019-10-22_fixed.csv"
    # label_mapping_path = '/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/label_mapping_train.json'
    label_mapping_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/label_mapping_train_2019-10-22.json'

    # transform_train = custom_transforms.Compose([
    #     custom_transforms.CropDepthwise(crop_size=image_depth, crop_mode='random'),
    #     custom_transforms.CustomResize(output_size=image_size),
    #     custom_transforms.CropInplane(crop_size=192, crop_mode='center'),
    #     custom_transforms.RandomBrightness(),
    #     custom_transforms.RandomContrast(),
    #     # custom_transforms.RandomElasticTransform3D_2(p=0.7),
    #     custom_transforms.RandomRotate3D(p=0.3)      
    # ])

    transform_train = custom_transforms.Compose([
        custom_transforms.CustomResize(output_size=image_size),
        # custom_transforms.CropLabel(p=1.0, crop_sizes=crop_sizes, class_weights=class_sample_freqs, 
        #          rand_transl_range=(5,25,25), bg_class_idx=0),
        custom_transforms.CropLabel(p=1.0, crop_sizes=crop_sizes, class_weights=class_sample_freqs, 
                 rand_transl_range=(5,25,25), bg_class_idx=0),        
        # custom_transforms.RandomBrightness(),
        # custom_transforms.RandomContrast(),
        # custom_transforms.RandomRotate3D(p=0.3)      
    ])    

    # transform_val = custom_transforms.Compose([
    #     custom_transforms.CropDepthwise(crop_size=image_depth, crop_mode='random'),
    #     custom_transforms.CustomResize(output_size=image_size),
    #     # custom_transforms.CropInplane(crop_size=384, crop_mode='center')
    #     custom_transforms.CropInplane(crop_size=192, crop_mode='center')
    #     ])

    # transform_val = custom_transforms.Compose([
    #     custom_transforms.CustomResize(output_size=image_size),
    #     custom_transforms.CropLabel(p=1.0, crop_sizes=crop_sizes, class_weights=class_sample_freqs, 
    #              rand_transl_range=(5,25,25), bg_class_idx=0),
    #     ])

    # dataset_train_logpath = '/export/scratch3/bvdp/segmentation/OAR_segmentation/experiments/unet/dataset_train_log_shapes.txt'
    # dataset_val_logpath = '/export/scratch3/bvdp/segmentation/OAR_segmentation/experiments/unet/dataset_val_log.txt'
    train_dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=image_size, is_training=True, transform=transform_train, filter_label=filter_label, log_path=None)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batchsize, num_workers=1)
        
    for epoch in range(0, 1):
        for nbatches, (image, label) in enumerate(train_dataloader):
            print(nbatches, end=',')                                   
            
            image = image.data.cpu().numpy()
            label = label.view(*image.shape).data.cpu().numpy()            

            visualize_output(image[0, 0, :, :, :], label[0, 0, :, :, :], image[0, 0, :, :, :],
             out_dir, classes=train_dataset.classes, base_name="out_{}".format(nbatches))                

if __name__ == '__main__':
    main()


