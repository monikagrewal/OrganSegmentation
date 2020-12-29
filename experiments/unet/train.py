import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
import pandas as pd

import os, argparse
import numpy as np
import cv2
from scipy import signal
import json
from skimage.io import imread, imsave
from sklearn.metrics import confusion_matrix
from model_unet import UNet
from torch_AMCDataset import AMCDataset
import sys
sys.path.append("..")
from utils import custom_transforms, custom_losses

from sacred import Experiment
ex = Experiment()


@ex.config
def my_config():
    device = 0
    out_dir = "./runs/tmp"
    # experiment_log_dir = "./experiment_logs/"
    load_weights = False
    # filter_label = ['bowel_bag', 'bladder', 'hip', 'rectum']
    depth=4 # network depth
    width=64 # network width
    # image_size=256
    # crop_sizes=[32,192,192]
    # image_depth=48    
    image_depth=32
    nepochs=100
    early_stopping_patience=None # number of evals with no improvement before stopping training (None = deactivate)
    lr=0.001 # learning rate
    weight_decay = 1e-4
    batchsize=1
    accumulate_batches=1
    loss_function='soft_dice'
    class_weights=None
    class_sample_freqs=[1,1,1,1,1] # sample freq weight per class    
    gamma=1
    alpha=None
    image_scale_inplane=None
    augmentation_brightness=0
    augmentation_contrast=0
    augmentation_rotate3d=0


"""
all labels:
['anal_canal', 'bowel_bag', 'bladder', 'hip', 'rectum', 'sigmoid', 'spinal_cord']
"""

# parser = argparse.ArgumentParser(description='Train UNet')
# parser.add_argument("-filter_label", help="list of labels to filter",
#                         nargs='+', default=['bowel_bag', 'bladder', 'hip', 'rectum'])
# parser.add_argument("-out_dir", help="output directory", type=str, default="./runs/tmp")    
# parser.add_argument("-device", help="GPU number", type=int, default=0)
# parser.add_argument("-load_weights", help="load weights", type=str, default='False')
# parser.add_argument("-depth", help="network depth", type=int, default=4)
# parser.add_argument("-width", help="network width", type=int, default=64)
# parser.add_argument("-image_size", help="image size", type=int, default=512)
# parser.add_argument("-crop_sizes", help="crop sizes", nargs='+', type=int, default=[48,192,192])
# parser.add_argument("-image_depth", help="image depth", type=int, default=48)
# parser.add_argument("-nepochs", help="number of epochs", type=int, default=500)
# parser.add_argument("-lr", help="learning rate", type=float, default=0.01)
# parser.add_argument("-batchsize", help="batchsize", type=int, default=1)
# parser.add_argument("-accumulate_batches", help="batchsize", type=int, default=16)
# parser.add_argument("-loss_function", help="loss function", default='cross_entropy')
# parser.add_argument("-class_weights", nargs='+', type=float, help="class weights", default=None)
# parser.add_argument("-class_sample_freqs", nargs='+', type=float, help="sample freq weight per class", default=[1,1,1,1,1])
# parser.add_argument("-gamma", help="loss function", type=float, default='1') 
# parser.add_argument("-alpha", help="loss function", type=float, nargs='+', default=None)


def process_input_arguments(run_params):
    # run_params = parser.parse_args()
    # run_params = vars(run_params)
    # run_params["load_weights"] = True if run_params["load_weights"] in ['True', 'true', '1'] else False
    out_dir_base = run_params["out_dir"]
    loss_function, alpha, gamma = run_params['loss_function'], run_params['alpha'], run_params['gamma']
    device = "cuda:{}".format(run_params["device"])
    # device = "cuda:0"
    if loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        run_params["out_dir"] = os.path.join(out_dir_base, f'{loss_function}') 
    elif loss_function == 'focal_loss':
        if alpha is not None: 
            alpha_string = "_".join([str(x) for x in alpha])
            alpha = torch.tensor(alpha, device=device)
            run_params["out_dir"] = os.path.join(out_dir_base, f'{loss_function}_gamma_{gamma}_alpha_{alpha_string}')
        else:
            run_params["out_dir"] = os.path.join(out_dir_base, f'{loss_function}_gamma_{gamma}')
        criterion = custom_losses.FocalLoss(gamma=gamma, alpha=alpha)
        
    elif loss_function == 'soft_dice':
        criterion = custom_losses.SoftDiceLoss(drop_background=False)
        run_params["out_dir"] = os.path.join(out_dir_base, f'{loss_function}')
    elif loss_function == 'weighted_cross_entropy':
        class_weights = run_params["class_weights"]
        class_weights_string = "_".join([str(x) for x in class_weights])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
        run_params["out_dir"] = os.path.join(out_dir_base, f'{loss_function}_{class_weights_string}')
    elif loss_function == 'weighted_soft_dice':
        class_weights = run_params["class_weights"]
        criterion = custom_losses.SoftDiceLoss(weight=torch.tensor(class_weights, device=device))
        run_params["out_dir"] = os.path.join(out_dir_base, f'{loss_function}')
    else:
        raise ValueError(f'unknown loss function: {loss_function}')

    os.makedirs(run_params["out_dir"], exist_ok=True)

    json.dump(run_params, open(os.path.join(run_params["out_dir"], "run_parameters.json"), "w"))
    return run_params, criterion




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

    recall = [round(item, 2) for item in recall]
    precision = [round(item, 2) for item in precision]
    dice = [round(item, 2) for item in dice]

    print("accuracy = {}, recall = {}, precision = {}, dice = {}".format(accuracy, recall, precision, dice))
    return accuracy, recall, precision, dice     


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


@ex.automain
def main(_config):
    # print("CONFIG UPDATES: ", config_updates)
    params = _config
    # if config_updates is not None:
        # params.update(config_updates['params'])
        # params.update(config_updates)
    # params = config['params']
    print("PARAMS: ", params)
    # sys.exit()

    device_number = params["device"]
    run_params, criterion = process_input_arguments(params.copy())
    print("RUN PARAMS: ", run_params)
    
    device = "cuda:{}".format(run_params["device"])

    # filter_label = run_params["filter_label"]
    out_dir, nepochs, lr, batchsize = run_params["out_dir"], run_params["nepochs"], run_params["lr"], run_params["batchsize"]
    # depth, width, image_size, crop_sizes, image_depth = run_params["depth"], run_params["width"], run_params["image_size"], run_params["crop_sizes"], run_params["image_depth"]
    depth, width, image_depth = run_params["depth"], run_params["width"], run_params["image_depth"]
    image_scale_inplane = run_params["image_scale_inplane"]
    augmentation_brightness= run_params["augmentation_brightness"]
    augmentation_contrast= run_params["augmentation_contrast"]
    augmentation_rotate3d= run_params["augmentation_rotate3d"]
    
    augmentation_options = dict(
        augmentation_brightness={
            1: dict(p= 0.5, rel_addition_range = (-0.2,0.2))
        },
        augmentation_contrast={
            1: dict(p=0.5, contrast_mult_range=(0.8,1.2))
        },
        augmentation_rotate3d={
            1: dict(p=0.3, x_range=(-20,20), y_range=(0,0), z_range=(0,0)),
            2: dict(p=0.3, x_range=(-10,10), y_range=(0,0), z_range=(0,0)),
            3: dict(p=0.3, x_range=(-10,10), y_range=(-10,10), z_range=(-10,10))
        }
    )

    accumulate_batches = run_params["accumulate_batches"]
    class_sample_freqs = run_params["class_sample_freqs"]

    out_dir_train = os.path.join(out_dir, "train")
    out_dir_val = os.path.join(out_dir, "val")
    out_dir_proper_val = os.path.join(out_dir, "proper_val")
    out_dir_wts = os.path.join(out_dir, "weights")
    out_dir_epoch_results = os.path.join(out_dir, "epoch_results")
    os.makedirs(out_dir_train, exist_ok=True)
    os.makedirs(out_dir_val, exist_ok=True)
    os.makedirs(out_dir_proper_val, exist_ok=True)
    os.makedirs(out_dir_wts, exist_ok=True)
    os.makedirs(out_dir_epoch_results, exist_ok=True)
    writer = SummaryWriter(out_dir)

    early_stopping_patience = run_params['early_stopping_patience']
    epochs_no_improvement = 0
    best_mean_dice = 0.0
    best_loss = 100.0

    # root_dir = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/'
    
    # meta_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/src/meta/dataset_train.csv'
    # meta_path = "/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/{}.csv".format("_".join(filter_label))
    # label_mapping_path = '/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/label_mapping_train.json'

    # root_dir = 'modir_newdata_dicom'
    # meta_path = "/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_2019-10-22_fixed.csv"
    # label_mapping_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/label_mapping_train_2019-10-22.json'
    
    # meta_path = "/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_2019-12-17.csv"
    # TMP REMOVE!!!!
    # meta_path = "/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_2019-12-17_filtered_for_binary.csv"
    # root_dir = '/export/scratch3/bvdp/segmentation/data/MODIR_data_train_2019-12-17/'
    # label_mapping_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/label_mapping_train_2019-12-17.json'

    # root_dir = '/export/scratch2/bvdp/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split_preprocessed_25-06-2020/'
    root_dir = '/export/scratch2/bvdp/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split_preprocessed_21-08-2020/'
    # meta_path = "/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_21-07-2020_slice_annot.csv"
    meta_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_21-08-2020_slice_annot.csv'




    # crop_inplane = 217
    # image_size = 217
    
    transform_any = custom_transforms.ComposeAnyOf([])
    if augmentation_brightness != 0:
        brightness_settings = augmentation_options['augmentation_brightness'][augmentation_brightness]
        print(f"Adding random brightness augmentation with params: {brightness_settings}")
        transform_any.transforms.append(custom_transforms.RandomBrightness(**brightness_settings))
    if augmentation_contrast != 0:
        contrast_settings = augmentation_options['augmentation_contrast'][augmentation_contrast]
        print(f"Adding random contrast augmentation with params: {contrast_settings}")
        transform_any.transforms.append(custom_transforms.RandomContrast(**contrast_settings))
    if augmentation_rotate3d != 0:
        rotate3d_settings = augmentation_options['augmentation_rotate3d'][augmentation_rotate3d]
        print(f"Adding random rotate3d augmentation with params: {rotate3d_settings}")
        transform_any.transforms.append(custom_transforms.RandomRotate3D(**rotate3d_settings))

    transform_train = custom_transforms.Compose([
        transform_any,
        custom_transforms.CropDepthwise(crop_size=image_depth, crop_mode='random')
        # custom_transforms.CustomResize(output_size=image_size),
        # custom_transforms.CropInplane(crop_size=crop_inplane, crop_mode='center'),
        # custom_transforms.RandomBrightness(p= 0.5, rel_addition_range = (-0.2,0.2)),
        # custom_transforms.RandomContrast(p=0.5, contrast_mult_range=(0.8,1.2)),
        # custom_transforms.RandomElasticTransform3D_2(p=0.7),
        # custom_transforms.RandomRotate3D(p=0.3)      
    ])



    transform_val = custom_transforms.Compose([
        custom_transforms.CropDepthwise(crop_size=image_depth, crop_mode='random'),
        # custom_transforms.CropInplane(crop_size=crop_inplane, crop_mode='center'),
        # custom_transforms.CustomResize(output_size=image_size),
        # custom_transforms.CropInplane(crop_size=384, crop_mode='center')
        # custom_transforms.CropInplane(crop_size=crop_inplane, crop_mode='center')
    ])


    transform_val_sliding_window = custom_transforms.Compose([
        # custom_transforms.CustomResize(output_size=image_size),
        # custom_transforms.CropInplane(crop_size=crop_inplane, crop_mode='center'),
    ])

    if image_scale_inplane is not None:
        transform_train.transforms.append(custom_transforms.CustomResize(scale=image_scale_inplane))
        transform_val.transforms.append(custom_transforms.CustomResize(scale=image_scale_inplane))
        transform_val_sliding_window.transforms.append(custom_transforms.CustomResize(scale=image_scale_inplane))        
    # transform_train = custom_transforms.Compose([
    #     custom_transforms.CustomResize(output_size=image_size),
    #     custom_transforms.CropLabel(p=1.0, crop_sizes=crop_sizes, class_weights=class_sample_freqs, 
    #              rand_transl_range=(5,25,25), bg_class_idx=0),        
    #     custom_transforms.RandomBrightness(),
    #     custom_transforms.RandomContrast(),
    #     # custom_transforms.RandomElasticTransform3D_2(p=0.7),
    #     custom_transforms.RandomRotate3D(p=0.3)      
    # ])    

    # transform_val = custom_transforms.Compose([
    #     custom_transforms.CustomResize(output_size=image_size),
    #     custom_transforms.CropLabel(p=1.0, crop_sizes=crop_sizes, class_weights=class_sample_freqs, 
    #              rand_transl_range=(5,25,25), bg_class_idx=0),
    #     ])




    
    # dataset_train_logpath = '/export/scratch3/bvdp/segmentation/OAR_segmentation/experiments/unet/dataset_train_log_shapes.txt'
    # dataset_val_logpath = '/export/scratch3/bvdp/segmentation/OAR_segmentation/experiments/unet/dataset_val_log.txt'
    
    train_dataset = AMCDataset(root_dir, meta_path, is_training=True, transform=transform_train, log_path=None)
    val_dataset = AMCDataset(root_dir, meta_path, is_training=False, transform=transform_val, log_path=None)
    proper_val_dataset = AMCDataset(root_dir, meta_path, is_training=False, transform=transform_val_sliding_window, log_path=None)

    # train_dataset.meta_df = train_dataset.meta_df.iloc[:4]
    # val_dataset.meta_df = val_dataset.meta_df.iloc[:4]
    # proper_val_dataset.meta_df = proper_val_dataset.meta_df.iloc[:4]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batchsize, num_workers=3)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batchsize, num_workers=3)
    proper_val_dataloader = DataLoader(proper_val_dataset, shuffle=False, batch_size=batchsize, num_workers=3)

    torch.manual_seed(0)
    np.random.seed(0)

    model = UNet(depth=depth, width=width, in_channels=1, out_channels=len(train_dataset.classes))
    model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    weight_decay = run_params['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=0.001)
    model, optimizer = amp.initialize(model, optimizer)

    if run_params["load_weights"]:
        weights = torch.load(os.path.join(out_dir_wts, "best_model.pth"), map_location=device)["model"]
        model.load_state_dict(weights)
    

    proper_eval_every_epochs = 1
    slice_weighting = True

    all_epoch_results = []
    train_steps = 0
    val_steps = 0

    for epoch in range(0, nepochs):
        epoch_results = {'epoch': epoch}
        # training
        model.train()
        train_loss = 0.
        train_acc = 0.
        nbatches = 0
        # accumulate gradients over multiple batches (equivalent to bigger batchsize, but without memory issues)
        # Note: depending on the reduction method in the loss function, this might need to be divided by the number
        #   of accumulation iterations to be truly equivalent to training with bigger batchsize
        # for accumulation in range(accumulate_batches):
        # i_accumulation = 0
        accumulated_batches = 0
        for nbatches, (image, label) in enumerate(train_dataloader):        
            print("Image shape: ", image.shape)
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion(output, label)

            # to make sure accumulated loss equals average loss in batch and won't depend on accumulation batch size
            loss = loss / accumulate_batches

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if ((nbatches+1) % accumulate_batches) == 0:
                accumulated_batches += 1
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                print("Iteration {}: Train Loss: {}".format(nbatches, loss.item()))
                writer.add_scalar("Loss/train_loss", loss.item(), train_steps)
                train_steps += 1  

                # if not a full batch left, break out of epoch to prevent wasted computation
                # on sample that won't add up to a full batch and therefore won't result in 
                # a step
                if len(train_dataloader) - (nbatches+1) < accumulate_batches:
                    break


            if (nbatches % accumulate_batches*3) == 0 or nbatches == len(train_dataloader)-1:
                with torch.no_grad():
                    image = image.data.cpu().numpy()
                    label = label.view(*image.shape).data.cpu().numpy()
                    output = torch.argmax(output, dim=1).view(*image.shape)
                    output = output.data.cpu().numpy()
                    
                    # calculate metrics and probably visualize prediction
                    accuracy, recall, precision, dice = calculate_metrics(label, output, classes=train_dataset.classes)

                    visualize_output(image[0, 0, :, :, :], label[0, 0, :, :, :], output[0, 0, :, :, :],
                     out_dir_train, classes=train_dataset.classes, base_name="out_{}".format(epoch))
                    
                    # log metrics
                    for class_no, classname in enumerate(train_dataset.classes):
                        writer.add_scalar(f"accuracy/train/{classname}", accuracy[class_no], train_steps)
                        writer.add_scalar(f"recall/train/{classname}", recall[class_no], train_steps)
                        writer.add_scalar(f"precision/train/{classname}", precision[class_no], train_steps)
                        writer.add_scalar(f"dice/train/{classname}", dice[class_no], train_steps)


        train_loss = train_loss / float(accumulated_batches)
        train_acc = train_acc / float(accumulated_batches)
        epoch_results.update({'train_loss': train_loss})
        # validation
        model.eval()
        val_loss = 0.
        for nbatches, (image, label) in enumerate(val_dataloader):            
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, label)
                val_loss += loss.item()

                print("Iteration {}: Validation Loss: {}".format(nbatches, loss.item()))
                writer.add_scalar("Loss/validation_loss", loss.item(), val_steps)
                val_steps += 1

                image = image.data.cpu().numpy()
                label = label.view(*image.shape).data.cpu().numpy()
                output = torch.argmax(output, dim=1).view(*image.shape)
                output = output.data.cpu().numpy()

                accuracy, recall, precision, dice = calculate_metrics(label, output, classes=train_dataset.classes)
                # log metrics
                for class_no, classname in enumerate(train_dataset.classes):
                    writer.add_scalar(f"accuracy/val/{classname}", accuracy[class_no], val_steps)
                    writer.add_scalar(f"recall/val/{classname}", recall[class_no], val_steps)
                    writer.add_scalar(f"precision/val/{classname}", precision[class_no], val_steps)
                    writer.add_scalar(f"dice/val/{classname}", dice[class_no], val_steps)

            # probably visualize
            if nbatches%10==0:
                visualize_output(image[0, 0, :, :, :], label[0, 0, :, :, :], output[0, 0, :, :, :],
                 out_dir_val, classes=train_dataset.classes, base_name="out_{}".format(epoch))


        val_loss = val_loss / float(nbatches+1)
        epoch_results.update({'val_loss': val_loss})
        print("EPOCH {} = Training Loss: {}, Validation Loss: {}\n".format(epoch, train_loss, val_loss))
        writer.add_scalar("epoch_loss/train_loss", train_loss, epoch)
        writer.add_scalar("epoch_loss/val_loss", val_loss, epoch)

        if val_loss <= best_loss:
            best_loss = val_loss
            weights = {"model": model.state_dict(), "epoch": epoch, "loss": val_loss}
            torch.save(weights, os.path.join(out_dir_wts, "best_model_loss.pth"))



        if proper_eval_every_epochs is not None and (epoch+1) % proper_eval_every_epochs == 0:
            metrics = np.zeros((4, len(train_dataset.classes)))
            min_depth = 2**depth
            model.eval()

            for nbatches, (image, label) in enumerate(proper_val_dataloader):
                label = label.view(*image.shape).data.cpu().numpy()
                with torch.no_grad():
                    nslices = image.shape[2]
                    image = image.to(device)

                    output = torch.zeros(batchsize, len(train_dataset.classes), *image.shape[2:])
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
                            weights = torch.tensor(weights, dtype=torch.float32)

                            output[:,:, indices, :,:] += mini_output.to(device='cpu', dtype=torch.float32)*weights.view(1,1,actual_slices,1,1) 
                            slice_overlaps[0,0,indices,0,0] += weights
                        else:
                            output[:,:, indices, :,:] += mini_output.to(device='cpu', dtype=torch.float32)
                            slice_overlaps[0,0,indices,0,0] +=  1 
                            
                    output = output / slice_overlaps
                    output = torch.argmax(output, dim=1).view(*image.shape)        

                image = image.data.cpu().numpy()
                output = output.data.cpu().numpy()                    
                im_metrics = calculate_metrics(label, output, classes=train_dataset.classes)
                metrics = metrics + im_metrics

                # probably visualize
                visualize_output(image[0, 0, :, :, :], label[0, 0, :, :, :], output[0, 0, :, :, :],
                 out_dir_proper_val, classes=train_dataset.classes, base_name=f"out_{nbatches}")
            metrics /= nbatches + 1
            results = f"accuracy = {metrics[0]}\nrecall = {metrics[1]}\nprecision = {metrics[2]}\ndice = {metrics[3]}\n"
            print(f"Proper evaluation results:\n{results}")
            recall, precision, dice = metrics[1], metrics[2], metrics[3]
            for class_no, classname in enumerate(train_dataset.classes):
                writer.add_scalar(f"sw_validation/recall/{classname}", recall[class_no], epoch)
                writer.add_scalar(f"sw_validation/precision/{classname}", precision[class_no], epoch)
                writer.add_scalar(f"sw_validation/dice/{classname}", dice[class_no], epoch)
                epoch_results.update({
                    f'recall_{classname}': recall[class_no],
                    f'precision_{classname}': precision[class_no],
                    f'dice_{classname}': dice[class_no]
                })

            mean_dice = np.mean(dice[1:])
            epoch_results.update({'mean_dice': mean_dice})
            if mean_dice >= best_mean_dice:
                best_mean_dice = mean_dice
                epochs_no_improvement = 0
                weights = {"model": model.state_dict(), "epoch": epoch, "mean_dice": mean_dice}
                torch.save(weights, os.path.join(out_dir_wts, "best_model.pth"))
            else:
                epochs_no_improvement += 1

            all_epoch_results.append(epoch_results)
            if early_stopping_patience is not None and epochs_no_improvement >= early_stopping_patience:
                print(f"{epochs_no_improvement} epochs without improvement. Stopping training!")
                break

    results_df = pd.DataFrame(all_epoch_results)
    results_df.to_csv(os.path.join(out_dir_epoch_results, 'epoch_results.csv'), index=False)

    writer.close()


# if __name__ == '__main__':
#     main()


