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
filter_label = ['bladder']
class_weights = [0.01, 0.99]
# class_weights = None

parser = argparse.ArgumentParser(description='Train UNet')
parser.add_argument("-out_dir", help="output directory", default="./runs/{}_fl10".format("_".join(filter_label)))
parser.add_argument("-device", help="GPU number", type=int, default=0)
parser.add_argument("-depth", help="network depth", type=int, default=4)
parser.add_argument("-width", help="network width", type=int, default=16)
parser.add_argument("-image_size", help="image size", type=int, default=512)
parser.add_argument("-image_depth", help="image depth", type=int, default=16)
parser.add_argument("-nepochs", help="number of epochs", type=int, default=100)
parser.add_argument("-lr", help="learning rate", type=float, default=0.01)
parser.add_argument("-batchsize", help="batchsize", type=int, default=1)


def parse_input_arguments():
    run_params = parser.parse_args()
    run_params = vars(run_params)
    out_dir = run_params["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    json.dump(run_params, open(os.path.join(out_dir, "run_parameters.json"), "w"))
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


def main():
    run_params = parse_input_arguments()

    device = "cuda:{}".format(run_params["device"])
    out_dir, nepochs, lr, batchsize = run_params["out_dir"], run_params["nepochs"], run_params["lr"], run_params["batchsize"]
    depth, width, image_size, image_depth = run_params["depth"], run_params["width"], run_params["image_size"], run_params["image_depth"]
    
    out_dir_train = os.path.join(out_dir, "train")
    out_dir_val = os.path.join(out_dir, "val")
    out_dir_wts = os.path.join(out_dir, "weights")
    os.makedirs(out_dir_train, exist_ok=True)
    os.makedirs(out_dir_val, exist_ok=True)
    os.makedirs(out_dir_wts, exist_ok=True)
    writer = SummaryWriter(out_dir)


    best_dice = 0.65
    best_loss = 100.0

    root_dir = '/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_train/'
    # meta_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/src/meta/dataset_train.csv'
    meta_path = "/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/{}.csv".format("_".join(filter_label))
    label_mapping_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/src/meta/label_mapping_train.json'

    # TODO: add transform to dataset. Somethis like this:
    transform_train = custom_transforms.Compose([
		# custom_transforms.RandomRotate3D(p=0.75),
		custom_transforms.RandomBrightness(),
        custom_transforms.RandomContrast(),
		custom_transforms.RandomElasticTransform3D_2(),
        custom_transforms.RandomRotate3D(),
        custom_transforms.CropDepthwise(crop_size=image_depth, crop_mode='annotation'),
        custom_transforms.CropInplane(crop_size=384, crop_mode='center')
	])

    transform_val = custom_transforms.Compose([
        custom_transforms.CropDepthwise(crop_size=image_depth, crop_mode='annotation'),
        custom_transforms.CropInplane(crop_size=384, crop_mode='center')
        ])


    train_dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=image_size, is_training=True, transform=transform_train, filter_label=filter_label)
    val_dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=image_size, is_training=False, transform=transform_val, filter_label=filter_label)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batchsize, num_workers=5)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batchsize, num_workers=5)
    
    model = UNet(depth=depth, width=width, in_channels=1, out_channels=len(train_dataset.classes))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model, optimizer = amp.initialize(model, optimizer)

    # if class_weights is None:
    #     criterion = nn.CrossEntropyLoss()
    #     # criterion = custom_losses.SoftDiceLoss()
    # else:
    #     # criterion = custom_losses.SoftDiceLoss(weight=torch.tensor(class_weights, device=device))
    #     criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    criterion = custom_losses.FocalLoss()

    # weights = torch.load(os.path.join(out_dir_wts, "best_model.pth"))["model"]
    # model.load_state_dict(weights)
    
    train_steps = 0
    val_steps = 0
    for epoch in range(0, nepochs):
        # training
        model.train()
        train_loss = 0.
        train_acc = 0.
        nbatches = 0
        for nbatches, (image, label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # output = F.softmax(output.permute(0,2,3,4,1).contiguous().view(-1, len(train_dataset.classes)), dim=1)
            # label_onehot = custom_losses.convert_idx_to_onehot(label, len(train_dataset.classes))
            # loss = criterion(output, label_onehot)
            # loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print("Iteration {}: Train Loss: {}".format(nbatches, loss.item()))
            writer.add_scalar("Loss/train_loss", loss.item(), train_steps)
            train_steps += 1


            if nbatches%25 == 0:
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
                    writer.add_scalar(f"accuracy/train_acc", accuracy, train_steps)
                    writer.add_scalar(f"recall/train/{classname}", recall[class_no], train_steps)
                    writer.add_scalar(f"precision/train/{classname}", precision[class_no], train_steps)
                    writer.add_scalar(f"dice/train/{classname}", dice[class_no], train_steps)


        train_loss = train_loss / float(nbatches+1)
        train_acc = train_acc / float(nbatches+1)

        # validation
        model.eval()
        val_loss = 0.
        for nbatches, (image, label) in enumerate(val_dataloader):
            # TEMP!!! REMOVE
            # if nbatches > 10:
            #     break
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, label)
                # output = F.softmax(output.permute(0,2,3,4,1).contiguous().view(-1, len(train_dataset.classes)), dim=1)
                # label_onehot = custom_losses.convert_idx_to_onehot(label, len(train_dataset.classes))
                # loss = criterion(output, label_onehot)
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
                writer.add_scalar(f"accuracy/val_acc", accuracy, val_steps)
                writer.add_scalar(f"recall/val/{classname}", recall[class_no], val_steps)
                writer.add_scalar(f"precision/val/{classname}", precision[class_no], val_steps)
                writer.add_scalar(f"dice/val/{classname}", dice[class_no], val_steps)

            # probably visualize
            if nbatches%10==0:
                visualize_output(image[0, 0, :, :, :], label[0, 0, :, :, :], output[0, 0, :, :, :],
                 out_dir_val, classes=train_dataset.classes, base_name="out_{}".format(epoch))


        val_loss = val_loss / float(nbatches+1)
        print("EPOCH {} = Training Loss: {}, Validation Loss: {}\n".format(epoch, train_loss, val_loss))

        if val_loss <= best_loss:
            best_loss = val_loss
            weights = {"model": model.state_dict(), "epoch": epoch, "loss": val_loss}
            torch.save(weights, os.path.join(out_dir_wts, "best_model.pth"))

    writer.close()


if __name__ == '__main__':
    main()


