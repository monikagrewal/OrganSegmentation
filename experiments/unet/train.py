import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from torch.utils.data import Dataset, DataLoader
from tensorboard_logger import configure, log_value

import os, argparse
import numpy as np
import json
from skimage.io import imread, imsave
from model_unet import UNet
from torch_AMCDataset import AMCDataset

parser = argparse.ArgumentParser(description='Train UNet')
parser.add_argument("-out_dir", help="output directory", default="./runs/run_3d_no_balance")
parser.add_argument("-device", help="GPU number", type=int, default=0)
parser.add_argument("-depth", help="network depth", type=int, default=3)
parser.add_argument("-width", help="network width", type=int, default=16)
parser.add_argument("-image_size", help="image size", type=int, default=128)
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


def main():
    run_params = parse_input_arguments()

    device = "cuda:{}".format(run_params["device"])
    out_dir, nepochs, lr, batchsize = run_params["out_dir"], run_params["nepochs"], run_params["lr"], run_params["batchsize"]
    depth, width, image_size = run_params["depth"], run_params["width"], run_params["image_size"]
    
    out_dir_train = os.path.join(out_dir, "train")
    out_dir_val = os.path.join(out_dir, "val")
    out_dir_wts = os.path.join(out_dir, "weights")
    os.makedirs(out_dir_train, exist_ok=True)
    os.makedirs(out_dir_val, exist_ok=True)
    os.makedirs(out_dir_wts, exist_ok=True)
    configure(out_dir, flush_secs=5)

    best_dice = 0.65
    best_loss = 100.0

    root_dir = '/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_v2/'
    meta_path = '/export/scratch3/bvdp/segmentation/amc_dataprep/src/meta/dataset_v2.csv'
    label_mapping_path = '/export/scratch3/bvdp/segmentation/amc_dataprep/src/meta/label_mapping_v2.json'

    # TODO: add transform to dataset. Somethis like this:
 #    transform = custom_transforms.Compose([
	# 	# custom_transforms.RandomRotate3D(p=0.75),
	# 	custom_transforms.RandomBrightness(p=0.5),
	# 	custom_transforms.RandomElasticTransform3D(p=0.5, alpha=100, sigma=5), # TODO: fix elastictransform based on modification Monika
	# ])

    train_dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=image_size, is_training=True)
    val_dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=image_size, is_training=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batchsize)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batchsize)
    
    model = UNet(depth=depth, width=width, in_channels=1, out_channels=len(train_dataset.classes))
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

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
            # # TEMP!!! REMOVE
            # if nbatches > 10:
            #     break
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = torch.max(output, 1)
            pred = pred.data.cpu().numpy()
            label = label.data.cpu().numpy()
            acc = np.sum(pred == label) / pred.size
            train_acc += acc
            print("Iteration {}: Train Loss: {}, Train Accuracy: {}".format(nbatches, loss.item(), acc))
            log_value("Loss/train_loss", loss.item(), train_steps)
            train_steps += 1


            # if nbatches%100 == 0:
            #   pred = pred.reshape(label.shape)
            #   alpha = 0.6
            #   color = [1, 0, 0]
            #   for i in range(image.shape[2]):
            #       im = image.data.cpu().numpy()[0, 0, i, :, :]
            #       lbl = label[0, 0, i, :, :]
            #       prob = pred[0, 0, i, :, :]

            #       im = np.repeat(im.flatten(), 3).reshape(im.shape[0], im.shape[1], 3)
            #       lbl = np.repeat(lbl.flatten(), 3).reshape(lbl.shape[0], lbl.shape[1], 3)
            #       prob = np.repeat(prob.flatten(), 3).reshape(prob.shape[0], prob.shape[1], 3)

            #       im_plus_label = (1-alpha*lbl)*im + alpha*lbl*color
            #       im_plus_pred = (1-alpha*prob)*im + alpha*prob*color

            #       out_im = np.concatenate((im, im_plus_label, im_plus_pred), axis=1)
            #       imsave(os.path.join(out_dir_train, "iter_{}_{}_{}.jpg".format(epoch, nbatches, i)), (out_im*255).astype(np.uint8))

        train_loss = train_loss / float(nbatches+1)
        train_acc = train_acc / float(nbatches+1)

        # validation
        model.eval()
        val_loss = 0.
        tp, tn, fp, fn = 0, 0, 0, 0
        accuracy = 0
        for nbatches, (image, label) in enumerate(val_dataloader):
            # TEMP!!! REMOVE
            # if nbatches > 10:
            #     break
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, label)
                val_loss += loss.item()

            _, pred = torch.max(output, 1)
            pred = pred.data.cpu().numpy()
            label = label.data.cpu().numpy()
            accuracy += np.sum(pred == label) / pred.size

            print("Iteration {}: Validation Loss: {}".format(nbatches, loss.item()))
            log_value("Loss/validation_loss", loss.item(), val_steps)
            val_steps += 1

        val_loss = val_loss / float(nbatches+1)
        accuracy = accuracy / float(nbatches+1)


        print("EPOCH {}".format(epoch))
        print("Training Loss: {}, Training Accuracy: {}".format(train_loss, train_acc))
        print("Validation Loss: {}, Accuracy: {} \n".format(val_loss, accuracy))
        # log_value("Epochwise_Loss/train_loss", train_loss, epoch)
        # log_value("Epochwise_Loss/validation_loss", val_loss, epoch)
        # log_value("Metric/train_accuracy", train_acc, epoch)
        # log_value("Metric/validation_accuracy", accuracy, epoch)
        # log_value("Metric/validation_precision", precision, epoch)
        # log_value("Metric/validation_recall", recall, epoch)
        # log_value("Metric/validation_dice_score", dice, epoch)

        # # saving model
        # weights = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "train_accuracy": train_acc, "validation_accuracy": accuracy}
        # torch.save(weights, os.path.join(out_dir_wts, "model_{}.pth".format(epoch)))

        if val_loss <= best_loss:
            best_loss = val_loss
            weights = {"model": model.state_dict(), "epoch": epoch, "loss": val_loss}
            torch.save(weights, os.path.join(out_dir_wts, "best_model_no_balance.pth"))


if __name__ == '__main__':
    main()


