"""
run main script to visualize spleen ag jpg
use SpleenDataset class for training
"""
import json
import logging
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage.io import imsave
from torch.utils.data import Dataset


def normalize(image):
    image = (image - np.min(image)) / float(np.max(image) - np.min(image))
    return image


def apply_ww_wl(image, ww, wl):
    ub = wl + ww // 2
    lb = wl - ww // 2
    image[image > ub] = ub
    image[image < lb] = lb
    image = (image - lb) / float(ub - lb)
    return image


class SpleenDataset(Dataset):
    """Spleen Dataset."""

    def __init__(
        self,
        root_dir,
        jsonname="dataset.json",
        classes=["background", "spleen"],
        image_size=128,
        slice_thickness=5,
        transform=None,
        log_path=None,
    ):
        """
        Args:
                root_dir (string): Directory containing data.
                jsonname (string): json filename that contains data info.
        """
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_size = image_size
        self.slice_thickness = slice_thickness

        jsonpath = os.path.join(root_dir, jsonname)
        f = json.load(open(jsonpath, "r"))
        self.datainfo = f["training"]

    def __len__(self):
        return len(self.datainfo)

    def __getitem__(self, idx):
        sample = self.datainfo[idx]
        imgname, labelname = sample["image"], sample["label"]
        imgpath = os.path.join(self.root_dir, imgname)
        labelpath = os.path.join(self.root_dir, labelname)
        image, label = self.read_nifti_images(
            imgpath,
            labelpath,
            inplane_size=self.image_size,
            slice_thickness=self.slice_thickness,
        )

        if self.transform is not None:
            image, label = self.transform(image, label)

        # add channel axis; required for neural network training
        image = np.expand_dims(image, axis=0)

        return image.astype(np.float32), label.astype(np.long)


    @staticmethod
    def read_nifti_images(filepath, labelpath, inplane_size=256, slice_thickness=2):
        im = nib.load(filepath)
        org_slice_thickness = im.affine[2][2]
        im = im.get_fdata()
        label = nib.load(labelpath)
        label = label.get_fdata()

        # Apply WW and WL
        WW = 400
        WL = 50
        im = apply_ww_wl(im, WW, WL)
        im = normalize(im)

        # resample to given slice thickness and inplane size
        zoom_factor = [
            inplane_size / float(im.shape[0]),
            inplane_size / float(im.shape[1]),
            org_slice_thickness / slice_thickness,
        ]
        im = zoom(im, zoom_factor, order=1)
        label = zoom(label, zoom_factor, order=0)

        # image is currently in rl, pa, cc space;
        # convert it to ap, rl, cc space (so that we can see it right)
        im = im[:, ::-1, :]
        im = im.transpose(1, 0, 2)
        label = label[:, ::-1, :]
        label = label.transpose(1, 0, 2)

        # bring cc along depth dimension (D, H, W in Pytorch)
        im = im.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        return im, label

    def partition(self, indices):
        self.datainfo = [self.datainfo[i] for i in indices]
        return self


class DatasetPartialAnnotation(SpleenDataset):
    def __init__(
        self,
        root_dir,
        jsonname="dataset.json",
        classes=["background", "spleen"],
        image_size=128,
        slice_thickness=5,
        transform=None,
        log_path=None,
    ):
        """
        Args:
            root_dir (string): Directory containing data.
            jsonname (string): json filename that contains data info.
        """
        super().__init__(root_dir,
                    jsonname=jsonname,
                    classes=classes,
                    image_size=image_size,
                    slice_thickness=slice_thickness,
                    transform=transform,
                    log_path=log_path)

    def __len__(self):
        return len(self.datainfo)

    def __getitem__(self, idx):
        sample = self.datainfo[idx]
        imgname, labelname = sample["image"], sample["label"]
        imgpath = os.path.join(self.root_dir, imgname)
        labelpath = os.path.join(self.root_dir, labelname)
        image, label = self.read_nifti_images(
            imgpath,
            labelpath,
            inplane_size=self.image_size,
            slice_thickness=self.slice_thickness,
        )

        if self.transform is not None:
            image, label = self.transform(image, label)

        # add channel axis; required for neural network training
        image = np.expand_dims(image, axis=0)

        # TODO:create non-ambiguity mask s.t. 1 = label present, 0 = label ambiguous
        non_ambiguity_mask = np.ones_like(label)

        return (image.astype(np.float32), 
                label.astype(np.long),
                non_ambiguity_mask.astype(np.float32),
            )                

    def add_samples(self, other_dataset):
        """
        Adds samples from other dataset class of same type
        This is useful for cross-validation consistency between 
        fully annotated and fully + partially annotated dataset.

        CV splits are made from fully annotated dataset and 
        the samples from partially annotated are added only in the training split.
        """
        self.datainfo = self.datainfo + other_dataset.datainfo
        return self


if __name__ == "__main__":
    out_dir = "sanity"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    root_dir = "data/raw/Task09_Spleen"
    jsonname = "dataset.json"
    alpha = 0.5
    color = [1, 0, 0]
    batchsize = 1
    train_dataset = SpleenDataset(root_dir, jsonname)

    for batch_no in range(len(train_dataset)):
        images, labels = train_dataset[batch_no]
        logging.info(f"Image shape: {images.shape}, Labels shape: {labels.shape}")

        if batch_no < 5:
            nslices = images.shape[1]
            for i in range(nslices):
                im = images[0, i, :, :]
                lbl = labels[i, :, :]

                im = np.repeat(im.flatten(), 3).reshape(im.shape[0], im.shape[1], 3)
                lbl = np.repeat(lbl.flatten(), 3).reshape(lbl.shape[0], lbl.shape[1], 3)

                im_plus_label = (1 - alpha * lbl) * im + alpha * lbl * color
                out_im = np.concatenate((im, im_plus_label), axis=1)
                imsave(
                    os.path.join(out_dir, "iter_%d_%d.jpg" % (batch_no, i)),
                    (out_im * 255).astype(np.uint8),
                )
        else:
            break
