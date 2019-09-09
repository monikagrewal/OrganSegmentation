import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, glob
from pathlib import Path
import json
import numpy as np
import skimage
import pandas as pd
from skimage.io import imread, imsave
from skimage.transform import resize
# from tqdm.auto import tqdm as tqdm

import sys

def normalize_fov(image, mask, pixelspacing, fov=512, output_size=(512, 512)):
    mask = mask.astype(np.float32)
    nslices = image.shape[0]
    shp = image.shape[1]
    org_fov = round(shp * float(pixelspacing[0]), 0)
    if org_fov < fov:
        pad = int((fov - org_fov)// (2 * float(pixelspacing[0])))
        image = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
        mask = np.pad(mask, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    elif org_fov > fov:
        crop = int((org_fov - fov)// (2 * float(pixelspacing[0])))
        image = image[:, crop : shp - crop, crop : shp - crop]
        mask = mask[:, crop : shp - crop, crop : shp - crop]

    image_resized = np.zeros((nslices, output_size[0], output_size[1]), dtype=np.float32)
    mask_resized = np.zeros((nslices, output_size[0], output_size[1]), dtype=np.float32)
    for i in range(nslices):
        image_resized[i, :, :] = resize(image[i, :, :], output_size, mode='constant')
        mask_resized[i, :, :] = resize(mask[i, :, :], output_size, mode='constant', order=0)
    return image_resized, mask_resized.astype(np.long)


class AMCDataset(Dataset):
    def __init__(self, root_dir, meta_path, label_mapping_path, is_training=True, output_size=128, transform=None, filter_label=[]):
        """
        Args:
            root_dir (string): Directory containing data.
            jsonname (string): json filename that contains data info.
        """
        self.root_dir = root_dir
        self.is_training = is_training
        self.transform = transform
        self.output_size = output_size
        self.meta_df = pd.read_csv(meta_path)
        self.meta_df = self.meta_df[self.meta_df.train == is_training]
        # maybe remove label mapping altogether. Everything needed can also be derived from csv
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.loads(f.read())
#         self.inverse_label_mapping =  {vx:k for k,v in self.label_mapping.items() for vx in v}
        if len(filter_label)==0:
            self.classes = ['background'] + list(sorted(self.label_mapping.keys()))
        else:
            self.classes = ['background'] + filter_label
        self.class2idx = dict(zip(self.classes, range(len(self.classes))))


    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        # row = self.meta_df.loc[self.meta_df["path"]=="/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_train/2063253691_2850400153/20131011", :].iloc[0]
        # print(row.path)
        study_path = Path(row.path)        
        with open(study_path / 'meta.json', 'r') as f:
            meta_list = json.loads(f.read())
        meta_sorted = sorted(meta_list, key=lambda x: x['SliceLocation'])
        with open(study_path / 'annotations.json', 'r') as f:
            annotations = json.loads(f.read())
        
        volume = self.load_volume(meta_sorted)
        mask_volume = self.create_mask(meta_sorted, annotations, row)
        # volume, mask_volume = normalize_fov(volume, mask_volume, meta_sorted[0]['PixelSpacing'], output_size=(self.output_size, self.output_size))

        if self.transform is not None:
            volume, mask_volume = self.transform(volume, mask_volume)

        # add color channel for 3d convolution
        volume = np.expand_dims(volume, 0)

        return volume, mask_volume.astype(np.long)
    
    
    def create_mask(self, meta_sorted, annotations, row):
        mask_volume = np.zeros((len(meta_sorted), self.image_size, self.image_size))
        uid_to_slice_idx = dict([(meta['uid'], i) for i, meta in enumerate(meta_sorted)])
        labels_to_mapped = dict(zip(row.final_labels.split("|"), row.final_labels_mapped.split("|")))
        for entry in annotations:
            label = entry["label_name"]
            label_mapped = labels_to_mapped.get(label)
            if label_mapped is None:
                continue
            label_idx = self.class2idx.get(label_mapped)
            if label_idx is None:
                continue

            uid = entry["uid"]
            coord_list = entry["coords"]
            slice_idx = uid_to_slice_idx.get(uid, None)
            if slice_idx is None:
                continue
            coords_np = np.array(coord_list)
            rr, cc = skimage.draw.polygon(coords_np[:,0], coords_np[:,1], shape=(self.image_size, self.image_size))
            mask_volume[slice_idx, cc, rr] = label_idx
        return mask_volume

    
    def load_volume(self, meta_sorted):
        img_list = []
        for i, meta in enumerate(meta_sorted):
            img_path = meta['output_path']
            img = skimage.io.imread(img_path, as_gray=True) / 255.0
            if i==0:
                self.image_size = img.shape[0]

            img_list.append(img)

        volume = np.array(img_list)
        return volume



if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from utils import custom_transforms

    filter_label = ["bladder"]

    root_dir = '/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_train/'
    meta_path = "/export/scratch3/grewal/OAR_segmentation/data_preparation/src/meta/{}.csv".format("_".join(filter_label))
    label_mapping_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/src/meta/label_mapping_train.json'
    transform = custom_transforms.Compose([
        custom_transforms.CropDepthwise(crop_size=16, crop_mode='annotation'),
        custom_transforms.CropInplane(crop_size=384, crop_mode='center')
        ])
    dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=512, is_training=True, transform=transform, filter_label=filter_label)

    for i in range(len(dataset)):
        volume, mask_volume = dataset[i]

    # sftp://grewal@meteor03/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_train/3955374440_1949203334/20140610/2