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
from collections import Counter
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
    def __init__(self, root_dir, meta_path, label_mapping_path, is_training=True, output_size=128, transform=None, filter_label=[], log_path=None):
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
        # filter rows in meta_df for which all the classes are present
        self.meta_df = self.meta_df[(self.meta_df[filter_label] >= 1).all(axis=1)]
        # maybe remove label mapping altogether. Everything needed can also be derived from csv
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.loads(f.read())
#         self.inverse_label_mapping =  {vx:k for k,v in self.label_mapping.items() for vx in v}
        if len(filter_label)==0:
            self.classes = ['background'] + list(sorted(self.label_mapping.keys()))
        else:
            self.classes = ['background'] + filter_label
        self.class2idx = dict(zip(self.classes, range(len(self.classes))))
        self.log_path = log_path



    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):     
        row = self.meta_df.iloc[idx]        

        # row = self.meta_df.loc[self.meta_df["path"]=="/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_train/2063253691_2850400153/20131011", :].iloc[0]
        # print(row.path)
        study_path = Path(self.root_dir) / Path(row.path).relative_to(row.root_path)        
        with open(study_path / 'meta.json', 'r') as f:
            meta_list = json.loads(f.read())
        meta_sorted = sorted(meta_list, key=lambda x: x['SliceLocation'])
        with open(study_path / 'annotations.json', 'r') as f:
            annotations = json.loads(f.read())
        
        volume = self.load_volume(meta_sorted, study_path=study_path)
        mask_volume = self.create_mask(meta_sorted, annotations, row)
        # volume, mask_volume = normalize_fov(volume, mask_volume, meta_sorted[0]['PixelSpacing'], output_size=(self.output_size, self.output_size))

        # if self.log_path is not None:
        #     with open(self.log_path, 'a') as log_file:
        #         log_file.write(str(dict(row)) + '\n')
        #         log_file.write("Shapes: " + str(volume.shape) + ' / ' + str(mask_volume.shape) + '\n')
        #         sys.stdout.flush()

        if self.transform is not None:
            volume, mask_volume = self.transform(volume, mask_volume)        

        # add color channel for 3d convolution
        volume = np.expand_dims(volume, 0)

        # if self.log_path is not None:
        #     with open(self.log_path, 'a') as log_file:                
        #         log_file.write("Shapes after transforms: " + str(volume.shape) + ' / ' + str(mask_volume.shape) + '\n\n')
        #         sys.stdout.flush()
        

        return volume.astype(np.float32), mask_volume.astype(np.long)
    
    
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

    
    def load_volume(self, meta_sorted, study_path):
        img_list = []
        for i, meta in enumerate(meta_sorted):
            # new format should be 'output_path_rel' for new data generated with prepare_data script. Added this hack for backwards compatability with
            # data were currently using. SHould remove this in the future and just use 'output_path_rel'
            if 'output_path_rel' in meta:            
                img_path_rel = meta['output_path_rel']
            else:
                img_path_rel = meta['output_path'] 
            img_path = study_path / img_path_rel
            img = skimage.io.imread(img_path, as_gray=True) / 255.0
            if i==0:
                self.image_size = img.shape[0]

            img_list.append(img)

        volume = np.array(img_list)
        return volume

    def get_class_frequencies(self):
        counter = Counter()
        for idx in range(len(self.meta_df)):
            row = self.meta_df.iloc[idx]
            study_path = Path(row.path)        
            with open(study_path / 'meta.json', 'r') as f:
                meta_list = json.loads(f.read())
            meta_sorted = sorted(meta_list, key=lambda x: x['SliceLocation'])
            with open(study_path / 'annotations.json', 'r') as f:
                annotations = json.loads(f.read())
            
            # self.image_size gets set on first iteration of load_volume, which is needed
            # for mask creation
            self.load_volume(meta_sorted)
            mask_volume = self.create_mask(meta_sorted, annotations, row)
            label, count = np.unique(mask_volume, return_counts=True)
            counter.update(dict(zip(label, count)))
            print(idx, end=',', flush=True)
        return counter


def visualize(volume1, volume2, out_dir="./sanity", base_name=0):
    os.makedirs(out_dir, exist_ok=True)
    slices = volume1.shape[0]

    imlist = []
    for i in range(slices):
        im = np.concatenate([volume1[i], volume2[i]], axis=1)
        imlist.append(im)
        if len(imlist)==4:
            im = np.concatenate(imlist, axis=0)
            imsave(os.path.join(out_dir, "im_{}_{}.jpg".format(base_name, i)), (im*255).astype(np.uint8))
            imlist = []



if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from utils import custom_transforms

    filter_label = ["bowel_bag", "bladder", "hip", "rectum"]

    root_dir = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/'
    meta_path = "/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/{}.csv".format("_".join(filter_label))

    label_mapping_path = '/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/label_mapping_train.json'
    # transform = custom_transforms.Compose([
    #     custom_transforms.CropDepthwise(crop_size=48, crop_mode='random'),
    #     custom_transforms.CropInplane(crop_size=384, crop_mode='center')
    #     ])

    # transform2 =  custom_transforms.Compose([
    #     custom_transforms.RandomRotate3D(p=0.3),
    #     custom_transforms.RandomElasticTransform3D_2(p=0.7)
    #     ])
    # dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=512, is_training=True, transform=transform, filter_label=filter_label)

    # for i in range(len(dataset)):
    #     print(i)
    #     org_volume, mask_volume = dataset[i]
    #     new_volume = transform2(org_volume[0])

    #     visualize(org_volume[0], new_volume, base_name=i)
    #     if i>10:
    #         break

    dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=512, is_training=True, filter_label=filter_label)
    print("Calculating class frequencies on training set of size: ", len(dataset))
    counter = dataset.get_class_frequencies()
    print("Done.")
    class_counts = {dataset.classes[i]: v for i, (k,v) in enumerate(sorted(counter.items(), key=lambda x: x[0]))}
    print("Raw counts:\n", class_counts)
    # total_voxels = sum(counter.values())
    # print("Normalized counts: ", {k: v/total_voxels for k,v in counter.items()})

    inverse_weights = {k: 1/v for k,v in class_counts.items()}
    weight_sum = sum(inverse_weights.values())
    normalized_weights = {k: v/weight_sum for k,v in inverse_weights.items()}
    print("Normalized inverse weights:\n", normalized_weights)

    # equation to transform the inverse class frequencies/weights to something more uniform based on a beta parameter: 
    # x_transformed = x_inv**beta/ (x_inv**beta).sum()

    # result: 
    # {'background': 0.00029566728389566024, 'bowel_bag': 0.01284458558691968, 'bladder': 0.09494373118477141, 'hip': 0.3622280758050873, 'rectum': 0.5296879401393261}


