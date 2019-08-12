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
# from tqdm.auto import tqdm as tqdm


import sys


from utils import custom_transforms



class AMCDataset(Dataset):
    def __init__(self, root_dir, meta_path, label_mapping_path, is_training=True, image_size=512, output_size=128, transform=None):
        """
        Args:
            root_dir (string): Directory containing data.
            jsonname (string): json filename that contains data info.
        """
        self.root_dir = root_dir
        self.is_training = is_training
        self.transform = transform
        self.image_size = image_size
        self.output_size = output_size
        self.meta_df = pd.read_csv(meta_path)
        self.meta_df = self.meta_df[self.meta_df.train == is_training]
        # maybe remove label mapping altogether. Everything needed can also be derived from csv
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.loads(f.read())
#         self.inverse_label_mapping =  {vx:k for k,v in self.label_mapping.items() for vx in v}
        self.classes = ['background'] + list(sorted(self.label_mapping.keys()))
        self.class2idx = dict(zip(self.classes, range(len(self.classes))))


    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        study_path = Path(row.path)
        # temporary hack for local testing
        study_path = Path('/run/user/1000/gvfs/sftp:host=meteor03' +  row.path)
        with open(study_path / 'meta.json', 'r') as f:
            meta_list = json.loads(f.read())
        meta_sorted = sorted(meta_list, key=lambda x: x['SliceLocation'])
        with open(study_path / 'annotations.json', 'r') as f:
            annotations = json.loads(f.read())
#         mapped_labels = self.map_labels(annotations.keys())
        
        volume = self.load_volume(meta_sorted)
        mask_volume = self.create_mask(meta_sorted, annotations, row)
        

        if self.transform is not None:
			image, label = self.transform(image, label)

		# TODO: check the transformed mask_volume to make sure all the elements are valid class indici
		
        return volume, mask_volume
    
    
    def create_mask(self, meta_sorted, annotations, row):
        rescale_factor = self.output_size / self.image_size
        # mask_volume = np.zeros((len(self.classes), len(meta_sorted), self.output_size, self.output_size), dtype=np.float32)
        mask_volume = np.zeros((len(meta_sorted), self.output_size, self.output_size), dtype=np.long)
        uid_to_slice_idx = dict([(meta['uid'], i) for i, meta in enumerate(meta_sorted)])
        labels_to_mapped = dict(zip(row.final_labels.split("|"), row.final_labels_mapped.split("|")))
        
        for label, label_annotations in annotations.items():
#             label_mapped = self.inverse_label_mapping.get(label)
            label_mapped = labels_to_mapped[label]
            if label_mapped is None:
                continue
            label_idx = self.class2idx[label_mapped]
            for uid, coord_list in label_annotations.items():
                slice_idx = uid_to_slice_idx[uid]
                for coords in coord_list:
                    coords_np = np.array(coords) * rescale_factor
                    rr, cc = skimage.draw.polygon(coords_np[:,0], coords_np[:,1], shape=(self.output_size, self.output_size))
                    mask_volume[slice_idx, cc, rr] = label_idx
        return mask_volume
    
    def load_volume(self, meta_sorted):
        volume = np.zeros((1, len(meta_sorted), self.output_size, self.output_size), dtype=np.float32)

#         for i, meta in enumerate(meta_sorted):
        for i, meta in enumerate(meta_sorted):
            img_path = meta['output_path']
            # temp hack for local testing!
            img_path = '/run/user/1000/gvfs/sftp:host=meteor03' + img_path
            img = skimage.io.imread(img_path, as_gray=True) / 255.0
            img = skimage.transform.resize(img, (self.output_size, self.output_size))

            volume[0,i,:,:] = img
        return volume

# if __name__ == '__main__':
#     root_dir = '/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_v2/'
#     meta_path = '/export/scratch3/bvdp/segmentation/amc_dataprep/src/meta/dataset_v2.csv'
#     label_mapping_path = '/export/scratch3/bvdp/segmentation/amc_dataprep/src/meta/label_mapping_v2.json'
#     dataset = AMCDataset(root_dir, meta_path, label_mapping_path, output_size=256)

#     for i in range(len(dataset)):
#         volume, mask_volume = dataset[i]