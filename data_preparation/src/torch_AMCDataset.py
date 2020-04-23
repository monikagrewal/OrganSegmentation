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


class AMCDataset(Dataset):
    def __init__(self, root_dir, meta_path, is_training=True, output_size=128, transform=None, log_path=None):
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

        self.classes = ['background', 'bowel_bag', 'bladder', 'hip', 'rectum']
                # filter rows in meta_df for which all the classes are present
        self.meta_df = self.meta_df[(self.meta_df[self.classes[1:]] >= 1).all(axis=1)]

        self.class2idx = dict(zip(self.classes, range(len(self.classes))))
        self.log_path = log_path



    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):     
        row = self.meta_df.iloc[idx]        

        # row = self.meta_df.loc[self.meta_df["path"]=="/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_train/2063253691_2850400153/20131011", :].iloc[0]
        # print(row.path)
        study_path = Path(self.root_dir) / Path(row.path).relative_to(row.root_path)                

        
        np_filepath = str(study_path / f'{row.SeriesInstanceUID}.npz')
        
        with np.load(np_filepath) as datapoint:
            volume, mask_volume = datapoint['volume'], datapoint['mask_volume']
        

        if self.transform is not None:
            volume, mask_volume = self.transform(volume, mask_volume)        

        # add color channel for 3d convolution
        volume = np.expand_dims(volume, 0)

        # if self.log_path is not None:
        #     with open(self.log_path, 'a') as log_file:                
        #         log_file.write("Shapes after transforms: " + str(volume.shape) + ' / ' + str(mask_volume.shape) + '\n\n')
        #         sys.stdout.flush()
        

        return volume.astype(np.float32), mask_volume.astype(np.long)
    


if __name__ == '__main__':
    import sys
    sys.path.append("..")

    root_dir = '/export/scratch3/bvdp/segmentation/data/MODIR_data_preprocessed_train_09-04-2020/'
    meta_path = "/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_09-04-2020.csv"


    dataset = AMCDataset(root_dir, meta_path, output_size=512, is_training=True)
        

