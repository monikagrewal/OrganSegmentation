import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class AMCDataset(Dataset):
    def __init__(
        self, root_dir, meta_path, is_training=True, transform=None, log_path=None
    ):
        """
        Args:
            root_dir (string): Directory containing data.
            jsonname (string): json filename that contains data info.
        """
        self.root_dir = root_dir
        self.is_training = is_training
        self.transform = transform
        self.meta_df = pd.read_csv(meta_path)

        self.meta_df = self.meta_df[self.meta_df["missing_annotation"] == 0]

        if is_training is not None:
            self.meta_df = self.meta_df[self.meta_df.train == is_training]

        self.classes = ["background", "bowel_bag", "bladder", "hip", "rectum"]
        # filter rows in meta_df for which all the classes are present
        self.meta_df = self.meta_df[(self.meta_df[self.classes[1:]] >= 1).all(axis=1)]

        self.class2idx = dict(zip(self.classes, range(len(self.classes))))
        self.log_path = log_path

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]

        # row = self.meta_df.loc[self.meta_df["path"]=="/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_train/2063253691_2850400153/20131011", :].iloc[0]  # noqa
        # print(row.path)
        study_path = Path(self.root_dir) / Path(row.path).relative_to(row.root_path)

        np_filepath = str(study_path / f"{row.SeriesInstanceUID}.npz")

        with np.load(np_filepath) as datapoint:
            volume, mask_volume = datapoint["volume"], datapoint["mask_volume"]

        end_slice_annotation = int(row.end_slice_annotation)
        end_slice_scan = int(row.end_slice_scan)
        volume = volume[:end_slice_scan]
        mask_volume = mask_volume[:end_slice_scan]
        mask_volume[end_slice_annotation:] = 0

        if self.transform is not None:
            volume, mask_volume = self.transform(volume, mask_volume)

        # add color channel for 3d convolution
        volume = np.expand_dims(volume, 0)

        if self.log_path is not None:
            with open(self.log_path, "a") as log_file:
                log_file.write(
                    f"Shapes after transforms: {volume.shape}/{mask_volume.shape}\n\n"
                )
                sys.stdout.flush()

        return volume.astype(np.float32), mask_volume.astype(np.long)


if __name__ == "__main__":
    from config import config

    dataset = AMCDataset(config.DATA_DIR, config.META_PATH, is_training=True)
