import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class AMCDataset(Dataset):
    def __init__(
        self,
        root_dir,
        meta_path,
        slice_annot_csv_path,
        classes=["background", "bowel_bag", "bladder", "hip", "rectum"],
        transform=None,
        log_path=None,
    ):
        """
        Args:
            root_dir (string): Directory containing data.
            jsonname (string): json filename that contains data info.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes

        self.meta_df = pd.read_csv(meta_path)

        if slice_annot_csv_path is not None:
            self.slice_annot = True
            # load slice_annot_csv and merge with meta_df
            slice_annot_df = pd.read_csv(slice_annot_csv_path)
            self.meta_df = pd.merge(
                self.meta_df, slice_annot_df, on=list(self.meta_df.columns), how="left"
            )

            # remove scans that have undersegmented bowel bag annotations from training
            try:
                self.meta_df = self.meta_df[self.meta_df["missing_annotation"] != 1]
            except KeyError as e:
                logging.warning(e)
                logging.warning("missing_annotation not present in meta_df. Continueing")
        else:
            self.slice_annot = False

        # filter rows in meta_df for which all the classes are present
        self.meta_df = self.meta_df[(self.meta_df[self.classes[1:]] >= 1).all(axis=1)]

        self.class2idx = {
            "background": 0,
            "bowel_bag": 1,
            "bladder": 2,
            "hip": 3,
            "rectum": 4,
        }
        self.log_path = log_path

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        study_path = Path(self.root_dir) / Path(row.path).relative_to(row.root_path)

        np_filepath = str(study_path / f"{row.SeriesInstanceUID}.npz")

        with np.load(np_filepath) as datapoint:
            volume, mask_volume = datapoint["volume"], datapoint["mask_volume"]

        if self.slice_annot:
            # annotations may not be avaialable for all row, use wherever available
            if not pd.isna(row.end_slice_scan):
                # Restricting field of view
                end_slice_scan = int(row.end_slice_scan)
                volume = volume[:end_slice_scan]
                mask_volume = mask_volume[:end_slice_scan]

            if not pd.isna(row.end_slice_annotation):
                # Restricting bowel bag annotation (semi-auto)
                end_slice_annotation = int(row.end_slice_annotation)
                mask_volume[end_slice_annotation:] = 0

        # Binary segmentation logic, only when not all classes are used
        if len(self.classes) != len(self.class2idx):
            class_idx = [self.class2idx[c] for c in self.classes]

            # Remove other classes & clip to binary
            mask_volume[~np.isin(mask_volume, class_idx)] = 0
            mask_volume[mask_volume > 1] = 1

        if self.transform is not None:
            volume, mask_volume = self.transform(volume, mask_volume)

        # add color dimension for 3d convolution
        volume = np.expand_dims(volume, 0)

        if self.log_path is not None:
            with open(self.log_path, "a") as log_file:
                log_file.write(
                    f"Shapes after transforms: {volume.shape}/{mask_volume.shape}\n\n"
                )
                sys.stdout.flush()

        return volume.astype(np.float32), mask_volume.astype(np.long)

    def partition(self, indices):
        self.meta_df = self.meta_df.iloc[indices]
        return self


class AMCDatasetNoisy(AMCDataset):
    def __init__(
        self,
        root_dir,
        meta_path,
        slice_annot_csv_path,
        classes=["background", "bowel_bag", "bladder", "hip", "rectum"],
        transform=None,
        log_path=None,
    ):
        """
        Args:
            root_dir (string): Directory containing data.
            jsonname (string): json filename that contains data info.
        """
        super().__init__(root_dir, meta_path, slice_annot_csv_path, classes=classes, \
                        transform=transform, log_path=log_path)
        self.meta_df = pd.read_csv(meta_path)

        if slice_annot_csv_path is not None:
            self.slice_annot = True
            # load slice_annot_csv and merge with meta_df
            slice_annot_df = pd.read_csv(slice_annot_csv_path)
            self.meta_df = pd.merge(
                self.meta_df, slice_annot_df, on=list(self.meta_df.columns), how="left"
            )

            # select only the scans that have undersegmented bowel bag annotations from training
            try:
                self.meta_df = self.meta_df[self.meta_df["missing_annotation"] == 1]
            except KeyError as e:
                logging.warning(e)
                logging.warning("missing_annotation not present in meta_df. Continueing")
        else:
            self.slice_annot = False

        # filter rows in meta_df for which all the classes are present
        self.meta_df = self.meta_df[(self.meta_df[self.classes[1:]] >= 1).all(axis=1)]

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        study_path = Path(self.root_dir) / Path(row.path).relative_to(row.root_path)

        np_filepath = str(study_path / f"{row.SeriesInstanceUID}.npz")

        with np.load(np_filepath) as datapoint:
            volume, mask_volume = datapoint["volume"], datapoint["mask_volume"]

        # Binary segmentation logic, only when not all classes are used
        if len(self.classes) != len(self.class2idx):
            class_idx = [self.class2idx[c] for c in self.classes]

            # Remove other classes & clip to binary
            mask_volume[~np.isin(mask_volume, class_idx)] = 0
            mask_volume[mask_volume > 1] = 1

        if self.transform is not None:
            volume, mask_volume = self.transform(volume, mask_volume)

        # add color dimension for 3d convolution
        volume = np.expand_dims(volume, 0)

        if self.log_path is not None:
            with open(self.log_path, "a") as log_file:
                log_file.write(
                    f"Shapes after transforms: {volume.shape}/{mask_volume.shape}\n\n"
                )
                sys.stdout.flush()

        return volume.astype(np.float32), mask_volume.astype(np.long)

    def partition(self, indices):
        self.meta_df = self.meta_df.iloc[indices]
        return self

    def add_samples(self, other_dataset):
        self.meta_df = pd.concat([self.meta_df, other_dataset.meta_df], axis=0)
        return self


class AMCDatasetPartialAnnotation(Dataset):
    def __init__(
        self,
        root_dir,
        meta_path,
        slice_annot_csv_path,
        classes=["background", "bowel_bag", "bladder", "hip", "rectum"],
        transform=None,
        log_path=None,
    ):
        """
        Args:
            root_dir (string): Directory containing data.
            jsonname (string): json filename that contains data info.
        """
        self.root_dir = root_dir
        self.transform = transform
        meta_df = pd.read_csv(meta_path)

        # load slice_annot_csv and merge with meta_df
        slice_annot_df = pd.read_csv(slice_annot_csv_path)
        self.meta_df = pd.merge(
            meta_df, slice_annot_df, on=list(meta_df.columns), how="left"
        )

        # remove scans that have undersegmented bowel bag annotations from training
        self.meta_df = self.meta_df[self.meta_df["missing_annotation"] != 1]

        self.classes = classes
        # filter rows in meta_df for which at least one class is absent
        self.meta_df = self.meta_df[(self.meta_df[self.classes[1:]] == 0).any(axis=1)]

        self.class2idx = {
            "background": 0,
            "bowel_bag": 1,
            "bladder": 2,
            "hip": 3,
            "rectum": 4,
        }
        self.log_path = log_path

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        study_path = Path(self.root_dir) / Path(row.path).relative_to(row.root_path)

        np_filepath = str(study_path / f"{row.SeriesInstanceUID}.npz")

        with np.load(np_filepath) as datapoint:
            volume, mask_volume = datapoint["volume"], datapoint["mask_volume"]

            # annotations may not be avaialable for all row, use wherever available
            if not pd.isna(row.end_slice_scan):
                # Restricting field of view
                end_slice_scan = int(row.end_slice_scan)
                volume = volume[:end_slice_scan]
                mask_volume = mask_volume[:end_slice_scan]

            if not pd.isna(row.end_slice_annotation):
                # Restricting bowel bag annotation (semi-auto)
                end_slice_annotation = int(row.end_slice_annotation)
                mask_volume[end_slice_annotation:] = 0

        # Binary segmentation logic, only when not all classes are used
        if len(self.classes) != len(self.class2idx):
            class_idx = [self.class2idx[c] for c in self.classes]

            # Remove other classes & clip to binary
            mask_volume[~np.isin(mask_volume, class_idx)] = 0
            mask_volume[mask_volume > 1] = 1

        if self.transform is not None:
            volume, mask_volume = self.transform(volume, mask_volume)

        # add color dimension for 3d convolution
        volume = np.expand_dims(volume, 0)

        if self.log_path is not None:
            with open(self.log_path, "a") as log_file:
                log_file.write(
                    f"Shapes after transforms: {volume.shape}/{mask_volume.shape}\n\n"
                )
                sys.stdout.flush()

        # create non-ambiguity mask
        classes = np.array(self.classes[1:])
        present_classes = classes[(row[classes] >= 1).to_list()].tolist()
        if len(present_classes) == len(classes):
            non_ambiguity_mask = np.ones_like(mask_volume)
        else:
            non_ambiguity_mask = np.zeros_like(mask_volume)
            for classname in present_classes:
                idx = self.class2idx[classname]
                non_ambiguity_mask[mask_volume == idx] = 1

        return (
            volume.astype(np.float32),
            mask_volume.astype(np.long),
            non_ambiguity_mask.astype(np.float32),
        )

    def partition(self, indices):
        self.meta_df = self.meta_df.iloc[indices]
        return self

    def add_samples(self, other_dataset):
        self.meta_df = pd.concat([self.meta_df, other_dataset.meta_df], axis=0)
        return self
