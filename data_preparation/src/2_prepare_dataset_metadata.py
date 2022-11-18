#!/usr/bin/env python
# coding: utf-8

import os, glob
from pathlib import Path
import json
import numpy as np
import pandas as pd
from functools import reduce
from collections import Counter
import re


dataset = 'test'

# root_dir = Path(f'/export/scratch3/bvdp/segmentation/data/MODIR_data_preprocessed_{dataset}_09-04-2020/')
# root_dir = Path(f'/export/scratch3/bvdp/segmentation/data/MODIR_data_preprocessed_{dataset}_25-06-2020/')
# root_dir = Path(f'/export/scratch2/bvdp/Data/Projects_DICOM_data/ThreeD/MODIR_data_{dataset}_split_preprocessed_21-08-2020')
root_dir = Path('/export/scratch3/grewal/Data/Projects_JPG_data/ThreeD/segmentation/MODIR_data_test_split')

# output_dataset = f'../meta/dataset_{dataset}_2019-12-17.csv'
# output_label_mapping = f'../meta/label_mapping_{dataset}_2019-12-17.json'
output_dataset = f'../meta/dataset_{dataset}_14-11-2022_deduplicated.csv'



paths = list(root_dir.glob('**/*.json'))

print("total data: {}".format(len(paths)))

meta_list = []
for path in paths:
    with open(str(path)) as f:
        meta = json.loads(f.read())
        meta_list.append(meta)
    
df = pd.DataFrame(meta_list)
df = df.assign(root_path=root_dir)
df = df.rename(columns={'output_directory': 'path'})

df = df.assign(patient_id=df.apply(lambda x: Path(x.path).relative_to(x.root_path).parts[0],axis=1))
# df = df.assign(labels=label_classes)

label_dummies = df.labels.apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0).astype(int)
df = df.join(label_dummies)

# remove for testing
#  Create train/test set on CT level (in case of test set, this column can be ignored)
if dataset=="train":
    np.random.seed(1234)
    train_frac = 0.80
    # all_ids = df.path.unique()
    all_ids = df.patient_id.unique()
    num_train_ids = int(len(all_ids) * train_frac)
    shuffled = np.random.permutation(all_ids)
    ids_train = shuffled[:num_train_ids]
    ids_test = shuffled[num_train_ids:]

    df_final = df.assign(train=df.patient_id.isin(ids_train))


    print(df_final.groupby("train")[label_dummies.columns].sum())

    df_output = df_final.copy()
else:
    df_output = df.copy()

for col in['labels']:
    df_output[col] = df_output[col].map(lambda x: "|".join(x))

if dataset=="test":
    """
    delete duplicate rows on [patient_id, series_date, labels, no_of_contours]
    delete corresponding json file as well as jpg image
    """
    duplicates_flag = df_output.duplicated(subset=["patient_id", "SeriesDate", "labels",\
                                             "No_of_contours", "image_size"],\
                                    keep="first")

    df_duplicates = df_output.loc[duplicates_flag]
    print(f"No. of duplicates: {len(df_duplicates)}")
    for i, row in df_duplicates.iterrows():
        json_path = os.path.join(str(row.path), f"{row.SeriesInstanceUID}.json")
        jpg_path = os.path.join(str(row.path), f"{row.SeriesInstanceUID}.jpg")
        npz_path = os.path.join(str(row.path), f"{row.SeriesInstanceUID}.npz")

        os.remove(json_path)
        os.remove(jpg_path)
        os.remove(npz_path)

    df_output = df_output.loc[duplicates_flag==False]

    # remove missing annotations
    complete_flag = []
    for i, row in df_output.iterrows():
        if row["hip"] and row["bowel_bag"] and \
            row["bladder"] and row["rectum"]:
            complete_flag.append(i)
        else:
            json_path = os.path.join(str(row.path), f"{row.SeriesInstanceUID}.json")
            jpg_path = os.path.join(str(row.path), f"{row.SeriesInstanceUID}.jpg")
            npz_path = os.path.join(str(row.path), f"{row.SeriesInstanceUID}.npz")

            os.remove(json_path)
            os.remove(jpg_path)
            os.remove(npz_path)
    
    df_output = df_output.loc[complete_flag, :]

df_output.to_csv(output_dataset, index=False)
