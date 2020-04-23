#!/usr/bin/env python
# coding: utf-8

import os, glob
from pathlib import Path
import json
import numpy as np
import pandas as pd
from functools import reduce
from collections import Counter
from tqdm.auto import tqdm as tqdm
import re


dataset = 'train'

root_dir = Path(f'/export/scratch3/bvdp/segmentation/data/MODIR_data_preprocessed_{dataset}_09-04-2020/')

# output_dataset = f'../meta/dataset_{dataset}_2019-12-17.csv'
# output_label_mapping = f'../meta/label_mapping_{dataset}_2019-12-17.json'
output_dataset = f'dataset_{dataset}.csv'



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

# ## Create train/test set on CT level

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

for col in['labels']:
    df_output[col] = df_output[col].map(lambda x: "|".join(x))

df_output.to_csv(output_dataset, index=False)
