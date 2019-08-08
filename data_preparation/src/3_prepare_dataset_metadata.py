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

# # Prepare metadata for dataset 

root_dir = Path('/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_v2/')
paths = list(root_dir.glob('*/*'))

# ## Load annotations


label_classes = []
for path in tqdm(paths):
    with open(str(path / 'annotations.json')) as f:
        annotations = json.loads(f.read())
    label_classes.append(list(annotations.keys()))


df = pd.DataFrame(paths, columns=['path'])
df = df.assign(patient_id=df.path.map(lambda path: path.parent)).assign(labels=label_classes)

# ## Label class mapping logic


general_exclude = ['ctv','ptv','gtv', 'itv', 'prv', 'brachy']
include = ['rectum', 'hip', 'bowel', 'bladder', 'sigmoid', 'spinal', 'anal_canal', 'anal canal', 'blaas']

def is_bladder(text):
    result = False
    if "bladder" in text or "blaas" in  text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    if "bt" in text or "bowel" in text:
        result = False
    return result

def is_hip(text): 
    if 'hip' in text and not 'hippoc' in text and not 'prothese' in text:
        return True
    else:
        return False

def is_rectum(text):
    result = False
    if "rectum" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    if "bt" in text or "meso" in text:
        result = False
    return result

def is_bowel_bag(text):
    result = False
    if "bowel_bag" in text or "bowel bag" in  text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    return result

def is_sigmoid(text):
    result = False
    if "sigmoid" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    if "bt" in text:
        result = False
    return result

def is_spinal_cord(text):
    result = False
    if "spinal" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    return result

def is_anal_canal(text):
    if 'anal' in text and "canal" in text:
        return True
    else:
        return False


# ## Create and store label class mapping
label_classes_flat = reduce(lambda x,y: x+y, label_classes)

class_detectors = [
    ('bladder', is_bladder), ('hip', is_hip), ('rectum', is_rectum), 
    ('spinal_cord', is_spinal_cord), ('sigmoid', is_sigmoid),
    ('anal_canal', is_anal_canal), ('bowel_bag', is_bowel_bag)
]

mapping = {}

for class_name, class_detector in class_detectors:
    mapping[class_name] = list(np.unique([label for label in label_classes_flat if class_detector(label)]))


with open('meta/label_mapping_v2.json', 'w') as f:
    f.write(json.dumps(mapping))
    
inverse_mapping = {vx:k for k,v in mapping.items() for vx in v}



# ## Clean/filter dataset
# * apply mapping to labels
# * remove cases where organgs are labeled multiple times (e.g. bladder_0 & bladder_100)
# * remove cases that don't have any labels after mapping and filtering 



def double_labels_removed(label_list):
    labels_to_ignore = ['hip']
    filtered_labels = [label for label, count in Counter(label_list).items() if (label in labels_to_ignore or count == 1)]
    return filtered_labels



df = df.assign(labels_mapped=df.labels.map(lambda x: [inverse_mapping.get(label) for label in x if label in inverse_mapping]))
df = df.assign(labels_clean=df.labels_mapped.map(double_labels_removed))
df = df[df.labels_clean.map(len) > 0]
df = df.join(pd.get_dummies(df.labels_clean.apply(pd.Series).stack()).sum(level=0))


# ## Create train/test set

np.random.seed(1234)

train_frac = 0.75
patient_ids = df.patient_id.unique()
num_train_ids = int(len(patient_ids) * train_frac)
shuffled = np.random.permutation(patient_ids)
ids_train = shuffled[:num_train_ids]
ids_test = shuffled[num_train_ids:]

df_final = df.assign(train=df.patient_id.isin(ids_train))


print(df_final.groupby("train").sum())

df_output = df_final.copy()

for col in['labels', 'labels_mapped', 'labels_clean']:
    df_output[col] = df_output[col].map(lambda x: "|".join(x))

df_output.to_csv('meta/dataset_v2.csv', index=False)