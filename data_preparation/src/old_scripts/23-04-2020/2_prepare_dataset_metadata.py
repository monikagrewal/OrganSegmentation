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

# # Prepare metadata for dataset 

# root_dir1 = Path('/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_sigmoid/')
# root_dir2 = Path('/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/')
# root_dir3 = Path('/export/scratch3/bvdp/segmentation/data/modir_newdata_dicom/')

dataset = 'test'

root_dir = Path(f'/export/scratch3/bvdp/segmentation/data/MODIR_data_{dataset}_2019-12-17/')

output_dataset = f'../meta/dataset_{dataset}_2019-12-17.csv'
output_label_mapping = f'../meta/label_mapping_{dataset}_2019-12-17.json'

paths = list(root_dir.glob('**/annotations.json'))

print("total data: {}".format(len(paths)))
paths = [path.parent for path in paths]
# root_paths = [root_dir1]*len(paths1) + [root_dir2]*len(paths2) + [root_dir3]*len(paths3)
# ## Load annotations

label_classes = []
for path in tqdm(paths):
    with open(str(path / 'annotations.json')) as f:
        annotations = json.loads(f.read())
    # print(path)
    labels = [item["label_name"] for item in annotations]
    labels = list(set(labels))
    label_classes.append(labels)


df = pd.DataFrame(paths, columns=['path'])
# df = df.assign(root_path=root_paths)
df = df.assign(root_path=root_dir)
df = df.assign(patient_id=df.apply(lambda x: x.path.relative_to(x.root_path).parts[0],axis=1))
df = df.assign(labels=label_classes)

# ## Label class mapping logic


general_exclude = ['ctv','ptv','gtv', 'itv', 'prv', 'brachy']
include = ['rectum', 'hip', 'bowel', 'bladder', 'sigmoid', 'spinal', 'anal_canal', 'anal canal', 'blaas']

def check_invalid_numeric_label(s):
    '''
    Labels like bladder_33 need to be removed, because they are usually interpolations. 100 is always an actual image, so keep those
    '''
    match = re.search(r'\d+$', s)
    if match:
        if match.group() != '100':
            return True
    return False

def is_bladder(text):
    text = text.lower()
    result = False
    if "bladder" in text or "blaas" in  text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    for word in ['bt', 'bowel', 'mm', 'cm']:
        if word in text:
            result = False
    if check_invalid_numeric_label(text):
        result = False
    return result

def is_hip(text): 
    text = text.lower()
    if 'hip' in text and not 'hippoc' in text and not 'prothese' in text:
        return True
    else:
        return False

def is_rectum(text):
    text = text.lower()
    result = False
    if "rectum" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    if "bt" in text or "meso" in text:
        result = False
    if check_invalid_numeric_label(text):
        result = False
    return result

def is_bowel_bag(text):
    text = text.lower()
    result = False
    if "bowel_bag" in text or "bowel bag" in  text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    for word in ['cm', 'gy', 'cyste', 'legeblaas']:
        if word in text:
            result = False
    if check_invalid_numeric_label(text):
        result = False
    return result

def is_sigmoid(text):
    text = text.lower()
    result = False
    if "sigmoid" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    for word in ['bt', 'mm', 'cm']:
        if word in text:
            result = False
    if check_invalid_numeric_label(text):
        result = False
    return result

def is_spinal_cord(text):
    text = text.lower()
    result = False
    if "spinal" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    return result

def is_anal_canal(text):
    text = text.lower()
    if 'anal' in text and "canal" in text:
        return True
    else:
        return False

def is_rectum_merged(text):
    """
    merges rectum and anal canal
    """
    text = text.lower()
    result = False
    if ('anal' in text and "canal" in text) or "rectum" in text:
        return True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    if "bt" in text or "meso" in text:
        result = False
    if check_invalid_numeric_label(text):
        result = False
    return result


# ## Create and store label class mapping
label_classes_flat = reduce(lambda x,y: x+y, label_classes)

# class_detectors = [
#     ('bladder', is_bladder), ('hip', is_hip), ('rectum', is_rectum), 
#     ('spinal_cord', is_spinal_cord), ('sigmoid', is_sigmoid),
#     ('anal_canal', is_anal_canal), ('bowel_bag', is_bowel_bag)
# ]
# merging rectum and anal_canal and calling both rectum
# class_detectors = [
#     ('bladder', is_bladder), ('hip', is_hip), ('rectum', is_rectum_merged), 
#     ('spinal_cord', is_spinal_cord), ('sigmoid', is_sigmoid),
#     ('bowel_bag', is_bowel_bag)
# ]

class_detectors = [
    ('bladder', is_bladder), ('hip', is_hip), ('rectum', is_rectum), ('anal_canal', is_anal_canal),
    ('spinal_cord', is_spinal_cord), ('sigmoid', is_sigmoid),
    ('bowel_bag', is_bowel_bag)
]

mapping = {}

for class_name, class_detector in class_detectors:
    mapping[class_name] = list(np.unique([label for label in label_classes_flat if class_detector(label)]))


with open(output_label_mapping, 'w') as f:
    f.write(json.dumps(mapping))
    
inverse_mapping = {vx:k for k,v in mapping.items() for vx in v}



# ## Clean/filter dataset
# * apply mapping to labels
# * remove cases where organgs are labeled multiple times (e.g. bladder_0 & bladder_100)
# * remove cases that don't have any labels after mapping and filtering 



# def double_labels_removed(label_list):
#     labels_to_ignore = ['hip']
#     filtered_labels = [label for label, count in Counter(label_list).items() if (label in labels_to_ignore or count == 1)]
#     return filtered_labels

def map_and_clean(label_list):
    mapped_results = [inverse_mapping.get(label) for label in label_list]
    mapped_to_original = {}
    for k,v in list(zip(label_list, mapped_results)):
        mapped_to_original[v] = mapped_to_original.get(v, []) + [k]

    # ignore these when removing duplicates    
    labels_to_ignore = ['hip']
    mapped_counts = Counter(mapped_results)

    final_results = []
    for label, mapped_label in zip(label_list, mapped_results):
        if mapped_label is None:
            continue
        mapped_count = mapped_counts[mapped_label]
        keep_label = True
        if mapped_count > 1:
            options = mapped_to_original[mapped_label]
            # just select first option for now..
            if label != options[0]:
                keep_label = False

        if mapped_label in labels_to_ignore or keep_label:
            final_results.append((label, mapped_label))
    final_results_mapped = [v for k,v in final_results]
    final_results_original = [k for k,v in final_results]
    return final_results_original, final_results_mapped


# df = df.assign(labels_mapped=df.labels.map(lambda x: [inverse_mapping.get(label) for label in x if label in inverse_mapping]))
# df = df.assign(labels_clean=df.labels_mapped.map(double_labels_removed))

df_mapped = pd.DataFrame([map_and_clean(label_list) for label_list in df.labels.values], columns=['final_labels', 'final_labels_mapped'])
df = df.join(df_mapped)
df = df[df.final_labels_mapped.map(len) > 0]
df = df.join(pd.get_dummies(df.final_labels_mapped.apply(pd.Series).stack()).sum(level=0))


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


print(df_final.groupby("train").sum())

df_output = df_final.copy()

for col in['labels', 'final_labels', 'final_labels_mapped']:
    df_output[col] = df_output[col].map(lambda x: "|".join(x))

df_output.to_csv(output_dataset, index=False)