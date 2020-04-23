import os
from pathlib import Path
import pandas as pd
import re
import json

meta_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_2019-10-22_fixed.csv'

df = pd.read_csv(meta_path)
for path in df.path:
	json_path = Path(path) / 'meta.json'
	with open(json_path) as f:
		json_string = f.read()
	# /export/scratch3/grewal/Data/Projects_JPG_data/segmentation_prepared_data/AMC_dicom_train/3409170930_3386939515/20110203
	json_fixed = json_string.replace('/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train', '/export/scratch3/grewal/Data/Projects_JPG_data/segmentation_prepared_data/AMC_dicom_train')
	json_fixed = json_fixed.replace(path + '/', '')

	with open(json_path, 'w') as f:
		f.write(json_fixed)