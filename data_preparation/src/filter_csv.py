import os
import json
import pandas as pd

root_path = "/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/"
label_mapping_path = '/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/label_mapping_train.json'
meta_path = '/export/scratch3//grewal/OAR_segmentation/data_preparation/meta/dataset_train.csv'

meta_df = pd.read_csv(meta_path)
label_mapping = json.load(open(label_mapping_path, 'r'))
classes = list(sorted(label_mapping.keys()))

for classname in classes:
	mini_df = meta_df.loc[meta_df[classname] > 0]
	print("total samples = {}".format(len(mini_df)))
	print(mini_df.head())
	mini_df.to_csv(os.path.join(root_path, "{}.csv".format(classname)))

# mini_df = meta_df.query("bladder > 0 and bowel_bag > 0 and hip > 0 and rectum > 0")
# print("total samples = {}".format(len(mini_df)))
# print(mini_df.head())
# mini_df.to_csv(os.path.join(root_path, "{}.csv".format("bowel_bag_bladder_hip_rectum")))

