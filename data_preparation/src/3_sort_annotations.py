import glob
from pathlib import Path
import json
from tqdm.auto import tqdm as tqdm

def sort_annotations(annotations, label_mapping):
	special_entries = []

	for entry in annotations:
		raw_label = entry["label_name"]
		if label_mapping.get(raw_label, "") == "bowel_bag":
			special_entries.append(entry)

	for entry in special_entries:
		annotations.remove(entry)
	sorted_annotations = special_entries + annotations

	return sorted_annotations

# root_dir1 = Path('/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/')
# root_dir2 = Path('/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_sigmoid/')

# label_mapping_path = '/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/label_mapping_train.json'

root_dir = Path('/export/scratch3/bvdp/segmentation/data/MODIR_data_train_2019-12-16/')
label_mapping_path = '/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/label_mapping_train_2019-12-16.json'

with open(label_mapping_path) as f:
	label_mapping = json.load(f)

inverse_label_mapping = {value:key for key, values in label_mapping.items() for value in values}


# paths = list(root_dir1.glob('**/annotations.json')) + list(root_dir2.glob('**/annotations.json'))
paths = list(root_dir.glob('**/annotations.json'))
paths = [path.parent for path in paths]
print ("total data: {}".format(len(paths)))
for path in tqdm(paths):
	with open(str(path / 'annotations.json')) as f:
		annotations = json.loads(f.read())
	annotations = sort_annotations(annotations, inverse_label_mapping)
	with open(str(path / 'annotations.json'), "w") as f:
		json.dump(annotations, f)
