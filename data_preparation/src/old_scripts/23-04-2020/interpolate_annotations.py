import os
from pathlib import Path
import json

root_dir = Path('/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/')

# for path in root_dir.glob('**/annotations.json'):
# 	meta_path = path.parent / 'meta.json'
# 	print(path, meta_path)

# label_mapping_path = '/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/label_mapping_train.json'
# with open(label_mapping_path) as f:
# 	label_mapping = json.load(f)

# inverse_label_mapping = {value:key for key, values in label_mapping.items() for value in values}

annotation_path = Path('/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/4059790317_1855711096/20171010/annotations.json')
meta_path = Path('/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/4059790317_1855711096/20171010/meta.json')

with open(meta_path) as f:
	meta_list = json.loads(f.read())

# print(meta_list[0])
meta_list_sorted = sorted(meta_list, key=lambda x: x['SliceLocation'])
uid_to_slicelocation = {meta['uid']: (meta['SliceLocation']) for meta in meta_list}
slice_thickness = meta_list[0]['SliceThickness']
# print(uid_to_slicelocation)
# print(sorted([v for k,v in uid_to_slicelocation.items()]))
# for meta, meta_next in zip(meta_list_sorted, meta_list_sorted[1:]):
# 	slice_gap = meta_next['SliceLocation'] - meta['SliceLocation']
# 	if slice_gap > slice_thickness:
# 		print(slice_gap)


uid_to_labels = {}

label_to_annots = {}
with open(annotation_path) as f:
	annotations = json.loads(f.read())
	for annotation_dict in annotations:
		uid = annotation_dict['uid']
		label_name = annotation_dict['label_name']
		# mapped_label_name = inverse_label_mapping.get(annotation_dict['label_name'])
		# if mapped_label_name is not None:
			# uid_to_labels[uid] = list(set(uid_to_labels.get(uid, []) + [mapped_label_name]))
		annotation_dict['slice_location'] = uid_to_slicelocation[uid]
		label_to_annots[label_name] = label_to_annots.get(label_name, []) + [annotation_dict]
		# mapped_label_to_annots[mapped_label_name] = mapped_label_to_annots.get(mapped_label_name, []) + [annotation_dict]
		# print(uid, label_name)

		# print(uid)

# print(uid_to_labels)
for label_name, annots in label_to_annots.items():
	print(label_name)
	annots_sorted = sorted(annots, key=lambda x: x['slice_location'])
	for annot_prev, annot, annot_next in zip(annots_sorted, annots_sorted[1:], annots_sorted[2:]):
		if annot['slice_location'] - annot_prev['slice_location'] != slice_thickness:
			print(annot_prev['slice_location'], annot['slice_location'], annot_next['slice_location'])
			print(annot_prev['uid'], annot['uid'])


# to_interpolate = []
# for i, meta in enumerate(meta_list_sorted[1:-1]):
# 	uid = meta['uid'] 
# 	uid_prev = meta_list_sorted[i-1]['uid']
# 	uid_next = meta_list_sorted[i-1]['uid']
# 	for cl in interp_classes:
# 		if not cl in uid_to_labels.get(uid, []):
# 			if cl in uid_to_labels.get(uid_prev, []) and cl in uid_to_labels.get(uid_next, []):
# 				to_interpolate.append((cl, uid, uid_prev, uid_next))
# print(to_interpolate)



# with_slice_location = {}
# for k,v in mapped_label_to_annots.items():
# 	with_slice_location[k] = [(el, uid_to_slicelocation[el['uid']]) for el in v]

# print([v[1] for v in sorted(with_slice_location['bowel_bag'], key=lambda x: x[1])])



# print(annotations[1])

# study_path = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/477762856_724776677/20170102'

# os.listdir(study_path)

# (Path(study_path) / 'annotations.json').exists() and (Path(study_path) / 'meta.json').exists()