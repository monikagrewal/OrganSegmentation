import json
import os
import glob
from pathlib import Path
import pydicom
import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage.transform import resize
import pickle
from collections import Counter
from itertools import zip_longest
import shutil


def visualize_label(meta_list, annotations, label_pp):
	colors = {0: (1, 0, 0), 1: (1, 0, 1), 2: (0, 1, 0), 3: (0, 0, 1),
				4: (1, 1, 0), 5: (0, 1, 1), 6: (1, 0, 1),
				7: (1, 0.5, 0), 8: (0, 1, 0.5), 9: (0.5, 0, 1),
				10: (0.5, 1, 0), 11: (0, 0.5, 1), 12: (1, 0, 0.5)}
	meta_sorted = sorted(meta_list, key=lambda x: x['SliceLocation'])
	for i, meta in enumerate(meta_sorted):
		img_path = meta['output_path']
		img = imread(img_path, as_gray=True) / 255.0
		combined = np.stack((img,)*3, axis=-1)
		opacity = 0.75
		uid = meta['uid']
		annotation = [item for item in annotations if item["uid"]==uid]
		for idx, item in enumerate(annotation):
			coords = item["coords"]
			coords = np.asarray(coords).squeeze()
			rr, cc = skimage.draw.polygon(coords[:,0], coords[:,1], shape=img.shape)
			combined[cc, rr] = opacity*np.array(colors[int(idx%len(colors))]) + (1-opacity)*combined[cc, rr]

		combined = np.concatenate((combined, np.stack((img,)*3, axis=-1)), axis=1)

		output_path = str(label_pp) + f"/{i}.jpg"
		imsave(output_path, (combined * 255).astype(np.uint8))



def extract_info(im):
	info = dict.fromkeys(["SeriesInstanceUID", "uid", "orientation",
			"origin", "SliceLocation", "PixelSpacing",
			 "SliceThickness", "Modality", "RescaleIntercept", "RescaleSlope", "PatientPosition",
			 "WindowWidth", "WindowCenter"], None)

	for attribute in im.dir():
		if attribute=="ImageOrientationPatient":
			IOP = im.ImageOrientationPatient
			orientation = get_file_plane(IOP)
			info["orientation"] = orientation

		if attribute=="ImagePositionPatient":
			origin = im.ImagePositionPatient
			origin = [float(item) for item in origin]
			info["origin"] = origin

		if attribute=="SliceLocation":
			info["SliceLocation"] = float(im.SliceLocation)

		if attribute=="SOPInstanceUID":
			info["uid"] = im.SOPInstanceUID

		if attribute in ["SeriesInstanceUID", "PixelSpacing", "SliceThickness", "Modality",
			 "RescaleIntercept", "RescaleSlope", 
			 "PatientPosition", "WindowWidth", "WindowCenter"]:
			info[attribute] = eval('im.' + attribute)

	status = True
	for _, val in info.items():
		if val is None:
			status = False
			break

	return info, status

def get_file_plane(IOP):
	"""
	This function takes IOP of an image and returns its plane (Sagittal, Coronal, Transverse)

	Usage:
	a = pydicom.read_file(filepath)
	IOP = a.ImageOrientationPatient
	plane = file_plane(IOP)

	"""
	IOP_round = [round(x) for x in IOP]
	plane = np.cross(IOP_round[0:3], IOP_round[3:6])
	plane = [abs(x) for x in plane]
	if plane[0] == 1:
		return "Sagittal"
	elif plane[1] == 1:
		return "Coronal"
	elif plane[2] == 1:
		return "Transverse"

def rescale_intensity(image, intercept, slope):
	if intercept is None or slope is None:
		return image

	# Convert to Hounsfield units (HU)
	image = np.float16(slope) * image.astype(np.float16) + np.float16(intercept)
	image = image.astype(np.int16)

	return image

def apply_ww_wl(image, ww, wl):
	ub = wl + ww//2
	lb = wl - ww//2
#     print(f"Upper bound: {ub}\nLower bound: {lb}")
	image[image > ub] = ub
	image[image < lb] = lb
	image = (image - lb) / float(ub - lb)
	return image

def normalize_array(image):
	image = (image - np.min(image)) / float(np.max(image) - np.min(image))
	return image


def convert_dtypes(metadata):
	dtype_mapping = {
		'MultiValue': list,
		'DSfloat': float
	}
	for k,v in metadata.items():
		dtype = v.__class__.__name__
		if dtype in dtype_mapping:
			metadata[k] = dtype_mapping[dtype](metadata[k])
	return metadata


def process_annotation(item, meta):
	meta = meta[0]
	patientposition = meta['PatientPosition']
	origin = meta['origin'][:2]
	pixelspacing = np.array(meta['PixelSpacing'][:2])

	coords = item["coords"]
	if patientposition=="HFP":
		orientation = np.array([-1, -1])
		coords_pix = orientation*np.array(coords) - orientation*origin	
		coords_pix = coords_pix / pixelspacing
		coords_pix = meta['npixels'] - coords_pix
	else:
		coords_pix = np.array(coords) - origin
		coords_pix = coords_pix / pixelspacing

	item["coords"] = coords_pix.tolist()

	return item


def grouper(iterable, n, fillvalue=None):
	"Collect data into fixed-length chunks or blocks"
	# grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
	args = [iter(iterable)] * n
	return zip_longest(fillvalue=fillvalue, *args)


def process_rtstruct(rtstruct):
	annotation = []
	include = ['rectum', 'hip', 'bowel', 'bladder', 'sigmoid', 'spinal', 'anal_canal', 'anal canal', 'blaas']
	exclude = ['ctv','ptv','gtv', 'hippo']
	try:
		first_uid = rtstruct.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID
	except Exception as e:
		print(str(e))
		return annotation
	
	for label_idx, roi in enumerate(rtstruct.ROIContourSequence):
		try:
			label_name = rtstruct.StructureSetROISequence[label_idx].ROIName.lower()

			in_include = any([x for x in include if x in label_name])
			in_exclude = any([x for x in exclude if x in label_name])
			if (not in_include) or in_exclude:
				print(f"excluding annotation with label: {label_name}")
				continue

			for cont in roi.ContourSequence:
				if cont.ContourGeometricType == 'POINT':
					print("Annotation is a point, expected ContourSequence")
					break
				elif cont.ContourGeometricType != 'CLOSED_PLANAR':
					print("Unexpected geometric type: ", cont.ContourGeometricType)
				uid = cont.ContourImageSequence[0].ReferencedSOPInstanceUID	
				coords = np.array(list(grouper(cont.ContourData, 3)))[:, 0:2]
				entry = {"uid": uid,
						"label_name": label_name,
						"coords": coords.tolist()}
				annotation.append(entry)
		except Exception as e:
			print(str(e))
			break

	return annotation


def match_dicoms_and_annotation(dicom_metadata, annotations):
	# import pdb; pdb.set_trace()
	series_info = {}
	for _, annotation in annotations.items():
		annot_uids = [item["uid"] for item in annotation]
		for series_id, metadata_list in dicom_metadata.items():
			dicom_uids = [meta["uid"] for meta in metadata_list]
			matching_uids = [meta["uid"] for meta in metadata_list if meta["uid"] in annot_uids]
			if len(matching_uids) > 1:
				annotation = list(map(lambda x: process_annotation(x, metadata_list), annotation))
				series_info[series_id] = (metadata_list, annotation)

	return series_info


def process_dicom_array(im, metadata):
	try:
		arr = im.pixel_array
	except Exception as e:
		print(f"Exception: {e}\n")
		return None

	if arr.dtype==np.uint16:
		print("The image data type is not readable for file: {}".format(str(pp)))
		return None

	if arr.max() == arr.min():
		print("image is blank")
		return None

	intercept = float(metadata["RescaleIntercept"])
	slope = float(metadata["RescaleSlope"])
	ww = float(metadata["WindowWidth"])
	wl = float(metadata["WindowCenter"])
	arr = rescale_intensity(arr, intercept, slope)
	arr = apply_ww_wl(arr, ww, wl)
	arr = normalize_array(arr)
	if im.PatientPosition != "HFS":
		arr = arr[::-1, ::-1]

	return arr


def process_dicoms(input_directory, output_directory=None, label_output_dir=None, orientation="Transverse", modality="CT"):
	"""
	args:
	  input_directory: path to study date directory
	"""
	
	root_dir  = Path(input_directory)
	output_dir  = Path(output_directory)
	dicom_metadata = {}
	annotations = {}
	for i, pp in enumerate(root_dir.glob('**/*.dcm')):
		if not pp.is_file():
			continue        
		im = pydicom.dcmread(str(pp))
		metadata, status = extract_info(im)
		if metadata['Modality'] == 'RTSTRUCT':
			annotations[pp] = process_rtstruct(im)
			continue

		if status and metadata["Modality"] == modality and metadata["orientation"] == orientation:
			arr = process_dicom_array(im, metadata)
			if arr is None:
				continue
			
			metadata['npixels'] = arr.shape

			pp_rel = pp.relative_to(root_dir)
			output_pp = (output_dir / pp_rel).with_suffix('.jpg')
			output_pp.parent.mkdir(exist_ok=True, parents=True)
			metadata['original_path'] = str(pp)
			metadata['rel_path'] = str(pp_rel)

			if output_directory is not None:
				imsave(str(output_pp), (arr * 255).astype(np.uint8))
				metadata['output_path'] = str(output_pp)
			
			metadata = convert_dtypes(metadata)
			series_id = metadata["SeriesInstanceUID"]
			series_results = dicom_metadata.get(series_id, [])
			series_results.append(metadata)
			dicom_metadata[series_id] = series_results
	
	series_info = match_dicoms_and_annotation(dicom_metadata, annotations)

	for series_id, (metadata_list, annotation) in series_info.items():
		# last match overwrites all preceding matches. assuming that most of the studies have only one match
		with open(str(output_dir / 'meta.json'), "w") as output_file:
			json.dump(metadata_list, output_file)

		with open(str(output_dir / 'annotations.json'), "w") as output_file:
			json.dump(annotation, output_file)	

		if label_output_dir is not None:
			label_pp = (label_output_dir / pp_rel.parent)
			label_pp.mkdir(exist_ok=True, parents=True)
			visualize_label(metadata_list, annotation, label_pp)
			
	return None


if __name__ == '__main__':
	root_path = '/export/scratch3/grewal/Data/MODIR_data_train_split/'
	output_path = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train/'
	label_output_path = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_dicom_train_labels/'
	# root_path = '/export/scratch3/grewal/Data/__Tijdelijk/'
	# output_path = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_sigmoid/'
	# label_output_path = '/export/scratch3/grewal/Data/segmentation_prepared_data/AMC_sigmoid_labels/'

	root_dir = Path(root_path)
	output_dir = Path(output_path)
	label_output_dir = Path(label_output_path)

	for i, pp in enumerate(root_dir.glob('*/*')):
		# if str(pp) != "/export/scratch3/grewal/Data/MODIR_data_train_split/1479952689_3596254403/20130909":
		# 	continue
		print(f"\nProcessing {i} : {pp}\n")
		# if i >= 1:
		# 	break

		dicom_path = str(output_path / pp.relative_to(root_path))
		dicom_label_path = str(label_output_path / pp.relative_to(root_path))
		process_dicoms(str(pp), output_directory=dicom_path, label_output_dir=dicom_label_path)


