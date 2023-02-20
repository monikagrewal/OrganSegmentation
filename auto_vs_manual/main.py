import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import autocontouring
sys.path[0] = str(Path(sys.path[0]).parent)
from experiments.utils.metrics import calculate_metrics

import pdb


def process_data(patient_id, patient_path, output_path):
	class_names = ["background", "bowel_bag", "bladder", "hip", "rectum"]
	# read dicoms and obs1, obs2 delineations
	image = np.load(os.path.join(patient_path, "image.npz"))["volume"]
	seg_obs1 = np.load(os.path.join(patient_path, "observer1.npz"))["mask_volume"]
	seg_obs2 = np.load(os.path.join(patient_path, "observer2.npz"))["mask_volume"]

	# obtain segmentation mask from deep learning model
	seg_auto = autocontouring.main(image)

	# vizualization

	# compute metrics obs1 and auto, obs2 and auto, obs1 and obs2
	test_results = []
	metrics12 = calculate_metrics(seg_obs1, seg_obs2, class_names=class_names)
	print("metrics12: ", metrics12)
	dice = metrics12[3]
	for class_no, classname in enumerate(class_names):
		test_results.append({
			"patient_id": patient_id,
			"classname": classname,
			"observer_pair": "d12",
			"dice": dice[class_no]
		}
		)

	metrics1auto = calculate_metrics(seg_obs1, seg_auto, class_names=class_names)
	print("metrics1auto: ", metrics1auto)
	dice = metrics1auto[3]
	for class_no, classname in enumerate(class_names):
		test_results.append({
			"patient_id": patient_id,
			"classname": classname,
			"observer_pair": "d1auto",
			"dice": dice[class_no]
		}
		)

	metrics2auto = calculate_metrics(seg_obs2, seg_auto, class_names=class_names)
	print("metrics2auto: ", metrics2auto)
	dice = metrics2auto[3]
	for class_no, classname in enumerate(class_names):
		test_results.append({
			"patient_id": patient_id,
			"classname": classname,
			"observer_pair": "d2auto",
			"dice": dice[class_no]
		}
		)

	return test_results


if __name__=="__main__":
	root_path = "../outputs/auto_vs_manual_data"
	output_path = "../outputs/auto_vs_manual_results"
	
	dicom_patient_ids = [1, 3, 6, 2, 4, 5, 7]
	# dicom_patient_ids = [3]
	all_results = []
	for patient_id in dicom_patient_ids:
		print("Patient no: ", patient_id)
		patient_path = os.path.join(root_path, "{0:03d}_MODIR_CERVIX_PATIENT/{0:03d}_CT_EBRT".format(patient_id))
		try:
			results = process_data(patient_id, patient_path, output_path)
			all_results.extend(results)
		except Exception as e:
			print(e)
			continue
		print("")
	
	df = pd.DataFrame.from_records(all_results)
	print(df.head())
	os.makedirs(output_path, exist_ok=True)
	df.to_csv(os.path.join(output_path, "auto_vs_manual_results.csv"))