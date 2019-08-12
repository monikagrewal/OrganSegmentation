import json
import os
import glob
from pathlib import Path
import pydicom
import numpy as np
import pickle
# from skimage.io import imread, imsave
# import skimage
from collections import Counter
from itertools import zip_longest
import matplotlib.pyplot as plt
import shutil


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def process_rtstruct(rtstruct, dicom_meta, include, exclude):
    all_results = {}
    uid_to_meta = dict([(meta['uid'], meta) for meta in dicom_meta])
    for label_idx, roi in enumerate(rtstruct.ROIContourSequence):
        label_name = rtstruct.StructureSetROISequence[label_idx].ROIName
        in_include = any([x for x in include if x in label_name])
        in_exclude = any([x for x in exclude if x in label_name])
        if (not in_include) or in_exclude:
            continue

        category_results = {}    
        found_results = False
        for cont in roi.ContourSequence:
            if cont.ContourGeometricType == 'POINT':
                break
            elif cont.ContourGeometricType != 'CLOSED_PLANAR':
                print("Unexpected geometric type: ", cont.ContourGeometricType)
            uid = cont.ContourImageSequence[0].ReferencedSOPInstanceUID
            meta = uid_to_meta.get(uid)
            if meta is None:
                print(f"Could not find processed dicom file with uid {uid}. Skipping whole sequence")
                break
            # series_id = str(Path(meta['rel_path']).parent)
            # metas_in_series = [meta for meta in dicom_meta if meta['rel_path'].startswith(f'{series_id}/')]
            # slice_loc_to_idx = dict(zip(sorted([meta['SliceLocation'] for meta in metas_in_series]), range(len(metas_in_series))))            
            # slice_loc = meta.get('SliceLocation')
            # slice_idx = slice_loc_to_idx[slice_loc]
    #         print(slice_loc, slice_idx)
            coords = np.array(list(grouper(cont.ContourData, 3)))
            coords_pix = coords - np.array(meta['origin'])
            coords_pix = coords_pix[:,0:2] / np.array(meta['PixelSpacing'])
            uid_results = category_results.get(uid, [])
            uid_results.append(coords_pix.tolist())
            if uid in category_results:
                print(f"Warning: multiple contour sequences for same uid: {uid}")
            category_results[uid] = uid_results
            found_results = True

        if found_results:
            all_results[label_name] = category_results
        
    return all_results



if __name__ == '__main__':
    # base_input_dir = '/export/scratch3/grewal/Data/MODIR_data'
    base_input_dir = '/export/scratch3/grewal/Data/MODIR_data_train_split'
    # base_input_dir = '/export/scratch3/bvdp/segmentation/data/AMC_raw/'
    # base_dicom_dir = '/export/scratch3/bvdp/segmentation/data/AMC/'
    base_dicom_dir = '/export/scratch3/bvdp/segmentation/data/AMC_dicom_train/'    
    # study_dir = '961714545_2645599973/20170725'

    input_path = Path(base_input_dir)
    dicom_path = Path(base_dicom_dir)
    base_output_path = Path('/export/scratch3/bvdp/segmentation/data/AMC_dataset_clean_train/')

    include = ['rectum', 'hip', 'bowel', 'bladder', 'sigmoid', 'spinal', 'anal_canal', 'anal canal', 'blaas']
    exclude = ['ctv','ptv','gtv', 'hippo']

    counter = Counter()
    n_valid_structs = 0
    for i, pp in enumerate(dicom_path.glob('*/*')):
        # Temporary hack to skip some folders
        # if i < 3:
            # continue

        study_path = pp.relative_to(dicom_path)
        print(study_path)
        # if (base_output_path / study_path).exists():
            # print(f'Skipping {study_path}')
            # continue
        with open(str(dicom_path / study_path / "dicom_meta.json"), "r") as f:
            dicom_meta = json.loads(f.read())

        rtstruct_path = input_path / study_path / '1'
        print(rtstruct_path)
        # find rtstruct dicom file
        rtstruct_files = list(rtstruct_path.glob('*dcm'))
        if len(rtstruct_files) < 1:
            print("No rtstruct files found. Doing nothing")
        elif len(rtstruct_files) > 10:
            # seems to happen in 1 case, where actual image data is stored with series id "1" just like rtstruct files
            print("Too many rtstruct files found. Something wrong")
            continue
    #     elif len(rtstruct_files) > 1:
    #         print("More than 1 rtstruct file found. Finding one with selected orientation and modality")
    #         print(len(rtstruct_files))
        for rtstruct_file in rtstruct_files:
            rtstruct = pydicom.read_file(str(rtstruct_file))
            if rtstruct.Modality != 'RTSTRUCT':
                continue
            try:
                first_uid = rtstruct.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID
            except Exception as e:
                print(str(e))
                continue
            matching_dicoms = [meta for meta in dicom_meta if meta['uid'] == first_uid]
            if len(matching_dicoms) > 0:
                # print("Valid dicoms found matching rtstruct data. Using this rtstruct file")
                break
        else:
            print("no valid rtstruct file found")
            continue
        n_valid_structs += 1

        uid_to_meta = dict([(meta['uid'], meta) for meta in dicom_meta])

        # all_uids = [cont_seq.ContourImageSequence[0].ReferencedSOPInstanceUID for roi_seq in rtstruct.ROIContourSequence for cont_seq in roi_seq.ContourSequence]
        all_uids = []
        for roi_seq in rtstruct.ROIContourSequence:
            for cont_seq in roi_seq.get('ContourSequence', []):
                if cont_seq.ContourGeometricType == 'POINT':
                    break
                elif cont_seq.ContourGeometricType != 'CLOSED_PLANAR':
                    print("Unexpected geometric type: ", cont_seq.ContourGeometricType)
                all_uids.append(cont_seq.ContourImageSequence[0].ReferencedSOPInstanceUID)

        num_missing = len([uid for uid in all_uids if not uid in uid_to_meta])
        if num_missing > 0:
            print("Not all dicom images that are referenced in rstruct are present in processed dicom dataset. Skipping study")
            continue
        series_folders = set([str(Path(uid_to_meta[uid]['rel_path']).parent) for uid in all_uids if uid in uid_to_meta])
        if len(series_folders) > 1:
            print(f'RTSTRUCT references multiple series: {num_series_folders}. Skipping!')
            continue

        try:
            results = process_rtstruct(rtstruct, dicom_meta, include=include, exclude=exclude)
            if len(results) == 0:
                print(f'No annotation results for study {study_path}')
                continue
            series_id = list(series_folders)[0]
            dicom_meta_series = [meta for meta in dicom_meta if meta['rel_path'].startswith(f'{series_id}/')]
            # Maybe add a check here that the slices are all present (no gaps)

            output_path = base_output_path / study_path
            output_path.mkdir(exist_ok=True, parents=True)
            output_path_images = output_path / f'{series_id }' 
            output_path_annot = output_path / 'annotations.json'
            output_path_meta = output_path / 'meta.json'

            with open(output_path_annot, 'w') as f:
                f.write(json.dumps(results))
            with open(output_path_meta, 'w') as f:
                f.write(json.dumps(dicom_meta_series))

            shutil.copytree(base_dicom_dir / study_path / f'{series_id }', output_path_images)

            # with open('test.pickle', 'wb') as f:
                # pickle.dump(results, f)

        except Exception as e:
            # consider whole study failed 
            print(str(e))