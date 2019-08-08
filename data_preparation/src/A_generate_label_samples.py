import json
import os
import glob
from pathlib import Path
import pydicom
import numpy as np
# from skimage.io import imread, imsave
import skimage
from collections import Counter
from itertools import zip_longest
import matplotlib.pyplot as plt


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def generate_label_samples(rtstruct, dicom_meta, label_output_path, include, exclude):
    try:
        slice_loc_to_idx = dict(zip(sorted([meta['SliceLocation'] for meta in dicom_meta]), range(len(dicom_meta))))
    except Exception as e: 
        print("Problem sorting slices for output: {}. Exception: \n{}".format(label_output_path, str(e)))
        return
    uid_to_meta = dict([(meta['uid'], meta) for meta in dicom_meta])
    for label, roi_seq in enumerate(rtstruct.ROIContourSequence):
        label_name = rtstruct.StructureSetROISequence[label].ROIName
        in_include = any([x for x in include if x in label_name])
        in_exclude = any([x for x in exclude if x in label_name])
        if (not in_include) or in_exclude:
            continue

        print(label_name)
        for cont_seq in roi_seq.ContourSequence:
            if cont_seq.ContourGeometricType == 'POINT':
                break
            elif cont_seq.ContourGeometricType != 'CLOSED_PLANAR':
                print("Unexpected geometric type: ", cont_seq.ContourGeometricType)
            uid = cont_seq.ContourImageSequence[0].ReferencedSOPInstanceUID
            meta = uid_to_meta.get(uid)
            if meta is None:
                print(f"Could not find processed dicom file with uid {uid}. Skipping")
                continue
            slice_loc = meta.get('SliceLocation')
            slice_idx = slice_loc_to_idx[slice_loc]
    #         print(slice_loc, slice_idx)
            coords = np.array(list(grouper(cont_seq.ContourData, 3)))
            coords_pix = coords - np.array(meta['origin'])
            coords_pix = coords_pix[:,0:2] / np.array(meta['PixelSpacing']) 
            rel_path= Path(meta['rel_path'])
            img_path = dicom_path / study_path / rel_path.with_suffix('.jpg')
            output_path = label_output_path / rel_path.parent / label_name /(f"{slice_idx}_" + rel_path.with_suffix(".jpg").name)
    #         print(output_path)
            img = skimage.io.imread(str(img_path))
            rr, cc = skimage.draw.polygon(coords_pix[:,0], coords_pix[:,1], shape=img.shape)
            # combined = img.copy()
            combined = np.stack((img,)*3, axis=-1)
            opacity = 0.75
            combined[cc, rr] = opacity*np.array([255,0,0]) + (1-opacity)*combined[cc, rr]
            combined = np.concatenate((combined, np.stack((img,)*3, axis=-1)), axis=1)

            # fig, axes = plt.subplots(1,2, figsize=(20,10))
            # axes[0].imshow(img, cmap='gray')
            # axes[1].imshow(combined, cmap='gray')
            # axes[1].scatter(coords_pix[:,0], coords_pix[:,1], color='red', s=3)
            # for ax in axes:
            #     ax.axis('off')
            output_path.parent.mkdir(exist_ok=True, parents=True)
            skimage.io.imsave(output_path, combined, plugin='imageio')
            # fig.savefig(output_path, bbox_inches='tight', dpi=100)
            # plt.close(fig)


if __name__ == '__main__':
    base_input_dir = '/export/scratch3/grewal/Data/MODIR_data'
    base_dicom_dir = '/export/scratch3/bvdp/segmentation/data/AMC/'
    # study_dir = '961714545_2645599973/20170725'

    input_path = Path(base_input_dir)
    dicom_path = Path(base_dicom_dir)
    base_output_path = Path('/export/scratch3/bvdp/segmentation/data/AMC_label_samples_v2/')

    include = ['rectum', 'hip', 'bowel', 'bladder', 'sigmoid', 'spinal', 'anal_canal', 'anal canal', 'blaas']
    exclude = ['ctv','ptv','gtv']

    counter = Counter()
    n_valid_structs = 0
    for i, pp in enumerate(dicom_path.glob('*/*')):
        # Temporary hack to skip some folders
        # if i < 3:
            # continue

        study_path = pp.relative_to(dicom_path)
        print(study_path)
        if (base_output_path / study_path).exists():
            print(f'Skipping {study_path}')
            continue
        with open(str(dicom_path / study_path / "dicom_meta.json"), "r") as f:
            dicom_meta = json.loads(f.read())

        rtstruct_path = input_path / study_path / '1'
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
        labels = [x.ROIName.lower() for x in rtstruct.StructureSetROISequence]
        counter.update(labels)
        
        generate_label_samples(rtstruct, dicom_meta, base_output_path / study_path, include=include, exclude=exclude)


