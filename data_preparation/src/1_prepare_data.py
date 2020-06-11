import json
import os
import glob
from pathlib import Path
import pydicom
import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage.transform import resize
from scipy.ndimage import zoom, interpolation
import pickle
from collections import Counter
from itertools import zip_longest
import shutil
import sys
import label_mapping



def load_dicom(im):
    try:
        arr = im.pixel_array
    except Exception as e:
        print(f"Exception: {e}\n")
        return None

    if arr.max() == arr.min():
        print("image is blank")
        return None
    
    return arr


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
                # print(f"excluding annotation with label: {label_name}")
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

def process_volume(volume, metadata, desired_spacing=[0.976562, 0.976562], desired_slice_thickness=2.5):

    intercept = float(metadata["RescaleIntercept"])
    slope = float(metadata["RescaleSlope"])
    if isinstance(metadata["WindowWidth"], list):
        ww = float(metadata["WindowWidth"][0])
        wl = float(metadata["WindowCenter"][0])
    else:
        ww = float(metadata["WindowWidth"])
        wl = float(metadata["WindowCenter"])
    # ww = float(metadata["WindowWidth"])
    # wl = float(metadata["WindowCenter"])
    arr = rescale_intensity(volume, intercept, slope)
    arr = apply_ww_wl(arr, ww, wl)
    arr = normalize_array(arr)
    if metadata['PatientPosition'] != "HFS":
        arr = arr[:, ::-1, ::-1]
        
    spacing_inplane = [float(metadata['PixelSpacing'][0]), float(metadata['PixelSpacing'][1])]
    slice_thickness = float(metadata['SliceThickness'])
    zoom_factor_inplane = [spacing_inplane[0] / desired_spacing[0], spacing_inplane[1] / desired_spacing[1]]
    zoom_factor_slice = slice_thickness / desired_slice_thickness
    zoom_factor = [zoom_factor_slice] + zoom_factor_inplane
    resampled = zoom(arr, zoom_factor, order=1)


    return resampled

def process_annotations(annotations, sorted_metadata_list, target_shape, desired_spacing=[0.976562, 0.976562], 
                        desired_slice_thickness=2.5, classes=['background', 'bowel_bag', 'bladder', 'hip', 'rectum']):        
    
    # order in which to layer class annotations in case they overlap
    class_layering = ['background', 'bowel_bag', 'bladder', 'hip', 'rectum']

    uid_to_slice_idx = dict([(meta['uid'], i) for i, meta in enumerate(sorted_metadata_list)])    
        
    meta = sorted_metadata_list[0]
    patientposition = meta['PatientPosition']
    origin = meta['origin'][:2]
    pixelspacing = np.array(meta['PixelSpacing'][:2])
    slice_thickness = meta['SliceThickness']
    new_annotations = []
    
    mask_volume = np.zeros((len(sorted_metadata_list), target_shape[1], target_shape[2]), dtype=np.int)  
    class2idx = dict(zip(classes, range(len(classes))))    
    
    class2layeridx = dict(zip(class_layering, range(len(classes))))
    class_layer_indici = np.array([class2layeridx[class_name] for class_name in classes])


    slice_label_counts = {}
    for item in sorted(annotations, key=lambda x: uid_to_slice_idx.get(x['uid'], -1)):        
        uid = item["uid"]
        slice_idx = uid_to_slice_idx.get(uid, None)
        if slice_idx is None:
            continue

        coords = item["coords"]

        if patientposition=="HFP":
            orientation = np.array([-1, -1])
            coords_pix = orientation*np.array(coords) - orientation*origin  
            coords_pix = coords_pix / pixelspacing
            coords_pix = meta['npixels'] - coords_pix
            zoom_factor = pixelspacing / desired_spacing
            coords_pix = coords_pix * zoom_factor
        else:
            coords_pix = np.array(coords) - origin
            coords_pix = coords_pix / pixelspacing
            zoom_factor = pixelspacing / desired_spacing
            coords_pix = coords_pix * zoom_factor


        label = item["label_name"]
        label_mapped = label_mapping.map_label(label)
        if label_mapped is None:
            continue
        label_idx = class2idx.get(label_mapped)
        if label_idx is None:
            continue

            
        coords_np = coords_pix
        rr, cc = skimage.draw.polygon(coords_np[:,0], coords_np[:,1], shape=(target_shape[1], target_shape[2]))

        # determine whether to overwrite existing annotation label based on predefined ordering 
        # (for example bladder takes precedence bowel bag)
        overwrite_mask = class_layer_indici[mask_volume[slice_idx, cc,rr]] < class_layer_indici[label_idx]        
        rr, cc = rr[overwrite_mask], cc[overwrite_mask]

        mask_volume[slice_idx, cc, rr] = label_idx
        label_counts = slice_label_counts.get(slice_idx, {label_mapped: 0})
        label_counts[label_mapped] = label_counts.get(label_mapped, 0) + 1
        slice_label_counts[slice_idx] = label_counts
    zoom_factor_slice = slice_thickness / desired_slice_thickness
    # only resample slice dimension because we already handle the other resampling ourselves by
    # transforming the coordinates
    resampled = zoom(mask_volume, [zoom_factor_slice, 1, 1], order=0)
    
    encountered_classes = list(set([cat for count_dict in slice_label_counts.values() for cat in count_dict.keys()]))
    
    return resampled, encountered_classes
    


def match_dicoms_and_annotation(dicom_metadata, annotations):
    series_info = {}
    for _, annotation in annotations.items():
        annot_uids = [item["uid"] for item in annotation]
        for series_id, metadata_list in dicom_metadata.items():
            dicom_uids = [meta["uid"] for meta in metadata_list]
            matching_uids = [meta["uid"] for meta in metadata_list if meta["uid"] in annot_uids]
            if len(matching_uids) > 1:
                series_info[series_id] = (metadata_list, annotation)

    return series_info




def process_dicoms(input_directory, output_directory=None, orientation="Transverse", 
                   modality="CT", desired_spacing=[0.976562, 0.976562], desired_slice_thickness=2.5,
                   slice_cutoff=(0,140)):
    """
    args:
      input_directory: path to study date directory
    """
    
    root_dir  = Path(input_directory)
    output_dir  = Path(output_directory)
    dicom_metadata = {}
    dicom_imagedata = {}
    annotations = {}
    for i, pp in enumerate(root_dir.glob('*/*.dcm')):
    # for i, pp in enumerate(root_dir.glob('**/*.dcm')):
        if not pp.is_file():
            continue        
        im = pydicom.dcmread(str(pp))
        metadata, status = extract_info(im)
        if metadata['Modality'] == 'RTSTRUCT':
            annotations[pp] = process_rtstruct(im)
            continue

        if status and metadata["Modality"] == modality and metadata["orientation"] == orientation:
            metadata = convert_dtypes(metadata)
            series_id = metadata["SeriesInstanceUID"]
            arr = load_dicom(im)
            if arr is None:
                continue            
            
            metadata['npixels'] = arr.shape
            
            series_results = dicom_metadata.get(series_id, [])
            series_results.append(metadata)
            dicom_metadata[series_id] = series_results
            
            series_images = dicom_imagedata.get(series_id, [])
            series_images.append(arr)
            dicom_imagedata[series_id] = series_images
    
    series_info = match_dicoms_and_annotation(dicom_metadata, annotations)
    
    # take first one assuming in most cases there is only one match per study. Maybe extend to all matches?
    
    if len(series_info) == 0:
        print("No matches between dicoms and annotations")
        return None
    series_id = list(series_info.keys())[0]
    print(series_id)
    serie_info = series_info[series_id]
    metadata_list, annotations = serie_info
    metadata = metadata_list[0]
    sorted_indici = sorted(zip(range(len(metadata_list)), metadata_list), key=lambda x: x[1]['SliceLocation'])
    sorted_indici = [idx for idx, metadata in sorted_indici]
    sorted_images = [dicom_imagedata[series_id][i] for i in sorted_indici]
    volume = np.stack(sorted_images)
    volume = process_volume(
        volume, metadata, desired_slice_thickness=desired_slice_thickness, desired_spacing=desired_spacing)
    
    sorted_metadata_list = [metadata_list[i] for i in sorted_indici]
    mask_volume, encountered_classes = process_annotations(
        annotations, sorted_metadata_list, target_shape=volume.shape, 
        desired_slice_thickness=desired_slice_thickness, desired_spacing=desired_spacing)
    
    if slice_cutoff is not None:
    	start_slice, end_slice = slice_cutoff
    	volume = volume[start_slice: end_slice+1]
    	mask_volume = mask_volume[start_slice: end_slice+1]

    output_dir.mkdir(exist_ok=True, parents=True)
    
    
    np.savez_compressed(str(output_dir / f'{series_id}.npz'), volume=volume, mask_volume=mask_volume)
    
#     return volume, mask_volume, annotations
    output_keys = [
        'SeriesInstanceUID', 'orientation', 'origin', 'PixelSpacing', 'SliceThickness', 'Modality', 
        'RescaleIntercept', 'RescaleSlope', 'PatientPosition', 'WindowWidth', 'WindowCenter', 'npixels'
    ]
    metadata_filtered = {k:v for k,v in metadata.items() if k in output_keys}
    meta_result = {
        **metadata_filtered, 'labels': encountered_classes, 'input_directory': input_directory, 
        'output_directory': output_directory, 'desired_pixel_spacing': desired_spacing, 
        'desired_slice_thickness': desired_slice_thickness}
    with open(str(output_dir / f'{series_id}.json'), 'w') as meta_output:
        meta_output.write(json.dumps(meta_result))
    
    return meta_result


if __name__ == '__main__':
    
    root_path = '/export/scratch2/grewal/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split'
    output_path = '/export/scratch3/bvdp/segmentation/data/MODIR_data_preprocessed_train_10-06-2020'

    root_dir = Path(root_path)
    output_dir = Path(output_path)

    dcm_paths = root_dir.glob('**/*dcm')
    dcm_base_folders = list(set([dcm_p.parent.parent for dcm_p in dcm_paths]))
    print("Number of folders: ", len(dcm_base_folders))
    
    recompute = False

    for i, pp in enumerate(dcm_base_folders):
        print(f'{i} out of {len(dcm_base_folders)}')
        dicom_path = str(output_path / pp.relative_to(root_path))
        print(dicom_path)    
        sys.stdout.flush()
        if not recompute and len(list(Path(dicom_path).glob('*.json'))) > 0:
            print(f"Already processed. Skipping {dicom_path}..")
            continue
        results = process_dicoms(str(pp), output_directory=dicom_path)
        sys.stdout.flush()
        


