import json
import os
import glob
from pathlib import Path
import pydicom
import numpy as np
from skimage.io import imread, imsave


def extract_info(im):
    info = dict.fromkeys(["orientation", "origin", "SliceLocation", "PixelSpacing",
             "SliceThickness", "Modality", "RescaleIntercept", "RescaleSlope",
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

        if attribute in ["PixelSpacing", "SliceThickness", "Modality", "RescaleIntercept", "RescaleSlope",
             "WindowWidth", "WindowCenter"]:
            info[attribute] = eval('im.' + attribute)

    return info

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

def process_dicoms(input_directory, output_directory=None, orientation="Transverse", modality="CT"):
    """
    args:
      input_directory: path to study date directory
    """
    
    root_dir  = Path(input_directory)
    output_dir  = Path(output_directory)
    metadata_list = []
    for i, pp in enumerate(root_dir.glob('**/*.dcm')):
        # print(i, end=',')
        pp_rel = pp.relative_to(root_dir)
        # ignore rtstruct directory
        if pp_rel.parts[0] == '1':
            continue
        if not pp.is_file():
            continue        
        im = pydicom.dcmread(str(pp))
        metadata = extract_info(im)
        if metadata['Modality'] == 'RTSTRUCT':
            continue
        if metadata["Modality"] != modality or metadata["orientation"] != orientation:
            continue
        
        
        try:
            arr = im.pixel_array
        except Exception as e:
            print(f"Exception: {e} for file: {str(pp)} \n")
            continue

        if arr.dtype==np.uint16:
            print("The image data type is not readable for file: {}".format(str(pp)))
            break

        intercept = float(metadata["RescaleIntercept"])
        slope = float(metadata["RescaleSlope"])
        ww = float(metadata["WindowWidth"])
        wl = float(metadata["WindowCenter"])
        arr = rescale_intensity(arr, intercept, slope)
        arr = apply_ww_wl(arr, ww, wl)
        arr = normalize_array(arr)

        pp_rel = pp.relative_to(root_dir)
        output_pp = (output_dir / pp_rel).with_suffix('.jpg')
#         print(output_pp)
        output_pp.parent.mkdir(exist_ok=True, parents=True)
#             _, filename = os.path.split(filename)
#             output_path = os.path.join(output_directory, filename.replace(".dcm", ".jpg"))
        metadata['original_path'] = str(pp)
        metadata['rel_path'] = str(pp_rel)
        metadata['uid'] = str(pp.stem)
        
        if output_directory is not None:
            imsave(str(output_pp), (arr * 255).astype(np.uint8))
            metadata['output_path'] = str(output_pp)
        
        metadata = convert_dtypes(metadata)
        metadata_list.append(metadata)
    if len(metadata_list) > 0:
        with open(str(output_dir / 'dicom_meta.json'), "w") as output_file:
            json.dump(metadata_list, output_file)
            
    return metadata_list


if __name__ == '__main__':
    data_path = '/export/scratch3/grewal/Data/MODIR_data_train_split/'
    output_path = '/export/scratch3/bvdp/segmentation/data/AMC_dicom_train/'
    root_dir = Path(data_path)
    copy_dir = Path(output_path)

    for i, pp in enumerate(root_dir.glob('*/*')):
        print("\nProcessing", i, ":", str(pp), '\n')
        if (output_path / pp.relative_to(data_path) / 'dicom_meta.json').exists():
            print("Already processed. Skipping...")
            continue
        md_list = process_dicoms(
            str(pp),
            str(output_path / pp.relative_to(data_path)))