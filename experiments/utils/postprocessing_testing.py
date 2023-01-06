import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
import pdb


def postprocess_segmentation(segm_np, n_classes=5, bg_idx=0, multiple_organ_indici=[3]):
    segm_pp = segm_np.copy()
    struct = ndimage.generate_binary_structure(3, 3)
    for class_idx in range(n_classes):
        if class_idx == bg_idx:
            continue

        initial_segmentation = segm_np == class_idx
        # logging.info("before", initial_segmentation.shape)
        initial_segmentation_dilated = ndimage.binary_dilation(
            initial_segmentation, structure=struct
        ).astype(initial_segmentation.dtype)
        # logging.info("after", initial_segmentation_dilated.shape)
        valid_mask, num = label(
            initial_segmentation_dilated, connectivity=2, return_num=True
        )
        props = regionprops(valid_mask)
        props = sorted(props, key=lambda x: x.area, reverse=True)

        if len(props) == 0:
            pass
        elif class_idx == 1: # for bowel bag
            segm_pp[initial_segmentation & (valid_mask != props[0].label)] = bg_idx
            segm_pp[initial_segmentation & (valid_mask == props[0].label)] = class_idx            
        elif class_idx in multiple_organ_indici: # for hip
            if (
                (len(props) == 1)
                or (props[0].area > 2 * props[1].area)
            ):
                segm_pp[initial_segmentation & (valid_mask != props[0].label)] = bg_idx
                segm_pp[initial_segmentation & (valid_mask == props[0].label)] = class_idx
            else:
                mask = np.logical_or(
                    (valid_mask == props[0].label), (valid_mask == props[1].label)
                )
                segm_pp[initial_segmentation & np.logical_not(mask)] = bg_idx
                segm_pp[initial_segmentation & mask] = class_idx
        elif class_idx in [2, 4]: # for bladder and rectum
            if (
                (len(props) == 1)
                or (props[0].area > 3 * props[1].area)
            ):
                segm_pp[initial_segmentation & (valid_mask != props[0].label)] = bg_idx
                segm_pp[initial_segmentation & (valid_mask == props[0].label)] = class_idx
            else: # select the one closer to hip
                hip_segmentation = segm_np == 3
                hip_mask, num = label(
                    hip_segmentation, connectivity=2, return_num=True
                )
                hip_props = regionprops(hip_mask)
                if len(hip_props)>=2:
                    hip_props = sorted(hip_props, key=lambda x: x.area, reverse=True)
                    # pdb.set_trace()
                    hip_centroid = (np.array(hip_props[0].centroid) + np.array(hip_props[1].centroid)) / 2.

                    dist_from_hip = [np.linalg.norm(hip_centroid - np.array(props[i].centroid)) for i in [0, 1]]
                    selected_comp = np.argmin(dist_from_hip)

                    segm_pp[initial_segmentation & (valid_mask != props[selected_comp].label)] = bg_idx
                    segm_pp[initial_segmentation & (valid_mask == props[selected_comp].label)] = class_idx
                else:
                    mask = np.logical_or(
                    (valid_mask == props[0].label), (valid_mask == props[1].label)
                    )
                    segm_pp[initial_segmentation & np.logical_not(mask)] = bg_idx
                    segm_pp[initial_segmentation & mask] = class_idx

    return segm_pp