import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops


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
        elif (
            (len(props) == 1)
            or (class_idx not in multiple_organ_indici)
            or (props[0].area > 1.5 * props[1].area)
        ):
            segm_pp[initial_segmentation & (valid_mask != props[0].label)] = bg_idx
            segm_pp[initial_segmentation & (valid_mask == props[0].label)] = class_idx
        else:
            mask = np.logical_or(
                (valid_mask == props[0].label), (valid_mask == props[1].label)
            )
            segm_pp[initial_segmentation & np.logical_not(mask)] = bg_idx
            segm_pp[initial_segmentation & mask] = class_idx
    return segm_pp
