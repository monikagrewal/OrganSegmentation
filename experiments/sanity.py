import os
import skimage
import numpy as np

from experiments.config import config
from experiments.utils.augmentation_v2 import *
from experiments.datasets.amc import *

def visualize(volume1, volume2, out_dir="./sanity", base_name=0):
    slices = volume1.shape[0]

    imlist = []
    for i in range(slices):
        im = np.concatenate([volume1[i], volume2[i]], axis=1)
        imlist.append(im)
        if len(imlist)==4:
            im = np.concatenate(imlist, axis=0)
            skimage.io.imsave(os.path.join(out_dir, "im_{}_{}.jpg".format(base_name, i)), (im*255).astype(np.uint8))
            imlist = []


if __name__ == '__main__':
    # Random augmentations
    transform_train = ComposeAnyOf([])
    transform_train.transforms.append(RandomElasticTransform3DOrgan(p=1.0, organ_idx=2))

    # Set seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)

    dataset_plain = AMCDataset(
    config.DATA_DIR,
    config.META_PATH,
    config.SLICE_ANNOT_CSV_PATH,
    classes=config.CLASSES,
    transform=None,
    log_path=None,
    )

    dataset = AMCDataset(
    config.DATA_DIR,
    config.META_PATH,
    config.SLICE_ANNOT_CSV_PATH,
    classes=config.CLASSES,
    transform=transform_train,
    log_path=None,
    )

    # make experiment dir
    os.makedirs(config.OUT_DIR, exist_ok=True)

    for i in range(len(dataset)):
        print(i)
        image1, _ = dataset_plain[i]
        image2, _ = dataset[i]
        visualize(image1[0], image2[0], out_dir=config.OUT_DIR, base_name=i)
        if i>5:
            break