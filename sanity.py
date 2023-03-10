import os
import skimage
import numpy as np

from config import config
from utils.augmentation import *
from datasets.spleen import *

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

    dataset_plain = SpleenDataset(
    config.DATA_DIR,
    classes=config.CLASSES,
    transform=None,
    )

    dataset = SpleenDataset(
    config.DATA_DIR,
    classes=config.CLASSES,
    transform=transform_train,
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