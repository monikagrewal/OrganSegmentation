from skimage.io import imread, imsave
import torch_AMCDataset
import numpy as np
from pathlib import Path
import sys
from tqdm.auto import tqdm

def visualize_data(volume, mask_volume, output_dir):
    colors = {0: (1, 0, 0), 1: (1, 0, 1), 2: (0, 1, 0), 3: (0, 0, 1),
                4: (1, 1, 0), 5: (0, 1, 1), 6: (1, 0, 1),
                7: (1, 0.5, 0), 8: (0, 1, 0.5), 9: (0.5, 0, 1),
                10: (0.5, 1, 0), 11: (0, 0.5, 1), 12: (1, 0, 0.5)}
    
    volume = volume[0]
    n_slices = volume.shape[0]
    for i in range(n_slices):
        img = volume[i]
        mask = mask_volume[i]
        combined = np.stack((img,)*3, axis=-1)
        opacity = 0.5
        for j in [1,2,3,4]:
            combined[mask == j] = opacity*np.array(colors[j]) + np.stack(((1-opacity)*img[mask == j],)*3, axis=-1)
        combined = np.concatenate((combined, np.stack((img,)*3, axis=-1)), axis=1)
        
        output_path = Path(output_dir) / f"{i}.jpg"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        imsave(str(output_path), (combined * 255).astype(np.uint8))


if __name__ == '__main__':

    # root_dir = '/export/scratch3/bvdp/segmentation/data/MODIR_data_preprocessed_train_23-06-2020/'
    # root_dir = '/export/scratch2/bvdp/Data/Projects_DICOM_data/ThreeD/MODIR_data_test_split_preprocessed_25-06-2020'
    # root_dir = '/export/scratch2/bvdp/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split_preprocessed_21-08-2020'
    # meta_path = "/export/scratch3/bvdp/segmentation/OAR_segmentation/data_preparation/meta/dataset_train_21-08-2020.csv"

    # vis_output_dir = '/export/scratch3/bvdp/segmentation/data/MODIR_data_preprocessed_train_23-06-2020_visualized'
    # vis_output_dir = '/export/scratch2/bvdp/Data/Projects_JPG_data/ThreeD/MODIR_data_preprocessed_train_21-08-2020_visualized'
    
    root_dir = '/export/scratch3/grewal/Data/Projects_JPG_data/ThreeD/segmentation/MODIR_data_test_split'
    meta_path = "/export/scratch3/grewal/OAR_segmentation/data_preparation/meta/dataset_test_14-11-2022_deduplicated_annotated.csv"
    vis_output_dir = '/export/scratch3/grewal/Data/Projects_JPG_data/ThreeD/segmentation/MODIR_data_test_split_labels'
    
    dataset = torch_AMCDataset.AMCDataset(root_dir, meta_path, output_size=512, is_training=False)
    # dataset = torch_AMCDataset.AMCDataset(root_dir, meta_path, output_size=512, is_training=True)

    studies = dataset.meta_df.apply(lambda x: Path(x.path).relative_to(Path(x.root_path)), axis=1)
    for i, study in tqdm(zip(range(len(studies)), studies.values), total=len(studies)):
        volume, mask_volume = dataset[i]
        visualize_data(volume, mask_volume, f'{vis_output_dir}/{i}_{study}')