# Data preparation

## Steps

1. modify input and output paths in `src/1_prepare_data.py` to perpare either the train or test dataset, and run the script to turn the dicom files into numpy array and store them:
   * `root_path`: path to raw dicom data
   * `output_path`: path in which to store the processed numpy arrays in files containing both the image data and label masks
2. modify input and output paths in `src/2_prepare_dataset_metadata.py` to read the prepared train or test dataset, and generate a csv file with paths to the datapoints, and a train/val columns (to be used in the PyTorch Dataset class torch_AMCDataset)
   * `root_dir`: path to directory that contains the prepared data resulting from the previous step
   * `output_dataset`: csv file to save the results in 
3. use `notebooks/determine_and_apply_thresholds.ipynb` (*sorry, didn't have time to make pretty scripts...*) to determine scan and annotations thresholds and output an updated version of the dataset that contains this information:
   1. Make sure root_dir and meta_path in Section 1 of the notebook are set correctly. Running the cells in Section 1 will then load in all the data and determine the slices at which the scan and annoations start/end, and calculate some statistics for analysis
   2. Running the cells in Section 2 will then apply hardcoded thresholds (based on the analysis of section 1) and create a new dataset metaframe that is used in the actual training scripts
   3. Running the cell in Section 3 will actually write the new dataset metaframe to disk

## Notes

* `label_mapping.py` is used by `1_prepare_data.py` to map the messy labels as defined in the RTSTRUCT files to one of our defined classes
* `3_visualize_data.py` can be used to write the dataset to jpegs containing slices of the scans with annotations drawn over them (make sure to update the input/meta/output paths). It uses an outdated version of the torch_AMCDataset we use during training to load the data, but not apply any of the cleaning steps that we apply during our experiments (cutting of scans and bowel bag annotations based on thresholds). So `torch_AMCDataset.py` in this directory is outdated, but will work for this visualization script.




