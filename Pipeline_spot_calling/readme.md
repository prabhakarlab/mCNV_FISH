# mCNV_FISH spot calling pipeline

Here we detail the python pipeline to perform 3D spot calling of z-stacks across hybridization cycles.

## Running the pipeline

A demo dataset containing images in .tif format is available upon request (as a Google drive link). For reproducibility, the link also contains the environment packaged as a singularity .sif file; alternatively, the .yml file is provided in the 'Environment' folder. All python code needed for these steps in the pipeline are provided in the 'Pipeline_spot_calling' folder.

The pipeline assumes that steps in the 'Pipeline_qc_registration' folder have been run. We now describe the steps to run the spot-calling itself.

## Pipeline overview
The steps are available as a single-pass run bash script (see '/Pipeline_bash_scripts/pipeline_onepassrun.sh'). We include an explanation of each step below.

## Input data format

The pipeline takes the expected output following the qc_registration, segmentation and stitching steps. See 'Expected output' from the readme in the 'Pipeline_qc_registration' folder.

## Steps in the spot-calling pipeline
Step 1: Preprocess background (for prehybridization images)
```
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD --preprocess_background --registration_mode precomputed -p 8 --fov_file fovs.txt --run_notes run_tracker 
```

Step 2: Postprocess segmentation (filters for 3D masks segmented by Cellpose)
```
python /path/to/your/directory/Pipeline_spot_calling/run_postprocess_segmentation.py --data_id your_dataset_id --seg_model cellpose_modelD -p 8 --run_notes run_tracker --fov_file fovs.txt 
```

Step 3: Run the spot-calling itself:
This step includes steps 5-11 of the overview figure in the Main readme.md. A brief description is given below:
- Flatfielding - we correct for non-uniform illumination and capture in the microscope.
- Background subtraction - we remove Cy5 and Cy3 background capture in the prehybridization imaging cycle.
- Bandpass filtering - we further filter the image using a 3D Butterworth filter.
- 3D Peak calling - we use a 5 x 5 x 5 maximum filter on the image.
- Peak subsetting - identified peaks (where the maximum filter == the image) are subsetted to only those that are in a segmented cell mask.
- Peak thresholding - total aggregated peaks following subsetting, across all FOVs, are separated into signal versus noise through a bilinear fit.  
```
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD --run_all_bit --registration_mode precomputed --bit all -p 8 --fov_file fovs.txt --run_notes run_tracker 
```

Step 4: Compile information about peaks that will be retained after stitching FOVs, and plot.
```
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD -p 8 --fov_file fovs.txt --run_notes run_tracker --stitch_fov --registration_mode precomputed -p 8
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD -p 8 --fov_file fovs.txt --run_notes run_tracker --count_spots --registration_mode precomputed -p 8
```

Step 5: Cell typing: generate a stitched MIP image of the antibody staining, across all FOVs, and calculate mean intensity values for each antibody channel on a per-cell basis.
```
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD --celltype_ab_1 --registration_mode precomputed -p 8 --run_notes run_tracker
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD --celltype_ab_raw --registration_mode precomputed -p 8 --run_notes run_tracker
```

## Expected output
The expected output is a large number of primarily .h5 files; a full list is provided in the 'expected_output_files' file in this folder ('Pipeline_spot_calling'). Note that the output covers FDR <= 0.1, 0.05 and 0.01 thresholds. A user choosing just one FDR threshold value would have significantly fewer .h5 files.

## Expected running time
Note that the code in this section will provide time stamps for each step in the pipeline. Overall, for a ~100 FOV dataset, steps 1 and 2 take ~ 1 hr each, step 3 takes ~0.5 hrs per hybridization cycle, and steps 4 and 5 take ~1-2 hrs.
