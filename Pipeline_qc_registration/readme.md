# mCNV_FISH QC, Registration, Stitching and Segmentation Pipeline

Here we detail the python pipeline to perform QC of the 3D image z-stacks across imaging cycles (prehybridization, hybridization, antibody imaging), registration of z-stacks to the prehybridization DAPI, as well as the stitching pipeline to stitch the prehybridization DAPI z-stacks across FOVs into a unified whole.

## Installation

A demo dataset containing images in .tif format is available upon request (as a Google drive link). For reproducibility, the link also contains the environment packaged as a singularity .sif file; alternatively, the .yml file is provided in the 'Environment' folder. All python code needed for these steps in the pipeline are provided in the 'Pipeline_qc_registration' folder.

## Pipeline overview
- [Stages](#stages)
  - [Stage 1: create the data folder](#stage-1-generate-data-folder)
  - [Stage 2: registration + QC](#stage-2-registration--qc)
  - [Stage 3: post-processing](#stage-3-post-processing)
  - [Stage 4: segmentation](#stage-4-segmentation).
  - [Stage 5: stitching](#stage-4-stitching)

These steps are also available as a single-pass run bash script (see '/Pipeline_bash_scripts/pipeline_onepassrun.sh').
  
## Input data format

The pipeline expects a single source directory containing subdirectories per imaging round. Note that the hybridization cycles have been split into multiple folders:
```
└── demo
    ├── ab
    │   ├── ab_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_488_GFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F080.tif
    │   ├── ab_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_488_GFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F081.tif
    │   └── ab_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_488_GFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_metadata.txt
    ├── hyb00-04
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_F081.tif
    │   └── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_metadata.txt
    ├── hyb05-10
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_5_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_5_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_5_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_F081.tif
    │   └── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_metadata.txt
    ├── hyb11-17
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_5_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_5_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_5_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_6_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_6_F081.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_6_metadata.txt
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_F080.tif
    │   ├── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_F081.tif
    │   └── hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_metadata.txt
    └── prehyb
        ├── prehyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F080.tif
        ├── prehyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F081.tif
        └── prehyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_metadata.txt

```

Naming conventions are due to the confocal spinning-disk microscope used; your microscope may name things different. It will be important to tweak the code to accept the naming conventions of the microscope in use. Likewise, your microscope may output a variety of other files in addition to the .tif file and the metadata.txt file, but these will not be used.


## Stage 1: create the data folder
This step concatenates all the hyb subfolders, keeps track of the index order according to the naming system, and returns softlinks to the raw data. The exact directory and filename details are handled by `utils.parse_directory` — see the source for the full table of recognized cycle/channel prefixes.
```
python /path/to/your/directory/Pipeline_qc_registration/generate_data_folder.py --source /path/to/your/data/ --dest /path/to/your/data/_data/your_dataset
```

## Stage 2: run the qc_registration
This step performs the quality control (qc) and registration steps. For qc, the pipeline is reading each 3D z-stack and looking for the following aspects of the image:
- Is the z-stack in focus?
- Are there nuclei in the z-stack?

For the registration, the pipeline runs registration to the pre-hybridization DAPI channel in the xy dimensions first, using StackReg and 2D phase-cross-correlation. If there is good agreement between both algorithms, it proceeds to run registration along the z-dimension, using 3 algorithms for cross-comparisons and stability. See the source code or the Methods section in the submitted manuscript for additional methodological details.
```
python /path/to/your/directory/Pipeline_qc_registration/run_qc_registration.py --source /path/to/your/data/_data/your_dataset/ --dest /path/to/your/data/_data/your_dataset/registration/ --fovs 080,081
```

## Stage 3: run the qc_registration postprocessing
Checks to ensure that the registration results are correct are performed in this step. In particular, the postprocessing outputs files that enable quick checks for FOVs and hybridization cycles that have divergent (>= 7/75 planes) z-shifts between the algorithms, and likewise for the xy-dimensions. We highly recommend that this step is manually checked and corrected if microscope performance and imaging parameters might lead to divergence in the reported registration values amongst algorithms. 
```
python /path/to/your/directory/Pipeline_qc_registration/qc_registration_postprocessing.py --source /path/to/your/data/_data/your_dataset/registration/ --dest /path/to/your/data/_data/your_dataset/registration/ --source_im /path/to/your/data/_data/your_dataset/
```

## Stage 4: run the segmentation
We provide the Cellpose nuclei model used to segment our nuclei in this dataset, which is focused on colorectal cancer epithelia. There are two trivial steps, one before and one after the main Cellpose call; the step before extracts out the DAPI z-stack from the 4D (CZYX) prehybridization .tif file, the step after renames the Cellpose masks for the spot-calling pipeline.
```
python /path/to/your/directory/Pipeline_spot_calling/downsample.py --source /path/to/your/data/_data/your_dataset/prehyb/ --dest /path/to/your/data/_data/your_dataset/segmentation/cellpose_modelD/ --keyword .tif --ds_xy 1 --ds_z 1

singularity run /path/to/your/container/cellpose.sif python -m cellpose --dir /path/to/your/data/_data/your_dataset/segmentation/cellpose_modelD/ --pretrained_model /path/to/your/container/modelD --diameter 35 --save_tif --verbose --no_npy --anisotropy 0.472 --do_3D --cellprob_threshold -6

python /path/to/your/directory/Pipeline_spot_calling/rename_cellpose_masks.py /path/to/your/data/_data/your_dataset/segmentation/cellpose_modelD/
```

## Stage 5: run the stitching
The stitching pipeline uses core elements of the registration pipeline, but only registers the prehybridization z-stacks to each other across the FOVs using the overlap (10% in our study; the overlap pixel number will change depending on the percent of overlap and the image size in your data).
```
python /path/to/your/directory/Pipeline_qc_registration/stitching_3D_dapi.py --source /path/to/your/data/_data/your_dataset/ --dest /path/to/your/data/_data/your_dataset/stitching/ --fov_layout /path/to/your/data/_data/your_dataset/fovs.txt
```

Outputs from stages 1-5 are ready for downstream spot-calling. To complete the input_data pipeline for the downstream spot-calling step, we copy three more helper files:
```
# copy the utility files
cp /path/to/fovs.txt /path/to/your/data/_data/your_dataset/
cp /path/to/fpkm_data.txt /path/to/your/data/_data/your_dataset/
cp /path/to/mCNV-FISH_gene\ inseq_Dec2023KMS.csv /path/to/your/data/_data/your_dataset/
```

## Expected output
The following lists the folders and directories created after stages 1-5 are completed:
```
ab
dapi
filters
fovs.txt
fpkm_data.txt
hyb
mCNV-FISH_gene inseq_Dec2023KMS.csv
prehyb
registration
segmentation
stitching

/path/to/your/data/_data/your_dataset/ab:
ab_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_488_GFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F080.tif
ab_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_488_GFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F081.tif
ab_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_488_GFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_metadata.txt

/path/to/your/data/_data/your_dataset/filters:
butter3d_order2_lowcut_100_highcut_300_75_2048_2048.npy

/path/to/your/data/_data/your_dataset/hyb:
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_0_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_0_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_0_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_11_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_11_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_11_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_5_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_5_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_5_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_10_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_10_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_10_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_12_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_12_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_12_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_13_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_13_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_13_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_14_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_14_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_14_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_15_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_15_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_15_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_16_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_16_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_16_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_17_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_17_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_17_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_2_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_3_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_4_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_6_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_6_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_6_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_7_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_7_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_7_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_8_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_8_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_8_metadata.txt
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_9_F080.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_9_F081.tif
hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_9_metadata.txt

/path/to/your/data/_data/your_dataset/prehyb:
prehyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F080.tif
prehyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F081.tif
prehyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_metadata.txt

/path/to/your/data/_data/your_dataset/registration:
_Acr_Bits__fov_080.png
_Acr_Bits__fov_081.png
FAILED_Reg_report_fov_080_.csv
FAILED_Reg_report_fov_081_.csv
FINAL_Reg_fov_080_.csv
FINAL_Reg_fov_081_.csv
MIPs_FOV_080.png
MIPs_FOV_081.png
Post_reg_alignment_FOV_080_.png
Post_reg_alignment_FOV_081_.png
Post_reg_alignment_z_mC_FOV_080__Neg_.png
Post_reg_alignment_z_mC_FOV_080_.png
Post_reg_alignment_z_mC_FOV_081__Neg_.png
Post_reg_alignment_z_mC_FOV_081_.png
QC_Rpt_fov_080.csv
QC_Rpt_fov_081.csv
Reg_attempts_fov_080_.csv
Reg_attempts_fov_081_.csv
Reg_report_fov_080_.csv
Reg_report_fov_081_.csv

/path/to/your/data/_data/your_dataset/segmentation:
cellpose_modelD

/path/to/your/data/_data/your_dataset/segmentation/cellpose_modelD:
F080_cp_masks.tif
F081_cp_masks.tif
prehyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F080_MF_F_DS_z_y_x_add_1_1_0.tif
prehyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_405_DAPI_CF40_Sona 1_F081_MF_F_DS_z_y_x_add_1_1_0.tif

/path/to/your/data/_data/your_dataset/stitching:
Master_coord_array_3D_Dapi_F80-81.npy
Master_shiftxcol_3D_Dapi_F80-81.npy
Master_shiftyrow_3D_Dapi_F80-81.npy
```

In particular, registration values should be corrected within the 'FINAL_Reg_fov_{FOV}_.csv' files. We attach a screenshot of this .csv file for reference:

<img width="924" height="598" alt="Screenshot 2026-06-21 125709" src="https://github.com/user-attachments/assets/4906e8a4-cc64-41c0-9682-60b762cf784b" />

We briefly describe the column headers:
- 'x': Final x-shift to be used by the spot-calling pipeline.
- 'y': Final y-shift to be used by the spot-calling pipeline.
- 'z': Final z-shift to be used by the spot-calling pipeline.
- 'ref': The reference prehybridization DAPI image code (we use the code '99' for prehybridization).
- 'tar': The target z-stack image code.
- 'refind': from the ref-tar pair of z-stacks, the plane with the brighest DAPI fluorescence.
- 'tarind': from the ref-tar pair of z-stacks, the corresponding plane from the target.
- 'shiftz_final': one of three z-shift algorithms used to determine the z-shift.
- 'diff_max_z': one of three z-shift algorithms used to determine the z-shift.
- 'pcc_z': one of three z-shift algorithms used to determine the z-shift.
- 'qc_flags': used to highlight z-stacks with issues (or if no tissue is found).
- 'comments': used to mark manual changes.
  
Corrected values should be placed in the 'x', 'y' or 'z' columns as appropriate (these will be used by the spot-calling pipeline).
