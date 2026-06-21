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
```python /path/to/your/directory/Pipeline_qc_registration/generate_data_folder.py --source /path/to/your/data/ --dest /path/to/your/data/_data/your_dataset```

## Stage 2: run the qc_registration
```python /path/to/your/directory/Pipeline_qc_registration/run_qc_registration.py --source /path/to/your/data/_data/your_dataset/ --dest /path/to/your/data/_data/your_dataset/registration/ --fovs 080,081```


