# mCNV_FISH QC, Registration, Stitching and Segmentation Pipeline

Here we detail the python pipeline to perform QC of the 3D image z-stacks across imaging cycles (prehybridization, hybridization, antibody imaging), registration of z-stacks to the prehybridization DAPI, as well as the stitching pipeline to stitch the prehybridization DAPI z-stacks across FOVs into a unified whole.

## Installation

A minimal example of images, in .tif format, is available upon request (as a Google drive link). For reproducibility, the link also contains the environment packaged as a singularity .sif file; alternatively, the .yml file is provided in the 'Environment' folder. All python code needed for these steps in the pipeline are provided in the 'Pipeline_qc_registration' folder.

## Pipeline overview
- [Stages](#stages)
  - [Stage 1: create the data folder](#stage-1-generate-data-folder)
  - [Stage 2: registration + QC](#stage-2-registration--qc)
  - [Stage 3: post-processing](#stage-3-post-processing)
  - [Stage 4: segmentation](#stage-4-segmentation).
  - [Stage 5: stitching](#stage-4-stitching)
  
