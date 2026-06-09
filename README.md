# mCNV_FISH
Workflow for the mCNV_FISH pipeline (Kong, Aow, ..., Prabhakar, submitted)


This pipeline contains the code and some utility files necessary for running the mCNV_FISH pipeline. For singularity containers, raw imaging data containing a minimal example, as well as other utility files not uploaded onto Github due to size restrictions, please request from the authors.

The core steps within the imaging analysis pipeline are shown in the figure below:

<img width="14790" height="8763" alt="Data_analysis_pipeline" src="https://github.com/user-attachments/assets/1072a322-73c9-4d3b-97ca-a279aeb43fae" />

## Data structure
The following describes the data structure and image processing steps that were used in this project.

A. Images were acquired under 3 distinct cycles:
   1. Prehybridization - 3 channels (Cy5, Cy3, DAPI).
   2. Hybridization - sequential rounds of 2 channels (Cy5, Cy3).
   3. Antibody staining - 4 channels (Cy5, Cy3, GFP, DAPI).
   4. For our datasets, step (2) was repeated 18 times, with a chemical bleaching step in between to remove signal from the previous hybridization round. Imaging is typically done on 64-130 FOVs per dataset.

B. Images are therefore processed using the following sequence.
   1. Registration - 3D z-stacks (z, y, x; per imaging channel) undergo a simple QC (to skip FOVs that have no discernible DAPI signal and therefore have no tissue of interest). Z-stacks passing the QC step are then registered to the prehybridization DAPI z-stack; this is the reference z-stack for all other z-stacks within the FOV. For more information, please see the QC_Registration_README.md.
   2. Segmentation - the prehybridization DAPI z-stack is segmented in 3D using Cellpose V2 and a refined model specifically trained for this dataset on colorectal cancer epithelia.
   3. Stitching - the prehybridization DAPI z-stacks are stitched together to provide a unified reference frame for eventual spot-decoding and cell assignments (e.g., how to assign cells in the overlapping regions between FOVs).
   4. Spot-calling - the main part of the pipeline. This has the following steps:
      1. Z-stacks are flatfield-corrected.
      2. Relative to the initial prehybridization channels, z-stacks are background corrected.
      3. Z-stacks are then filtered using a bandpass filter.
      4. Following normalization across FOVs, peaks are called in 3D using a maximum filter.
      5. Called peaks are then subsetted to only those within the cell masks. The resulting peaks are used to fit a bilinear model that decides the threshold that separates signal from noise peaks.
   5. Clean-up - spots that exhibit signal-bleedthrough are removed.
   6. Cell-typing - using the stitched antibody staining maximum intensity projection (MIP) image, we carry out cell-typing.
   7. Crypt-segmentation - using the stitched antibody staining MIP image, we carry out crypt segmentation.

## Running the pipeline
We provide a bash_script that provides a unified workflow for the entire pipeline. The code files are located in their respective folders - we have broken it up into the qc / registration / stitching, as well as spot-calling. 

For reproducibility, we provide the environment .yml file; else the user may request the environment packaged as a singularity .sif file. The same environment is used for all steps except the 3D segmentation step, which uses a Cellpose V2 .sif file and a custom model (likewise available upon request).

## Resources
Due to the large size of the raw images, the pipeline should be run on a HPC with 24GB RAM per cpu; the spot-calling pipeline (step 4) is designed to run in parallel across FOVs and is frequently run using the following SLURM settings (your HPC may have different specifications): --cpus-per-task 8 --mem 192G.
