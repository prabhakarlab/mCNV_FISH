# mCNV_FISH
Workflow for the mCNV_FISH pipeline (Kong, Aow, ..., Prabhakar, submitted)


This pipeline contains the code and some utility files necessary for running the mCNV_FISH pipeline. For singularity containers, raw imaging data containing a minimal example, as well as other utility files not uploaded onto Github due to size restrictions, please consult the Zenodo archives (in progress).

The core steps within the imaging analysis pipeline are shown in the figure below:

<img width="14790" height="8763" alt="Data_analysis_pipeline" src="https://github.com/user-attachments/assets/1072a322-73c9-4d3b-97ca-a279aeb43fae" />

The following describes the data structure and image processing steps that were used in this project.
A. Images were acquired under 3 distinct cycles:
   (1) Prehybridization - 3 channels (Cy5, Cy3, DAPI).
   (2) Hybridization - sequential rounds of 2 channels (Cy5, Cy3).
   (3) Antibody staining - 4 channels (Cy5, Cy3, GFP, DAPI).
   For our datasets, step (2) was repeated 18 times, with a chemical bleaching step in between to remove signal from the previous hybridization round. Imaging is typically done on 64-130 FOVs per dataset.

B. Images are therefore processed using the following sequence.
   (1) Registration - 3D z-stacks (z, y, x; per imaging channel) undergo a simple QC (to skip FOVs that have no discernible DAPI signal and therefore have no tissue of interest). Z-stacks passing the QC step are then registered to the prehybridization DAPI z-stack; this is the reference z-stack for all other z-stacks within the FOV. For more information, please see the QC_Registration_README.md.
   (2) Segmentation - the prehybridization DAPI z-stack is segmented in 3D using Cellpose V2 and a refined model specifically trained for this dataset on colorectal cancer epithelia. 
   (3) Stitching - the prehybridization DAPI z-stacks are stitched together to provide a unified reference frame for eventual spot-decoding and cell assignments (e.g., how to assign cells in the overlapping regions between FOVs).
