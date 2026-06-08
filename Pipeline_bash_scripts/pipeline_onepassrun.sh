#!/bin/bash
# generate data folder
python /path/to/your/directory/Pipeline_qc_registration/generate_data_folder.py --source /path/to/your/data/ --dest /path/to/your/data/_data/your_dataset

# run the qc_registration
python /path/to/your/directory/Pipeline_qc_registration/run_qc_registration.py --source /path/to/your/data/_data/your_dataset/ --dest /path/to/your/data/_data/your_dataset/registration/ --fovs 080,081

# run the qc_registration postprocessing
python /path/to/your/directory/Pipeline_qc_registration/qc_registration_postprocessing.py --source /path/to/your/data/_data/your_dataset/registration/ --dest /path/to/your/data/_data/your_dataset/registration/ --source_im /path/to/your/data/_data/your_dataset/

# downsample for dapi, run cellpose, rename the cellpose masks file
python /path/to/your/directory/Pipeline_spot_calling/downsample.py --source /path/to/your/data/_data/your_dataset/prehyb/ --dest /path/to/your/data/_data/your_dataset/segmentation/cellpose_modelD/ --keyword .tif --ds_xy 1 --ds_z 1
singularity run /path/to/your/container/cellpose.sif python -m cellpose --dir /path/to/your/data/_data/your_dataset/segmentation/cellpose_modelD/ --pretrained_model /path/to/your/container/modelD --diameter 35 --save_tif --verbose --no_npy --anisotropy 0.472 --do_3D --cellprob_threshold -6
python /path/to/your/directory/Pipeline_spot_calling/rename_cellpose_masks.py /path/to/your/data/_data/your_dataset/segmentation/cellpose_modelD/

# copy the utility files
cp /path/to/fovs.txt /path/to/your/data/_data/your_dataset/
cp /path/to/fpkm_data.txt /path/to/your/data/_data/your_dataset/
cp /path/to/mCNV-FISH_gene\ inseq_Dec2023KMS.csv /path/to/your/data/_data/your_dataset/

# run the stitching
python /path/to/your/directory/Pipeline_qc_registration/stitching_3D_dapi.py --source /path/to/your/data/_data/your_dataset/ --dest /path/to/your/data/_data/your_dataset/stitching/ --fov_layout /path/to/your/data/_data/your_dataset/fovs.txt

# now run the spot-calling pipeline
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD --preprocess_background --registration_mode precomputed -p 8 --fov_file fovs.txt --run_notes run_tracker 
python /path/to/your/directory/Pipeline_spot_calling/run_postprocess_segmentation.py --data_id your_dataset_id --seg_model cellpose_modelD -p 8 --run_notes run_tracker --fov_file fovs.txt 
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD --run_all_bit --registration_mode precomputed --bit all -p 8 --fov_file fovs.txt --run_notes run_tracker 
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD -p 8 --fov_file fovs.txt --run_notes run_tracker --stitch_fov --registration_mode precomputed -p 8
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD -p 8 --fov_file fovs.txt --run_notes run_tracker --count_spots --registration_mode precomputed -p 8
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD --celltype_ab_1 --registration_mode precomputed -p 8 --run_notes run_tracker
python /path/to/your/directory/Pipeline_spot_calling/run_all_bit.py --data_id your_dataset_id --seg_model cellpose_modelD --celltype_ab_raw --registration_mode precomputed -p 8 --run_notes run_tracker