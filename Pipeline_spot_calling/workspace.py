import numpy as np
import pandas as pd
import h5py
#import zarr

from skimage.transform import resize
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from skimage.io import imread, imsave

#from dask.distributed import Client, LocalCluster

import joblib 
import glob
import os
import time

from pathlib import Path

from env import *

class Workspace:
    def __init__(self, args): 
        self.base_path = BASE_PATH
        self.main_output_path = os.path.join(self.base_path, f"_processed_data/{args.run_type}-{args.data_id}_{args.run_notes}")
        self.rsfish_path = os.path.join(self.base_path, f"_rsfish/{args.data_id}")

        self.reports_path = os.path.join(self.main_output_path, 'reports')
        self.args = args

        self._data_path = None
        self._params_path = None
        self._coord_path = None
        self._reports_path = None
        
        self.load_params()

    @property
    def data_path(self): 
        if self._data_path is None: 
            data_path = os.path.join(self.base_path, f"_data/data_{self.args.data_id}*{self.args.data_keyword}*")
            print(f'[Initializing workspace]: looking for {data_path}')
            data_path = glob.glob(data_path)[0]
            self._data_path = data_path
        return self._data_path

    @property
    def coord_path(self): 
        if self._coord_path is None: 
            coord_path = os.path.join(self.main_output_path, 'confocal_default')
            self._coord_path = coord_path
        return self._coord_path

    @property
    def params_path(self): 
        if self._params_path is None: 
            params_path = os.path.join(self.main_output_path, '_params.joblib')
            self._params_path = params_path
        return self._params_path

    def load_precomputed_shifts(self, fov): 
        shifts_pat = os.path.join(self.data_path, 
            f"registration/FINAL*_fov_{fov}*")
        shifts_path = Path(glob.glob(shifts_pat)[0])
        if shifts_path.suffix == '.csv':
            shifts = pd.read_csv(shifts_path, index_col = 0)
        else:
            raise NotImplementedError
        return shifts

    def load_params(self):
        # load params and bits
        try:
            with open(self.params_path, 'rb') as f: 
                print(self.params_path)
                self.params = joblib.load(f)
            if self.params['main_output_path'] != self.main_output_path:
                self.params['main_output_path'] = self.main_output_path
                self.params['output_path'] = os.path.join(self.main_output_path,"confocal_default")
                self.params['existing_img_path'] = os.path.join(self.main_output_path,"filtered_images")
                self.params['stitch_path'] = os.path.join(self.main_output_path,"stitched")
                self.params['qc_path'] = os.path.join(self.main_output_path,"confocal_default/qc_plots")
                self.params['data_path'] = glob.glob(self.base_path+f"/_data/data_{self.args.data_id}*")[0]
            cols = ['type_list', 'hyb_list', 'chrom_list', 'bits_list', 'genes']
            self.bits = pd.DataFrame({col:self.params[col] for col in cols}).set_index('bits_list')
        except FileNotFoundError: 
            pass

    ###############

    def load_fovs(self): 
        fov_path = os.path.join(self.data_path, self.args.fov_file) 
        with open(fov_path, 'r') as f: 
            fovs = [line.strip().split(',') for line in f]
        fovs = np.array(fovs)
        return fovs 

    def format_fov(self, fov): 
        fov = f"{fov}".rjust(self.params['strlen'], '0')
        return fov

    def get_fovs_from_segmented(self): 
        seg_paths = Path(self.data_path, "segmentation").glob("*tif")
        seg_paths = [p.name for p in seg_paths]
        fovs = [p.split('.')[0].split('_')[-1][1:] for p in seg_paths]
        return fovs

    ###############

    def load_segmentation(self, fov): 
        fov = self.format_fov(fov) 
        seg_pat = os.path.join(self.data_path, f"segmentation/{self.args.seg_model}/F{fov}_cp_masks.tif")
        print(f'[workspace: load_segmentation] Looking for segmentation using {seg_pat}')
        seg_path = glob.glob(seg_pat)[0]
        st = time.time()
        seg = imread(seg_path)
        et = time.time()
        print(f'[workspace: reading imagefile] Read {seg_path}, shape:{seg.shape} in {et-st:.3f} seconds.') 
    
        if seg.shape[0] != 75: 
            assert False
            print(f'[workspace: load_segmentation] WARNING: Padding (1,1) to z...')
            seg = np.pad(seg, ((1,1),(0,0),(0,0)), mode = 'constant')
        return seg

    def load_signal(self, fov): 
        signal_path = os.path.join(self.params['main_output_path'], f'reports/F{fov}_fcn_bg.csv')
        signal = pd.read_csv(signal_path, index_col = 0)
        return signal

    def load_dapi(self, fov, z = None): 
        name = f"preprocess/prehyb/{fov}/dapi.h5"
        path = os.path.join(self.params['existing_img_path'], name)
        with h5py.File(path, 'r') as f: 
            if z: 
                dapi_rg = f['registered'][z,...]
            else:
                dapi_rg = f['registered'][...]
        return dapi_rg

    def load_prehyb_registered(self, fov, cy, z = None): 
        name = f"preprocess/prehyb/{fov}/{cy}.h5"
        path = os.path.join(self.params['existing_img_path'], name) 
        with h5py.File(path, 'r') as f: 
            if z: 
                prehyb_rg = f['registered'][z,...]
            else:
                prehyb_rg = f['registered'][...]
        return prehyb_rg

    def get_prehybs_registered(self, fov): 
        paths = Path(self.params['existing_img_path']).glob(
            f'preprocess/prehyb/{fov}/*.h5')
        return paths

    def get_prehybs_filtered(self, fov): 
        paths = Path(self.params['existing_img_path']).glob(
            f'preprocess/prehyb/{fov}/*.h5')
        return paths

    def load_prehyb_filtered(self, fov, cy): 
        name = f"preprocess/prehyb/{fov}/{cy}.h5"
        path = os.path.join(self.params['existing_img_path'], name)
        with h5py.File(path, 'r') as f: 
            hyb_ft = f['filtered']
        return hyb_ft

    def load_prehyb_registration_shifts(self, fov, cy): 
        name = f"preprocess/prehyb/{fov}/{cy}.h5"
        path = os.path.join(self.params['existing_img_path'], name)
        with h5py.File(path, 'r') as f: 
            shifts = f.attrs['registration_shifts']
            error = f.attrs['registration_error']
            reference = f.attrs['registration_target']
        return shifts, error, reference

    def load_registration_shifts(self, fov, bit): 
        name = f"preprocess/hyb/{fov}/{bit}.h5"
        path = os.path.join(self.params['existing_img_path'], name) 
        with h5py.File(path, 'r') as f: 
            shifts = f.attrs['registration_shifts']
            try:
                error = f.attrs['registration_error']
            except KeyError: 
                error = np.nan 
            reference = f.attrs['registration_target']
        return shifts, error, reference
    
    def get_hyb_registered(self, fov, bit): 
        name = f"preprocess/hyb/{fov}/{bit}.h5"
        path = Path(self.params['existing_img_path'], name)
        return path

    def get_hyb_filtered(self, fov, bit): 
        name = f"preprocess/hyb/fov/{bit}.h5"
        path = Path(self.params['existing_img_path'], name)
        return path

    def get_hybs_registered(self, fov): 
        paths = Path(self.params['existing_img_path']).glob(f'preprocess/hyb/{fov}/*.h5')
        return paths

    def get_hybs_filtered(self, fov):
        paths = Path(self.params['existing_img_path']).glob(f'preprocess/hyb/{fov}/*.h5')
        return paths

    def load_hyb_registered(self, fov, bit, z = None): 
        name = f"preprocess/hyb/{fov}/{bit}.h5"
        path = os.path.join(self.params['existing_img_path'], name) 
        with h5py.File(path, 'r') as f: 
            if z: 
                hyb_rg = f['registered'][z,...]
            else:
                hyb_rg = f['registered'][...]
        return hyb_rg

    def load_hyb_filtered(self, fov, bit): 
        name = f"preprocess/hyb/{fov}/{bit}.h5"
        path = os.path.join(self.params['existing_img_path'], name) 
        with h5py.File(path, 'r') as f: 
            hyb_ft = f['filtered']
            shifts = f.attrs['registration_shifts']
        return hyb_ft, shifts

    ###############    

    def get_preprocess_hyb_path(self, fov, bit): 
        path = (Path(self.params['existing_img_path']) / 
            f'preprocess/hyb/{fov}/{bit}.h5')
        return path

    def get_postprocess_path(self, fov, bit): 
        path = (Path(self.params['existing_img_path']) / 
            f'postprocess/{fov}/{bit}.h5')
        return path

    def load_background_removed(self, fov, bit): 
        name = f"postprocess/{fov}/{bit}.h5"
        path = os.path.join(self.params['existing_img_path'], name) 
        with h5py.File(path, 'r') as f: 
            hyb_rg = f['background_removed'][...]
        return hyb_rg

    def get_filtered_clipped_norm(self, fov, bit = None): 
        if bit is None:
            paths = Path(self.params['existing_img_path']).glob(
                f'postprocess/{fov}/*.h5')
            return paths
        else: 
            path = (Path(self.params['existing_img_path']) / 
                f'postprocess/{fov}/{bit}.h5')
            return path

    def load_filtered_clipped_norm(self, fov, bit, z = None): 
        name = f"postprocess/{fov}/{bit}.h5"
        path = os.path.join(self.params['existing_img_path'], name)
        with h5py.File(path, 'r') as f: 
            if z:
                hyb_fcn = f['filtered_clipped_norm'][z,...]
            else:
                hyb_fcn = f['filtered_clipped_norm'][...]
        return hyb_fcn

    def init_output_paths(self): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}'
        for output in ['segmentation', 'peaks', 'qc', 'compiled_peaks', 'flat_field', 'bgd_subtraction', 'FFT_filtering']: 
            path_ = path / output
            try:
                os.makedirs(path_)
            except FileExistsError: 
                print(f'Warning! {path_} already exists.')
    
    ###############

    def check_peak_flatness_output_path(self): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}'
        for output in ['peak_flatness_checks']: 
            path_ = path / output
            try:
                os.makedirs(path_)
            except FileExistsError: 
                print(f'Warning! {path_} already exists.')
                
    def check_peak_flatness_path(self, bit): 
        path1 = Path(self.params['main_output_path']) / f'{self.args.seg_model}/peak_flatness_checks/bit_{bit}_all_peaks.csv'
        path2 = Path(self.params['main_output_path']) / f'{self.args.seg_model}/peak_flatness_checks/bit_{bit}_all_peaks_counter.csv'
        path3 = Path(self.params['main_output_path']) / f'{self.args.seg_model}/peak_flatness_checks/bit_{bit}_by_fov.csv'
        path4 = Path(self.params['main_output_path']) / f'{self.args.seg_model}/peak_flatness_checks/bit_{bit}_by_fov_counter.csv'
        return path1, path2, path3, path4   
                
    def get_flat_field_path(self, img_name, fov): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/flat_field/{fov}_{img_name}.png'
        return path
    
    def get_bgd_subtraction_path(self, img_name, fov): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/bgd_subtraction/{fov}_{img_name}.png'
        return path
        
    def get_FFT_filtering_path(self, img_name, fov): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/FFT_filtering/{fov}_{img_name}.png'
        return path
    
    ###############
    
    def get_flat_field_tif_path(self, img_name, fov): 
        path_pre = Path(self.params['main_output_path']) / f'{self.args.seg_model}/flat_field/{fov}_{img_name}_pre.tif'
        path_post = Path(self.params['main_output_path']) / f'{self.args.seg_model}/flat_field/{fov}_{img_name}_post.tif'
        return path_pre, path_post
    
    def get_bgd_subtraction_tif_path(self, img_name, fov): 
        path_pre = Path(self.params['main_output_path']) / f'{self.args.seg_model}/bgd_subtraction/{fov}_{img_name}_pre.tif'
        path_post = Path(self.params['main_output_path']) / f'{self.args.seg_model}/bgd_subtraction/{fov}_{img_name}_post.tif'
        return path_pre, path_post
        
    def get_FFT_filtering_tif_path(self, img_name, fov): 
        path_pre = Path(self.params['main_output_path']) / f'{self.args.seg_model}/FFT_filtering/{fov}_{img_name}_pre.tif'
        path_post = Path(self.params['main_output_path']) / f'{self.args.seg_model}/FFT_filtering/{fov}_{img_name}_post.tif'
        return path_pre, path_post

    ###############
        
    def get_segmentation_qc_path(self, fov): 
        fov = self.format_fov(fov)
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/segmentation/{fov}.csv'
        return path
    
    def get_segmentation_masked_path(self, fov): 
        fov = self.format_fov(fov)
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/segmentation/{fov}.tif'
        return path
    
    def get_segmentation_stitched_path(self, fov): 
        fov = self.format_fov(fov)
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/segmentation/{fov}_stitched.tif'
        return path
    
    def get_segmentation_qc_dir(self): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/segmentation/'
        return path

    def get_peaks_path(self, bit, fov): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/peaks/{bit}/{fov}.h5'
        return path

    def get_peaks_qc_path(self, bit, fdr_val): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/qc/{bit}_{fdr_val}.png'
        return path
        
    def get_peaks_values_path(self, bit): 
        path_thresh = Path(self.params['main_output_path']) / f'{self.args.seg_model}/qc/{bit}_thresh.txt'
        path_counts = Path(self.params['main_output_path']) / f'{self.args.seg_model}/qc/{bit}_counts.txt'
        path_logcounts = Path(self.params['main_output_path']) / f'{self.args.seg_model}/qc/{bit}_logcounts.txt'
        path_firstdev = Path(self.params['main_output_path']) / f'{self.args.seg_model}/qc/{bit}_firstdev.txt'
        path_seconddev = Path(self.params['main_output_path']) / f'{self.args.seg_model}/qc/{bit}_seconddev.txt'       
        return (path_thresh, path_counts, path_logcounts, path_firstdev, path_seconddev)

    ###############

    def get_compiled_peaks_path(self, bit, fov, fdr_val): 
        path = (Path(self.params['main_output_path']) / 
            f'{self.args.seg_model}/compiled_peaks/{fdr_val}/{bit}/{fov}.h5')
        return path

    def get_compiled_peaks(self, fov, bit, fdr_val): 
        compiled_peaks_path = self.get_compiled_peaks_path(bit, fov, fdr_val) 
        with h5py.File(compiled_peaks_path, 'r') as f: 
            coords = pd.DataFrame(f['coords'][:])
            coords.columns = ['z', 'r', 'c']
            coords['intensities'] = f['intensities'][:]
            coords['volume'] = f['volume'][:]
            coords['intden'] = f['intden'][:]
            coords['cell_id'] = f['cell_id'][:]
        coords['r_micron'] = coords['r']*0.12755
        coords['c_micron'] = coords['c']*0.12755
        coords['z_micron'] = coords['z']*0.27
        coords['bit'] = bit
        coords['fov'] = fov 
        return coords
    
    def get_stitched_peaks_path(self, bit, fov, fdr_val): 
        path = (Path(self.params['main_output_path']) / 
            f'{self.args.seg_model}/stitched_peaks/{bit}/{fdr_val}/{fov}.h5')
        return path

    def get_stitched_peaks(self, fov, bit, fdr_val): 
        compiled_peaks_path = self.get_stitched_peaks_path(bit, fov, fdr_val) 
        with h5py.File(compiled_peaks_path, 'r') as f: 
            coords = pd.DataFrame(f['coords'][:])
            coords.columns = ['z', 'r', 'c']
            coords['intensities'] = f['intensities'][:]
            coords['volume'] = f['volume'][:]
            coords['intden'] = f['intden'][:]
            coords['cell_id'] = f['cell_id'][:]
        coords['r_micron'] = coords['r']*0.12755
        coords['c_micron'] = coords['c']*0.12755
        coords['z_micron'] = coords['z']*0.27
        coords['bit'] = bit
        coords['fov'] = fov 
        return coords

    def load_segmentation_masked(self, fov): 
        seg_masked_path = self.get_segmentation_masked_path(fov)
        st = time.time()
        seg_masked = imread(seg_masked_path)
        et = time.time()
        print(f'[workspace: reading seg_masked_path] Read {seg_masked_path}, shape:{seg_masked.shape} in {et-st:.3f} seconds.') 
        return seg_masked
    
    def load_segmentation_stitched(self, fov): 
        seg_masked_path = self.get_segmentation_stitched_path(fov)
        st = time.time()
        seg_masked = imread(seg_masked_path)
        et = time.time()
        print(f'[workspace: reading seg_masked_path] Read {seg_masked_path}, shape:{seg_masked.shape} in {et-st:.3f} seconds.') 
        return seg_masked

    ###############

    def get_stitched_dir(self): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/stitched/'
        return path
    
    def get_stitching_shift_dir(self): 
        path = Path(self.params['data_path']) / 'stitching/'
        return path
    
    def get_ab_intensity_path(self, fov): 
        fov = self.format_fov(fov)
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/membrane/{fov}_ab.csv'
        return path
        
    def get_ab_intensity_raw_path(self, fov): 
        fov = self.format_fov(fov)
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/membrane/Raw_{fov}_ab.csv'
        return path

    def get_ab_intensity_expand_labels_path(self, fov, dist): 
        fov = self.format_fov(fov)
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/membrane/Expanded_{dist}_{fov}_ab.csv'
        return path
    
    def get_ab_intensity_dir(self): 
        path = Path(self.params['main_output_path']) / f'{self.args.seg_model}/membrane/'
        return path

    def load_dt_coords(self, fov, dt, seg = None): 
        # load coordinates, applying a distance threshold 
        coords = self.load_coords(fov, seg) 
        collapsed = []

        # calculate distance in microns

        for hyb_channel, coords_ in coords.groupby('hyb_channel'): 
            for cell_id, cd_ in coords_.groupby('cell'): 
                if cell_id == 0: continue
                spot_distances = squareform(pdist(cd_[['r_micron', 'c_micron', 'z_micron']]))
                _, spot_ids = connected_components(spot_distances < dt) 
                cd_['spot_id'] = spot_ids
                collapsed.append(cd_)
   
        collapsed = pd.concat(collapsed)
        collapsed['duplicated'] = collapsed.duplicated(['cell', 'hyb_channel', 'spot_id'])
        return collapsed

###############
###############
###############

def load_params(ws): 
    params_path = os.path.join(ws.main_output_path, 'confocal_default/_params.joblib')
    with open(params_path, 'rb') as f: 
        params = joblib.load(f)
    cols = ['type_list', 'hyb_list', 'chrom_list', 'bits_list', 'genes']
    bits = pd.DataFrame({col:params[col] for col in cols}).set_index('bits_list')    
    print(f'Reporting {len(bits)} bits.')
    return params, bits
