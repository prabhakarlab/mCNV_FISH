# refactored so that the base run only considers one bit at a time
# internal parallelization across fovs (because some steps requires all fovs to be processed)
# external parallelization across bits

# -- external imports
import imageio as iio
import joblib
import numpy as np
import pandas as pd
import h5py
import cv2
import tifffile as tif

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import scipy.fftpack
import sklearn.linear_model as sk
from scipy.ndimage import maximum_filter, label
from sklearn.linear_model import HuberRegressor
from skimage.segmentation import watershed

# -- legacy imports
from config import DEFAULTS
from file_parser import ConfocalParser
from frequencyFilter_v2 import butter3d
from imageData import readImages_v2
from skimage.registration import phase_cross_correlation
from skimage.filters import window
from utils import _3D_translation

from workspace import Workspace
from stitch_multi_fdr import Stitcher
from run_stitch_membrane import AbRegistration

# -- system imports
import argparse
import math
import os
import shutil
import time

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

BackgroundFit = namedtuple('BackgroundFit', ('m_', 'bound'))

class BaseRunner: 

    def __init__(self, 
        ws, 
        fovs_to_process, 
    ): 

        self.ws = ws

        # --- define user parameters        
        user_params = DEFAULTS.copy()

        user_params['smfish_callout_method'] = 'peak_3D'
        user_params['z_slice'] = None
        user_params['subtract_chn'] = ['Cy3', 'Cy5']
        user_params['bg_sigma'] = (3, 11, 11)
        user_params['multithread'] = False
        user_params['fdr_percent'] = [0.001]
        user_params['downsize'] = 0.5
        user_params['norm_max_percentile'] = 99.999
        
        user_params['rc_edge_pad'] = 20
        user_params['z_edge_pad'] = 1

        user_params['fovs_to_process'] = fovs_to_process

        params = user_params.copy()

        # --- define file parameters
        file_params_default = {}
        codebook_dir = ws.data_path
        file_params_default['codebook_path'] = os.path.join(codebook_dir, 'fpkm_data.txt')
        file_params = file_params_default

        # -- read codebook file
        codebook = pd.read_csv(file_params['codebook_path'], delimiter = '\t')
        codebook['run'] = 1
        codebook = codebook.sort_values(['hyb_list', 'type_list'], 
            ascending = [True, False])
        params['type_list'] = list(codebook['type_list'])
        params['hyb_list'] = list(codebook['hyb_list'])
        params['chrom_list'] = list(codebook['Chr'])
        params['bits_list'] = codebook.index.values
        params['genes'] = list(codebook['Gene'])
        params['num_bits'] = len(params['hyb_list'])
        params['num_chns'] = len(set(params['type_list']))

        # -- initialize outputs
        params['main_output_path'] = ws.main_output_path
        params['output_path'] = os.path.join(ws.main_output_path, params['name'])
        params['existing_img_path'] = os.path.join(ws.main_output_path, 'filtered_images')
        params['stitch_path'] = os.path.join(ws.main_output_path, 'stitched')
        params['qc_path'] = os.path.join(params['output_path'], 'qc_plots')
        for path_ in ['output_path', 'existing_img_path', 'stitch_path', 'qc_path']: 
            path = params[path_]
            if not os.path.isdir(path):
                os.makedirs(path)
        
        # -- get file parser
        params['data_path'] = ws.data_path
        file_parser = ConfocalParser(ws.data_path, background=params['background'])
        file_parser.parseDirectory()

        filelist_byfov, bgfilelist_byfov, dapifilelist_byfov, params['first_roi'], params['grid_row'], params['grid_col'] = file_parser.dfToDict(
            params['fovs_to_process'], 
            bits_list = params['bits_list'], 
            hyb_list = params['hyb_list'], 
            type_list = params['type_list'], 
            background = params['background'], 
            verbose = False
        )

        # -- update params
        # --- CHANGE: files_df['ydim'/'xdim'/'frames'] are now populated by
        #     reading the tiff shape directly (tifffile.series[0].shape) instead
        #     of regex-matching `Width=` / `StepCount=` in a sidecar _metadata.txt.
        #     This fixes the manually-merged-channel case where the sidecar lied.
        #     NOTE: the existing y<->x swap below is INTENTIONALLY PRESERVED --
        #     it compensates for an axis-convention mismatch downstream. Do not
        #     'fix' it without auditing every consumer of params['xdim']/['ydim'].
        params['xdim'] = int(file_parser.files_df['ydim'].values[0])
        params['ydim'] = int(file_parser.files_df['xdim'].values[0])
        params['zdim'] = int(file_parser.files_df['frames'].values[0])
        params['strlen'] = max([len(fov) for fov in params['fovs_to_process'] if fov !='xxx'])

        self.params = params
        self.codebook = codebook
        self.filelist_byfov = filelist_byfov
        self.bgfilelist_byfov = bgfilelist_byfov
        self.dapifilelist_byfov = dapifilelist_byfov
        self._hyb_files = None

        # -- precalculate for registration 
        self.params['low_cut'] = 100
        self.params['high_cut'] = 300
        self.get_freq_filter()

        # -- for peak calling 
        self.params['peaks_min_dist'] = 5 
        self.params['max_peak_num_perfov'] = 150000
        self.params['num_bins'] = 300
        self.params['threshold_init_percentile'] = 90
        self.params['threshold_use_background'] = False
        self._global_max_norm = None
    
    ## SAVE PARAMS
    ##################
    def save_params(self):      
        params_file = self.ws.params_path
        if Path(params_file).exists(): 
            return
        
        with open(params_file, 'wb') as f: 
            joblib.dump(self.params, f) 
        print(f'Wrote params file to {params_file}')
    ##################
    
    ## GET FILTER
    ##################
    def get_freq_filter(self): 
        self.filter_shifted = self._get_freq_filter(
            self.params['low_cut'],
            self.params['high_cut'])
        
    def _get_freq_filter(self, low_cut, high_cut): 
        # note that the loaded filter is already fft and shifted
        # hence to apply it we can do ifft2(ifftshift(fftshift(fftn(im))*fft_filt)).real
        # equivalently we can do ifftn(fftn(im)*ifftshift(fft_filt)).real
        # and apparently fftshift and ifftshift are equivalent when the size of the vector is even
        # https://help.scilab.org/docs/6.1.0/en_US/ifftshift.html which is how it was originally implemented
        freq_filter = butter3d(
            low_cut = low_cut, 
            high_cut = high_cut, 
            filter_path = os.path.join(self.params['data_path'], 'filters'), 
            use_existing = True,
            order = self.params['bw_filter_order'], 
            xdim = self.params['xdim'], 
            ydim = self.params['ydim'], 
            zdim = self.params['zdim'], 
            plot_filter = False,
            verbose = False)
        filter_shifted = np.fft.fftshift(freq_filter)
        return filter_shifted
    ##################
    
    ## PROCESS MAX 10 PLANES IMAGE FOR SAVING, EITHER THE MAXIMUM OR THE AVERAGE OF THE MAX10P
    ##################
    def get_fluor_curve(self, stack):
        fc = np.zeros(min(stack.shape))
        for plane in range(len(fc)):
            fc[plane] = np.sum(stack[plane, :, :])
        fc = (fc - np.min(fc))/(np.max(fc)-np.min(fc))
        return fc
        
    def process_stack(self, stack, window=10, ds_xy=2):
        fluor_curve = self.get_fluor_curve(stack)
        argmax = np.argmax(fluor_curve)
        if argmax < int(window+1)/2:
            maxplanes = stack[:window, :, :]
        elif argmax > (len(fluor_curve) - int(window+1)/2):
            maxplanes = stack[-window:, :, :]
        else:
            maxplanes = stack[int(argmax-(window-1)/2):int(argmax+(window-1)/2), :, :]
        max_maxplanes = np.max(maxplanes, axis=0) 
        
        ## clipping:
        np.clip(max_maxplanes, 0, 4095, out=max_maxplanes)
           
        DS = np.zeros((int(np.floor(max_maxplanes.shape[1]/ds_xy)), int(np.floor(max_maxplanes.shape[0]/ds_xy)))) 
        DS = cv2.resize(max_maxplanes, (DS.shape[1], DS.shape[0]), interpolation = cv2.INTER_CUBIC)    
        return DS.astype('uint16') 
        
    def process_stack_avg(self, stack, window=10, ds_xy=2):
        fluor_curve = self.get_fluor_curve(stack)
        argmax = np.argmax(fluor_curve)
        if argmax < int(window+1)/2:
            maxplanes = stack[:window, :, :]
        elif argmax > (len(fluor_curve) - int(window+1)/2):
            maxplanes = stack[-window:, :, :]
        else:
            maxplanes = stack[int(argmax-(window-1)/2):int(argmax+(window-1)/2), :, :]
        
        mean_maxplanes = np.mean(maxplanes, axis=0)
        ## clipping:
        np.clip(mean_maxplanes, 0, 4095, out=mean_maxplanes)
            
        DS = np.zeros((int(np.floor(mean_maxplanes.shape[1]/ds_xy)), int(np.floor(mean_maxplanes.shape[0]/ds_xy)))) 
        DS = cv2.resize(mean_maxplanes, (DS.shape[1], DS.shape[0]), interpolation = cv2.INTER_CUBIC)    
        return DS.astype('uint16') 
    ##################
    
    ## FLATFIELD IMAGE: added options for darkbgd correction if a cidre darkbgd correction matrix is supplied.
    ##################    
    def _flatfield_image(self, img, bit):
        if bit % 2==0:
            corr_mat = np.loadtxt(os.path.join(args.flatfield_correction_folder, args.Cy5_mat_file), delimiter = ',')
            if args.Cy5_dn_file is None:
                darkbgd = int(args.Cy5_dark_bgd)
            else:
                darkbgd = np.loadtxt(os.path.join(args.flatfield_correction_folder, args.Cy5_dn_file), delimiter = ',')
        else:
            corr_mat = np.loadtxt(os.path.join(args.flatfield_correction_folder, args.Cy3_mat_file), delimiter = ',')
            if args.Cy3_dn_file is None:
                darkbgd = int(args.Cy3_dark_bgd)
            else:
                darkbgd = np.loadtxt(os.path.join(args.flatfield_correction_folder, args.Cy3_dn_file), delimiter = ',')
      
        newimg = np.zeros(img.shape)
        for i in np.arange(0, img.shape[0], 1):
            newimg[i, :, :] = np.subtract(img[i, :, :].astype(np.float32), darkbgd)
            newimg[i, :, :] = np.multiply(newimg[i, :, :], corr_mat)
            newimg[i, :, :] = np.add(newimg[i, :, :], darkbgd)
            ## clipping:
            np.clip(newimg, 0, 4095, out=newimg)
        
        newimg = newimg.astype(np.uint16)    
        return newimg 
    
    def _comparison_figure(self, before, after, before_title, after_title, output_path, minpct_1=1, minpct_2=5, minpct_3=20, maxpct_1=99, maxpct_2=99, maxpct_3=99.9):   
        before_mip = np.max(before, axis=0)
        after_mip = np.max(after, axis=0)
        
        fig, ax = plt.subplots(1, 4, figsize = (4*5, 1*4)) # set percentiles as 1-99
        _im = ax[0].imshow(before_mip, cmap=plt.cm.gray, vmin=np.percentile(before_mip, minpct_1), vmax=np.percentile(before_mip, maxpct_2))
        ax[0].set_title(f"{before_title}_{minpct_1}-{maxpct_2}")
        fig.colorbar(mappable=_im, ax=ax[0], pad=0.02, fraction=0.05, aspect=18)
        
        _im = ax[1].imshow(after_mip, cmap=plt.cm.gray, vmin=np.percentile(after_mip, minpct_1), vmax=np.percentile(after_mip, maxpct_1)) 
        ax[1].set_title(f"{after_title}_{minpct_1}-{maxpct_1}")
        fig.colorbar(mappable=_im, ax=ax[1], pad=0.02, fraction=0.05, aspect=18)
        
        _im = ax[2].imshow(after_mip, cmap=plt.cm.gray, vmin=np.percentile(after_mip, minpct_2), vmax=np.percentile(after_mip, maxpct_2)) 
        ax[2].set_title(f"{after_title}_{minpct_2}-{maxpct_2}")
        fig.colorbar(mappable=_im, ax=ax[2], pad=0.02, fraction=0.05, aspect=18)
        
        _im = ax[3].imshow(after_mip, cmap=plt.cm.gray, vmin=np.percentile(after_mip, minpct_3), vmax=np.percentile(after_mip, maxpct_3)) 
        ax[3].set_title(f"{after_title}_{minpct_3}-{maxpct_3}")
        fig.colorbar(mappable=_im, ax=ax[3], pad=0.02, fraction=0.05, aspect=18)
        
        fig.subplots_adjust(wspace=0.25)
        plt.savefig(output_path, dpi=200, format='png', bbox_inches='tight', pad_inches=0.5)
        plt.close()
    ##################
    
    ## GET FILES
    ##################
    @property
    def hyb_files(self): 
        if self._hyb_files is None: 
            self.get_hyb_files()
        return self._hyb_files
    
    def get_hyb_files(self): 
        hyb_files = []
        for fov in self.filelist_byfov.keys(): 
            for (path, channel, channel_index, hyb, bit) in self.filelist_byfov[fov]: 
                gene = self.codebook.loc[bit, 'Gene']
                chrom = self.codebook.loc[bit, 'Chr']
                hyb_files.append((fov, path, channel, channel_index, hyb, bit, gene, chrom))
        self._hyb_files = pd.DataFrame(hyb_files, 
            columns = ['fov', 'path', 'channel', 'channel_index', 
                    'hyb', 'bit', 'gene', 'chrom'])
    ##################   
    
    ## HELPER FUNCTION FOR ALL DOWNSTREAM PROCESSING
    ##################
    def _init_preprocess_bit(self, fov, bit): 
        # -- get bit_hyb_file 
        bit_hyb_file = self.hyb_files.query(f'fov == \"{fov}\"').query(f'bit == {bit}')
        assert bit_hyb_file.shape[0] == 1
        bit_hyb_file = bit_hyb_file.squeeze(axis = 0)

        # -- metadata and paths
        out_name = f"preprocess/hyb/{fov}/{bit}.h5"
        out_path = Path(self.params['existing_img_path']) / out_name
        out_path.parent.mkdir(parents = True, exist_ok = True)
            
        with h5py.File(out_path, mode = 'a') as f: 
            f.attrs['chn'] = bit_hyb_file.channel
            f.attrs['bit'] = bit_hyb_file.bit
            f.attrs['hyb'] = bit_hyb_file.hyb
            f.attrs['gene'] = bit_hyb_file.gene
            f.attrs['chr'] = bit_hyb_file.chrom
    
        bg_path = f"preprocess/prehyb/{fov}/{bit_hyb_file.channel}.h5"
        bg_path = Path(self.params['existing_img_path']) / bg_path

        return bit_hyb_file, out_path, bg_path
    ##################
     
    
    ## STEP 0: PREPROCESS BACKGROUND
    ##################
    def preprocess_background(self, mode): 
        # run fovs in parallel
        if self.ws.args.num_workers <= 1: 
            for fov in self.params['fovs_to_process']: 
                if (not self._check_preprocess_background(fov)) and fov!='xxx': 
                    self._preprocess_background(fov, mode)
        else: 
            Parallel(n_jobs = self.ws.args.num_workers)(
                delayed(self._preprocess_background)(fov, mode)
                    for fov in self.params['fovs_to_process'] if 
                    (not self._check_preprocess_background(fov)) and fov!='xxx'
            )
    
    def _check_preprocess_background(self, fov): 
        num_prehyb_filtered = len([p for p in self.ws.get_prehybs_filtered(fov)])
        check_num_prehyb_filtered = num_prehyb_filtered == 2 and fov == "xxx"
        return check_num_prehyb_filtered
    
    def _preprocess_background(self, fov, mode = 'precomputed'):         
        img_list = pd.DataFrame(self.bgfilelist_byfov[fov])
        img_list.columns = ['path', 'channel', 'channel_ix', 'hyb']
        img_list = img_list.set_index('channel')

        # read image 
        bgfile_path = img_list.iloc[0]['path']
        st = time.time()
        bgfile = iio.v3.imread(os.path.join(self.params['data_path'], bgfile_path))
        
        #####
        ## FLAT FIELD THE PREHYB HERE:
        bgfile_cy5 = bgfile[:, 0, :, :]
        bgfile[:, 0, :, :] = self._flatfield_image(bgfile_cy5, 0)
        cy5_output_path = ws.get_flat_field_path('Prehyb_Cy5', fov)
        if args.save_intermediate_MIPs:
            cy5_path_pre, cy5_path_post = ws.get_flat_field_tif_path('Prehyb_Cy5', fov)
            tif.imwrite(cy5_path_pre, self.process_stack_avg(bgfile_cy5), dtype ='uint16', photometric='minisblack')
            tif.imwrite(cy5_path_post, self.process_stack_avg(bgfile[:, 0, :, :]), dtype ='uint16', photometric='minisblack')
        
        bgfile_cy3 = bgfile[:, 1, :, :]
        bgfile[:, 1, :, :] = self._flatfield_image(bgfile_cy3, 0)
        cy3_output_path = ws.get_flat_field_path('Prehyb_Cy3', fov)
        if args.save_intermediate_MIPs:
            cy3_path_pre, cy3_path_post = ws.get_flat_field_tif_path('Prehyb_Cy3', fov)
            tif.imwrite(cy3_path_pre, self.process_stack_avg(bgfile_cy3), dtype ='uint16', photometric='minisblack')
            tif.imwrite(cy3_path_post, self.process_stack_avg(bgfile[:, 1, :, :]), dtype ='uint16', photometric='minisblack')
        
        del bgfile_cy5, bgfile_cy3
        et = time.time()
        print(f'[_preprocess_background] Read {bgfile_path}, shape:{bgfile.shape} in {et-st:.3f} seconds.') 
        #####
        
        # write dapi for convenience
        bg_name = f"preprocess/prehyb/{fov}/dapi.h5"
        bg_path = Path(self.params['existing_img_path']) / bg_name
        bg_path.parent.mkdir(parents = True, exist_ok = True)
        with h5py.File(bg_path, mode = 'w') as bg: 
            bg.create_dataset('registered', data = bgfile[:,img_list.loc['dapi', 'channel_ix'],...])

        # read shifts
        if mode == 'precomputed':
            shifts_ = self.ws.load_precomputed_shifts(fov)
            shifts_ = shifts_.set_index('tar')

        # assume all shifts are always relative to dapi
        for chn in self.params['subtract_chn']: 
            bg_name = f"preprocess/prehyb/{fov}/{chn}.h5"
            bg_path = os.path.join(self.params['existing_img_path'], bg_name)

            with h5py.File(bg_path, mode = 'w') as bg: 
                channel_ix = img_list.loc[chn, 'channel_ix']
                bg_img = bgfile[:,channel_ix,:,:]

                if mode == 'precomputed':
                    shifts = shifts_.loc[
                        f'prehyb_fov_{fov}_cycle_99_ch_{channel_ix}'
                    ]
                    try:
                        shifts = (int(shifts.z), int(shifts.y), int(shifts.x))
                    except ValueError:
                        shifts = (0, 0, 0)
                    bg_registered = _3D_translation(bg_img, shifts).astype(np.uint16)
                elif mode == 'register': 
                    raise NotImplementedError
                elif mode == 'none':
                    shifts = (0, 0, 0)
                    bg_registered = bg_img
                else:
                    raise NotImplementedError

                bg.create_dataset(f'registered', data = bg_registered)
                bg.attrs['registration_shifts'] = tuple(shifts)
                print(f'[_preprocess_background] Registered {chn} to dapi with precomputed shifts {shifts} for fov {fov}.')
    ##################
    
    
    ## STEP 2: BEGIN PROCESS OF PEAKCALLING: BEGIN BY REGISTERING DATASET TO PREHYB (STEP 1 IS THE POSTPROCESS SEGMENTATION)
    ##################
    def register_bit(self, bit, mode):
        if mode == 'precomputed': 
            _register_bit = self._translate_bit_helper
        else:
            raise NotImplementedError
        
        ## AT THIS POINT, BECAUSE MODE IS ALWAYS 'PRECOMPUTED', THE _REGISTER_BIT FUNCTION WILL CALL THE _TRANSLATE_BIT_HELPER FIRST.
        self.get_hyb_files()
        # create output directory if it doesn't exist
        out_dir = Path(self.params['existing_img_path']) / 'preprocess/hyb'
        out_dir.mkdir(parents = True, exist_ok = True)

        if self.ws.args.num_workers <= 1: 
            for fov in self.params['fovs_to_process']: 
                if (not self._check_register_bit(fov, bit)) and fov!='xxx': 
                    _register_bit(fov, bit)
        else: 
            Parallel(n_jobs = self.ws.args.num_workers)(
                delayed(_register_bit)(fov, bit)
                    for fov in self.params['fovs_to_process'] if 
                    (not self._check_register_bit(fov, bit)) and fov!='xxx'
            )
    
    def _check_register_bit(self, fov, bit): 
        # returns True /filtered or any other file post this step exists
        check = self.ws.get_hyb_registered(fov, bit).exists() 
        check = check or self._check_postprocess_bit(fov, bit) or self._check_clip_bit(fov, bit)
        return check
        
    def _translate_bit_helper(self, fov, bit): 
        bit_hyb_file, out_path, bg_path = self._init_preprocess_bit(fov, bit)
        hyb_registered = self._translate_bit(bit_hyb_file, out_path)
        
    def _translate_bit(self, bit_hyb_file, out_path):      
        # read hyb image
        hyb_img_raw, _ = readImages_v2(
            [(bit_hyb_file.path, bit_hyb_file.channel, bit_hyb_file.channel_index)], 
            self.params['data_path'], 
            self.params['z_slice'], 
            smfish_callout_method = self.params['smfish_callout_method']
        )               
        ## FLAT FIELD THE IMAGE HERE.
        hyb_img = self._flatfield_image(hyb_img_raw, int(bit_hyb_file.bit))
        hyb_img_output_path = ws.get_flat_field_path('Hyb_' + str(bit_hyb_file.bit), str(bit_hyb_file.fov))
        if args.save_intermediate_MIPs:
            hyb_path_pre, hyb_path_post = ws.get_flat_field_tif_path('Hyb_' + str(bit_hyb_file.bit), str(bit_hyb_file.fov))
            tif.imwrite(hyb_path_pre, self.process_stack_avg(hyb_img_raw), dtype ='uint16', photometric='minisblack')
            tif.imwrite(hyb_path_post, self.process_stack_avg(hyb_img), dtype ='uint16', photometric='minisblack')
        
        del hyb_img_raw   
        #####

        # read shifts 
        shifts_ = self.ws.load_precomputed_shifts(bit_hyb_file.fov)
        shifts_ = shifts_.set_index('tar').loc[
            f'hyb_fov_{bit_hyb_file.fov}_cycle_{bit_hyb_file.hyb}_ch_{bit_hyb_file.channel_index}']
        shifts = (int(shifts_.z), int(shifts_.y), int(shifts_.x))

        # run translation
        hyb_registered = _3D_translation(hyb_img, shifts).astype(np.uint16)

        with h5py.File(out_path, mode = 'a') as f: 
            f.attrs['registration_shifts'] = tuple(shifts)
            f.create_dataset('registered', data = hyb_registered) 
            
        print(f'[_translate_bit] Done with registration using precomputed shifts: {shifts}')
        return hyb_registered
    ##################
    
    
    # STEP 3: BEGIN POST-PROCESSING - BACKGROUND SUBTRACTION, FILTERING, AND CLIPPING AND NORMALIZATION.
    ##################
    def postprocess_bit(self, bit): 
        if self.ws.args.num_workers <= 1: 
            for fov in self.params['fovs_to_process']: 
                if (not self._check_postprocess_bit(fov, bit)) and fov!='xxx': 
                    self._postprocess_bit(fov, bit)
        else: 
            Parallel(n_jobs = self.ws.args.num_workers)(
                delayed(self._postprocess_bit)(fov, bit)
                    for fov in self.params['fovs_to_process'] if
                    (not self._check_postprocess_bit(fov, bit)) and fov!='xxx'
            )
            
    def _check_postprocess_bit(self, fov, bit): 
        postprocess_path = self.ws.get_postprocess_path(fov, bit)
        if postprocess_path.exists(): 
            with h5py.File(postprocess_path, 'r') as f: 
                check = 'background_removed' in f
                check = check or 'filtered_clipped_norm' in f
        else:
            check = False
        return check
    
    ## NOW POSTPROCESS
    def _postprocess_bit(self, fov, bit): 
        st = time.time()
        
        # -- subtract background
        fit = self._fit_background(fov, bit)
        bit_hyb_file, fg_path, bg_path = self._init_preprocess_bit(fov, bit)

        with h5py.File(bg_path, 'r') as bg: 
            ref = bg['registered'][...]
            bg_shifts = np.array(bg.attrs['registration_shifts']).astype(int)

        with h5py.File(fg_path, 'r') as f: 
            hyb_registered = f['registered'][...]
            shifts = np.array(f.attrs['registration_shifts']).astype(int)

        hyb_bgremoved = hyb_registered - fit.m_*ref
        
        ########
        hyb_bgd_rmved_output_path = ws.get_bgd_subtraction_path('Hyb_' + str(bit_hyb_file.bit), str(bit_hyb_file.fov))
        if args.save_intermediate_MIPs:
            path_pre, path_post = ws.get_bgd_subtraction_tif_path('Hyb_' + str(bit_hyb_file.bit), str(bit_hyb_file.fov))
            tif.imwrite(path_pre, self.process_stack_avg(hyb_registered), dtype ='uint16', photometric='minisblack')
            tif.imwrite(path_post, self.process_stack_avg(hyb_bgremoved), dtype ='uint16', photometric='minisblack')
        et1 = time.time()
        
        del hyb_registered
        ########
        
        # -- filter 
        hyb_bgremoved_filtered = self._filter(hyb_bgremoved)
        et2 = time.time() 
        
        # -- clip
        mask = hyb_bgremoved_filtered < 0
        hyb_bgremoved_filtered = np.where(mask, 0, hyb_bgremoved_filtered)
        hyb_bgremoved_filtered = self._clip_edges(hyb_bgremoved_filtered, shifts, bg_shifts)
        
        ########
        hyb_filtered_output_path = ws.get_FFT_filtering_path('Hyb_' + str(bit_hyb_file.bit), str(bit_hyb_file.fov))
        if args.save_intermediate_MIPs:
            path_pre, path_post = ws.get_FFT_filtering_tif_path('Hyb_' + str(bit_hyb_file.bit), str(bit_hyb_file.fov))
            tif.imwrite(path_pre, self.process_stack_avg(hyb_bgremoved), dtype ='uint16', photometric='minisblack')
            tif.imwrite(path_post, self.process_stack_avg(hyb_bgremoved_filtered), dtype ='uint16', photometric='minisblack')
        et3 = time.time()  
        
        del hyb_bgremoved 
        ########
        
        # FIND THE NORMALIZED VALUE (FOR THIS FOV) AND SAVE
        img_max = np.percentile(hyb_bgremoved_filtered, self.params['norm_max_percentile'])
        out_name = f"postprocess/{bit_hyb_file.fov}/{bit_hyb_file.bit}.h5"
        out_path = Path(self.params['existing_img_path']) / out_name
        out_path.parent.mkdir(parents = True, exist_ok = True)
        with h5py.File(out_path, 'w') as f: 
            f.attrs['norm_max'] = img_max
            f.create_dataset('background_removed_filtered', data = hyb_bgremoved_filtered)
    
        print(f'[_postprocess_bit] Writing to {out_name}, bkgd_sub:{et1-st:.3f}s, fil_bkgd:{et2-et1:.3f}s, clip_norm:{et3-et2:.3f}s.')
    
    ## GET THE FIT
    def _fit_background(self, fov, bit):  

        # -- get information about the bit
        bit_ = self.ws.bits.loc[bit]
        gene = bit_.genes
        bit = bit_.name
        hyb = bit_.hyb_list
        cy = bit_.type_list

        # choose z based on dapi channel with most signal
        dapi = self.ws.load_prehyb_registered(fov, 'dapi')
        iz = np.argmax(dapi.mean(axis = (1,2)))
        print(f'[_fit_background] Fitting background for fov:{fov} and bit:{bit} at z = {iz}')

        prehyb_ft = self.ws.load_prehyb_registered(fov, cy, iz)
        hyb_ft = self.ws.load_hyb_registered(fov, bit, iz)

        y = hyb_ft.flatten()
        x = prehyb_ft.flatten() 

        # -- select part of data to fit
        v = np.percentile(x, 99)
        hl = np.percentile(y, 5)
        hu = np.percentile(y, 99.99)
        select = (x > v) & (y > hl) & (y < hu) 
        x_ = x[select].astype(np.float32)
        y_ = y[select].astype(np.float32)
        # -- assume no intercept
        m_ = np.linalg.lstsq(x_[:,np.newaxis], y_, rcond = None)[0][0] 
        sigma = np.std(np.abs(y_ - x_*m_))
        bound = scipy.stats.norm.ppf(1-0.05, scale = sigma) # p < 0.01

        fit = BackgroundFit(m_, bound)
        return fit  
        
    ## NOW FILTER.
    def _filter(self, img): 
        img_filtered = self.bw_filter_fft(img)
        return img_filtered
    
    def bw_filter_fft(self, im):
        im_ = self._bw_filter_fft(im, self.filter_shifted)
        return im_
    
    def _bw_filter_fft(self, im, filter_shifted): 
        im_ = np.fft.ifftn(
            scipy.fftpack.fftn(im)*filter_shifted
        ).real.astype(np.float32)
        im_[im_ < 0] = 0
        return im_
    
    ## NOW CLIP EDGES
    def _clip_edges(self, im, shifts, bg_shifts):             
        z_shifts_up = int(np.max([shifts[0], bg_shifts[0]]))
        z_shifts_down = int(np.min([shifts[0], bg_shifts[0]]))
        r_shifts_top = int(np.max([shifts[1], bg_shifts[1]]))
        r_shifts_bottom = int(np.min([shifts[1], bg_shifts[1]]))
        c_shifts_left = int(np.max([shifts[2], bg_shifts[2]]))
        c_shifts_right = int(np.min([shifts[2], bg_shifts[2]])) 
        
        rc_pad = self.params['rc_edge_pad']
        z_pad = self.params['z_edge_pad']

        if z_shifts_up > 0: 
            im[:z_shifts_up+z_pad,:,:] = 0
        else:
            im[:z_pad,:,:] = 0
        if z_shifts_down < 0: 
            im[z_shifts_down-z_pad:,:,:] = 0
        else:
            im[-z_pad:,:,:] = 0
        if r_shifts_top > 0: 
            im[:,:r_shifts_top+rc_pad,:] = 0
        else:
            im[:,:rc_pad,:] = 0
        if r_shifts_bottom < 0: 
            im[:,r_shifts_bottom-rc_pad:,:] = 0
        else:
            im[:,-rc_pad:,:] = 0
        if c_shifts_left > 0: 
            im[:,:,:c_shifts_left+rc_pad] = 0
        else:
            im[:,:,:rc_pad] = 0
        if c_shifts_right < 0: 
            im[:,:,c_shifts_right-rc_pad:] = 0
        else:
            im[:,:,-rc_pad:] = 0
        return im
    ##################
    
    
    # STEP 4: CLIP_BIT
    ##################
    def clip_bit(self, bit): 
        # calculate global norm max
        norm_values = []
        for fov in self.params['fovs_to_process']: 
            if fov=='xxx':
                continue
            postprocess_path = self.ws.get_postprocess_path(fov, bit)
            with h5py.File(postprocess_path, 'r') as f: 
                norm_values.append(f.attrs['norm_max'])
        norm_max_global = np.max(norm_values) 
        if self._global_max_norm is None:
            self._global_max_norm = norm_max_global
            # print(f'[Norm_max_global] calculated value is: {norm_max_global}')
        
        # clip images
        if self.ws.args.num_workers <= 1: 
            for fov in self.params['fovs_to_process']: 
                if (not self._check_clip_bit(fov, bit)) and fov!='xxx': 
                    self._clip_bit(fov, bit, norm_max_global)
        else: 
            Parallel(n_jobs = self.ws.args.num_workers)(
                delayed(self._clip_bit)(fov, bit, norm_max_global)
                    for fov in self.params['fovs_to_process'] if 
                    (not self._check_clip_bit(fov, bit)) and fov!='xxx')
    
    def _check_clip_bit(self, fov, bit): 
        postprocess_path = self.ws.get_postprocess_path(fov, bit)
        if postprocess_path.exists(): 
            with h5py.File(postprocess_path, 'r') as f: 
                check = 'filtered_clipped_norm' in f
        else:
            check = False
        return check
                        
    def _clip_bit(self, fov, bit, norm_max_global): 
        st = time.time()
        postprocess_path = self.ws.get_postprocess_path(fov, bit) 
        with h5py.File(postprocess_path, 'r') as f: 
            hyb_bgremoved_filtered = f['background_removed_filtered'][...]
        hyb_bgremoved_filtered_clipped_norm = hyb_bgremoved_filtered / norm_max_global
        with h5py.File(postprocess_path, 'a') as f: 
            f.create_dataset('filtered_clipped_norm', data = hyb_bgremoved_filtered_clipped_norm)
            del f['background_removed_filtered']
        et = time.time()
        print(f'[clip_bit] Clip bit for FOV_{fov}: {et-st:.3f}s.')
    ##################
    
    
    ## STEP 5: NOW CALL PEAKS
    ##################
    def call_peaks_bit(self, bit): 
        if self.ws.args.num_workers <= 1: 
            for fov in self.params['fovs_to_process']: 
                if (not self._check_call_peaks(fov, bit)) and fov!='xxx':
                    self._call_peaks_bit(fov, bit)
        else: 
            Parallel(n_jobs = self.ws.args.num_workers)(
                delayed(self._call_peaks_bit)(fov, bit) for
                fov in self.params['fovs_to_process'] if 
                (not self._check_call_peaks(fov, bit)) and fov!='xxx')
    
    def _check_call_peaks(self, fov, bit): 
        path = self.ws.get_peaks_path(bit, fov)
        return path.exists()
        
    def _call_peaks_bit(self, fov, bit): 
        st = time.time()
        postprocess_path = self.ws.get_postprocess_path(fov, bit)
        with h5py.File(postprocess_path, 'r') as f: 
            norm_img = f['filtered_clipped_norm'][...]

        max_img = norm_img.astype(np.float32)
        max_img = maximum_filter(max_img, size =  self.params['peaks_min_dist'])

        init_threshold = np.percentile(max_img, self.params['threshold_init_percentile'])
        max_img[max_img < init_threshold] = -1
         
        z, y, x = np.where(norm_img == max_img)
        coords = np.stack((z, y, x), axis = -1)
        intensities = norm_img[z, y, x]

        peaks_path = self.ws.get_peaks_path(bit, fov)
        peaks_path.parent.mkdir(parents = True, exist_ok = True)
        
        et = time.time()
        print(f'[_call_peaks_bit] writing to {peaks_path} taking {et-st:.3f}s.')
        with h5py.File(peaks_path, 'a') as f: 
            f.create_dataset('coords', data = coords)
            f.create_dataset('intensities', data = intensities)
    ##################
    
    
    ## STEP 6: NOW THRESHOLD PEAKS
    ##################
    def threshold_peaks_bit(self, bit, fdr_values = [0.1, 0.05, 0.01], use_background_correction=False): 
        sm_intensities = []
        for fov in self.params['fovs_to_process']: 
            if fov == 'xxx':
                continue
            
            st = time.time()
            sm_intensities_ = self._load_peaks(bit, fov, get_cell_ids=True)
            
            # print(f"[threshold_peaks_bit] Selecting {(sm_intensities_['cell_id'] != 0).sum()} of {sm_intensities_.shape[0]} for fov {fov}")
            sm_intensities_ = sm_intensities_.loc[sm_intensities_['cell_id'] != 0]
             
            sm_intensities.append(sm_intensities_)
            et = time.time()
            print(f'[threshold_peaks_bit] for fov {fov}, sm_intensity time selection: time taken:{et-st:.3f}s.')
        sm_intensities = pd.concat(sm_intensities)

        # -- fit signal/noise
        st = time.time()
        peak_intensities = sm_intensities['intensity'].values
        peaks_qc_paths = []
        for fdr_val in fdr_values:
            peaks_qc_paths.append(self.ws.get_peaks_qc_path(bit, fdr_val))
        peaks_values_path_tuple = self.ws.get_peaks_values_path(bit)
        # print(peaks_values_path_tuple)
        knee_thresholds = self._get_knee_threshold(peak_intensities, fdr_values, peaks_qc_paths, peaks_values_path_tuple)
        # print(f'[threshold_peaks_bit] Calculated knee_thresholds for {fdr_values} are {knee_thresholds}.')
        et = time.time()
        print(f'[threshold_peaks_bit] Getting knee threshold, time taken:{et-st:.3f}s.')
        
        # -- output final coords
        for index, fdr_val in enumerate(fdr_values):
            if self.ws.args.num_workers <= 1: 
                for fov in self.params['fovs_to_process']: 
                    if fov != 'xxx':
                        self._compile_peaks_bit(fov, bit, fdr_values[index], knee_thresholds[index])
            else:
                Parallel(n_jobs = self.ws.args.num_workers)(
                    delayed(self._compile_peaks_bit)(fov, bit, fdr_values[index], knee_thresholds[index])
                    for fov in self.params['fovs_to_process'] if fov != 'xxx')
    
    ## FIRST LOAD PEAKS
    def _load_peaks(self, bit, fov, get_cell_ids = True):        
        peaks_path = self.ws.get_peaks_path(bit, fov)               
        with h5py.File(peaks_path, 'r') as f: 
            coords = f['coords'][:]
            intensities = f['intensities'][:]
        sm_intensities_ = pd.DataFrame(coords)
        sm_intensities_.columns = ['z', 'r', 'c']
        sm_intensities_['intensity'] = intensities

        if get_cell_ids:
            seg = self.ws.load_segmentation_masked(fov) 
            if len(seg.shape) == 3: 
                cell_ids = [ seg[int(smi['z']), int(smi['r']), int(smi['c'])]
                    for _, smi in sm_intensities_.iterrows() ]
            elif len(seg.shape) == 2: 
                cell_ids = [ seg[int(smi['r']), int(smi['c'])]
                    for _, smi in sm_intensities_.iterrows() ]
            else:
                raise NotImplementedError
            sm_intensities_['cell_id'] = cell_ids   
        return sm_intensities_
    
    ## NOW GET KNEE THRESHOLDS. THE CODE BELOW IS FOR THE ORIGINAL HUBER IMPLEMENTATION, WHICH HAS ERRORS FINDING ENDPOINTS FOR NOISE FITTING.
    def _get_knee_threshold(self, peak_intensities, fdr_values, peaks_qc_paths, peaks_values_path_tuple): 
        
        peak_inds_sorted = np.argsort(peak_intensities)
        max_peak_num = max(500000, 
            len(self.params['fovs_to_process'])*self.params['max_peak_num_perfov'])
        peak_intensities_sorted = peak_intensities[peak_inds_sorted]
        
        cdf_end = min(2.5, max(peak_intensities_sorted))
        cdf_start = min(peak_intensities_sorted) 
        thresholds = np.linspace(cdf_start, cdf_end, self.params['num_bins']) 
        counts=[]
        logcounts=[]
        for thresh in thresholds[:-1]:
            count = peak_intensities_sorted[peak_intensities_sorted>thresh].shape[0]
            counts.append(count)
            logcounts.append(math.log(count, 10))
           
        # -- use derivatives to estimate knee point 
        threshdev = thresholds[:int(self.params['num_bins']/2)]
        firstdev = np.diff(logcounts[:len(threshdev)],1)/np.diff(threshdev,1)
        seconddev = np.diff(firstdev,1)/np.diff(threshdev[1:],1)
        np.savetxt(fname = peaks_values_path_tuple[0], X=threshdev, delimiter=',')
        np.savetxt(fname = peaks_values_path_tuple[1], X=counts, delimiter=',')
        np.savetxt(fname = peaks_values_path_tuple[2], X=logcounts, delimiter=',')
        np.savetxt(fname = peaks_values_path_tuple[3], X=firstdev, delimiter=',')
        np.savetxt(fname = peaks_values_path_tuple[4], X=seconddev, delimiter=',')
        # print(seconddev)
                            
        if len(seconddev) == 0:
            local_min_ind = len(threshdev)-1
        else:                               
            local_min_ind = np.argmax(seconddev[np.where(threshdev[:-2]<0.5)[0]][2:])+4 # find the knee point at intensity < 0.5
        
        # find the steepest point
        steep = np.argmin(firstdev[2:local_min_ind])
        if steep-5 < 0:
            low_bin = 0
        else:
            low_bin = steep-5
        
        if steep+5 > local_min_ind:
            mid_bin = local_min_ind
        else:
            mid_bin = steep+5
            
        # print("steep:", steep)
        # print("low_bin:", low_bin)
        # print("mid_bin:", mid_bin)
        
        # -- fit noise model
        x_train = np.array(thresholds[low_bin:mid_bin], dtype=np.float64)
        y_train = np.array(logcounts[low_bin:mid_bin], dtype=np.float64)
        x_train = x_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)

        huber = HuberRegressor(epsilon=1.35)
        huber.fit(x_train, y_train.ravel())
        a = huber.coef_[0]
        b = huber.intercept_
        
        # change to knee_point + 0.5
        high_bin = local_min_ind+2+int(0.5/(cdf_end-cdf_start)*self.params['num_bins'])
        
        # print("local_min_ind:", local_min_ind)
        # print("high_bin:", high_bin)
        
        x_train = np.array(thresholds[local_min_ind+2:high_bin:8], dtype=np.float64) #make the bins smaller
        y_train = np.array(logcounts[local_min_ind+2:high_bin:8], dtype=np.float64)
        x_train = x_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)

        huber = HuberRegressor(epsilon=1.15)
        huber.fit(x_train, y_train.ravel())
        c = huber.coef_[0]
        d = huber.intercept_
                    
        # -- Huber piecewise linear curve
        r_knee_thresholds = []
        for ind, fdr_val in enumerate(fdr_values):
            h_noise_y, h_signal_y, h_FDR_y, h_local_min_ind_2, h_knee_threshold, h_gene_count = self.calc_values_to_plot(threshdev, peak_intensities, a, b, c, d, fdr_value = fdr_val)
            
            ### now do RANSAC
            minid, maxid = self.find_global_min_firstdev(firstdev, smooth_val=10, window=10)
            ra, rb = self.RANSAC_regressor_noise(threshdev, logcounts, minid, maxid)
            id_to_return = self.find_signal_id_firstdev(firstdev, maxid)
            rc, rd = self.RANSAC_regressor_signal(threshdev, logcounts, id_to_return)
    
            ### RANSAC piecewise linear curve
            r_noise_y, r_signal_y, r_FDR_y, r_local_min_ind_2, r_knee_threshold, r_gene_count = self.calc_values_to_plot(threshdev,
    peak_intensities, ra, rb, rc, rd, fdr_value = fdr_val)
    
            self.plot_regression(threshdev, counts, seconddev, 
                            h_noise_y, h_signal_y, h_FDR_y, h_knee_threshold, h_gene_count,
                            r_noise_y, r_signal_y, r_FDR_y, r_knee_threshold, r_gene_count,
                            fdr_value=fdr_val, peaks_qc_path=peaks_qc_paths[ind], h_regtype='Huber', r_regtype='RANSAC') 
            r_knee_thresholds.append(r_knee_threshold)                       
        return r_knee_thresholds
    
    ## FUNCTIONS BELOW ARE THE CORRECT IMPLEMENTATION
    ## CURVE FITTING, FOR FIRSTDEV SMOOTHING
    def smooth(self, y, box_pts): 
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    ## finds the global min of the first derivative, then extends it outwards to include points within 25% of the global min.
    def find_global_min_firstdev(self, firstdev, smooth_val=10, window=10):
        fd = self.smooth(firstdev, smooth_val)  
        gminid = int(np.where(fd == min(fd))[0])    
        gminrange = [gminid]
        i = 0
        while i < window: # extend towards the left, avoiding issues where it's at the terminal index
            i += 1
            if gminid-i >= 0:
                if fd[gminid-i] <= 0.75*min(fd):
                    gminrange.append(gminid-i)
                else:
                    break
        i = 0
        while i < window: # extend towards the right, avoiding issues where it's at the terminal index
            i += 1
            if gminid+i < len(fd):
                if fd[gminid+i] <= 0.75*min(fd):
                    gminrange.append(gminid+i)
                else:
                    break
            
        gminrange.sort()
        # print(f'gminrange is {gminrange}')
        return int(min(gminrange)), int(max(gminrange))

    ## after the noise fit, for the signal portion, find the local maximum
    def find_local(self, curve): 
        if int(np.where(curve == max(curve))[0]) == (len(curve) - 1)/2:  ## because indexing from 0
            return True
        else:
            return False
    
    ## find the signal local maximum. then, after it has been found, extend a little bit back towards the noise region
    def find_signal_id_firstdev(self, firstdev, maxid):
        fd = self.smooth(firstdev, 10)
        index = maxid
        # print(f'maxid for finding signal_id_firstdev: {index}')
        while not self.find_local(fd[index-2:index+3]): #imbalanced, because last index not counted
            index += 1
        
        # print(f'after finding local, index for expanding signal_id_firstdev region: {index}')
        if index >= len(fd):
            index = len(fd)-1 ## because indexing from 0
        i = 0
        while abs(fd[index - i]) <= 1.1*abs(fd[index]) and abs(fd[index - i]) >= 0.9*abs(fd[index]):
            i += 1  # go backwards in the curve 
            
        # print(f'after expanding curve, will return index-i={index-i}')
        
        if index-i < maxid: ## worst case scenario:
            return maxid
        else:   
            return index - i 
    
    ## use RANSAC to fit the noise
    def RANSAC_regressor_noise(self, thresholds, logcounts, minid, maxid):
        x_train = np.array(thresholds[minid:maxid], dtype=np.float64)
        y_train = np.array(logcounts[minid:maxid], dtype=np.float64)
        x_train = x_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
    
        ransac = sk.RANSACRegressor(min_samples=0.75)
        ransac.fit(x_train, y_train.ravel())
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        ra = ransac.estimator_.coef_[0]
        rb = ransac.estimator_.intercept_
        # print(f'Minid: {minid}, Maxid: {maxid}, RANSAC: coeff:{ra}, intercept:{rb}')    
        return ra, rb
    
    ## use RANSAC to fit the signal
    def RANSAC_regressor_signal(self, thresholds, logcounts, id_to_return, window=20):
        if id_to_return+window >= len(thresholds): ## if worst case scenario, because thresholds and logcounts are not the same length
            window = len(thresholds) - 1 - id_to_return
            if window < 3:## now if it turns out that window is too small, with almost nothing to fit: shift everything 5 before, and create a vector of length 5
                window = 5
                id_to_return = len(thresholds) - 1 - window
        # print(f'Range of values for RANSAC to fit signal: {id_to_return}, {id_to_return+window}')
        x_train = np.array(thresholds[id_to_return:(id_to_return+window)], dtype=np.float64)
        y_train = np.array(logcounts[id_to_return:(id_to_return+window)], dtype=np.float64)
        x_train = x_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
    
        ransac = sk.RANSACRegressor(min_samples=0.8)
        ransac.fit(x_train, y_train.ravel())
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        rc = ransac.estimator_.coef_[0]
        rd = ransac.estimator_.intercept_
        # print(f'RANSAC: coeff:{rc}, intercept:{rd}')
        return rc, rd
    
    ## function to calculate datapoints to plot the regression
    def calc_values_to_plot(self, thresholds, peak_intensities, a, b, c, d, fdr_value):
        noise_y = [10**x for x in a*np.array(thresholds[:-1]) + b]
        signal_y = [10**x for x in c*np.array(thresholds[:-1]) + d]                                                           
    
        FDR_y = np.array(noise_y)/(np.array(noise_y)+np.array(signal_y))        
        ## use FDR rate to identify threshold instead of derivative 
        local_min_ind_2 = np.where(FDR_y < fdr_value)[0]    
        if len(local_min_ind_2) < 1:
            knee_threshold = np.inf 
            gene_count = 0
        else:
            knee_threshold = thresholds[min(local_min_ind_2)]
            gene_count = peak_intensities[peak_intensities > knee_threshold].shape[0]
        return noise_y, signal_y, FDR_y, local_min_ind_2, knee_threshold, gene_count
    
    ## function to compare Huber versus RANSAC regression plots
    def plot_regression(self, thresholds, counts, seconddev, 
                        h_noise_y, h_signal_y, h_FDR_y, h_knee_threshold, h_gene_count,
                        r_noise_y, r_signal_y, r_FDR_y, r_knee_threshold, r_gene_count,
                        fdr_value, peaks_qc_path, h_regtype='Huber', r_regtype='RANSAC'):    
        
        cdf_start = min(thresholds)
        cdf_end = max(thresholds)
        fig, ax = plt.subplots(1, 2, figsize = (2*8, 1*7))
        
        local_min_ind = np.argmax(seconddev[np.where(thresholds[:-2]<0.5)[0]][2:])+4
        index = np.r_[slice(0,local_min_ind+2),slice(local_min_ind+2,len(thresholds)-1,4)]
        
        ax[0].plot(thresholds[index], np.array(counts)[index],color='orange', marker = '.', label = 'Data')
        ax[0].axvline(h_knee_threshold, ls = '-.', label = h_regtype + ' FDR thr') #change fdr_value to the threshold
        ax[0].text(h_knee_threshold, 0.99, f'FDR {fdr_value}:\nthr{h_knee_threshold: .2f}',
                 color = 'black', ha = 'right', va = 'top', rotation = 90, 
                 transform = ax[0].get_xaxis_transform()) 
        ax[0].plot(thresholds[:-1], h_noise_y, 'g', label = h_regtype + ' Noise Fit')
        ax[0].plot(thresholds[:-1], h_signal_y, 'black', label = h_regtype + ' Signal Fit')        
        ax[0].set_yscale('log')
        ax[0].set_ylabel('# spots above threshold')
        ax[0].set_xlabel('Intensity')  
        ax[0].set_xlim([cdf_start, cdf_end])
        ax[0].set_ylim(bottom = 0.001)
    
        ax2 = ax[0].twinx()
        line5, = ax2.plot(thresholds[:-1], h_FDR_y, 'm', label = h_regtype + ' FDR Fit')    
        ax2.set_ylabel('FDR rate')
        ax2.set_yscale('log')
        ax2.set_ylim(bottom = 0.001)
        ax2.set_title(f"{h_regtype} : knee thr = {h_knee_threshold:.3f}, gene count = {h_gene_count}")
        
        ###########################        
        
        ax[1].plot(thresholds[index], np.array(counts)[index],color='orange', marker = '.', label = 'Data')
        ax[1].axvline(r_knee_threshold, ls = '-.', label = r_regtype + ' FDR thr')#change fdr_value to the threshold
        ax[1].text(r_knee_threshold, 0.99, f'FDR {fdr_value}:\nthr{r_knee_threshold: .2f}',
                 color = 'black', ha = 'right', va = 'top', rotation = 90, 
                 transform = ax[1].get_xaxis_transform())
        ax[1].plot(thresholds[:-1], r_noise_y, 'blue', label = r_regtype + ' Noise Fit')
        ax[1].plot(thresholds[:-1], r_signal_y, 'black', label = r_regtype + ' Signal Fit')
        ax[1].set_yscale('log')
        ax[1].set_ylabel('# spots above threshold')
        ax[1].set_xlabel('Intensity')  
        ax[1].set_xlim([cdf_start, cdf_end])
        ax[1].set_ylim(bottom = 0.001)
    
        ax3 = ax[1].twinx()
        line6, = ax3.plot(thresholds[:-1], r_FDR_y, 'm', label = r_regtype + ' FDR Fit')
        ax3.set_ylabel('FDR rate')
        ax3.set_yscale('log')
        ax3.set_ylim(bottom = 0.001)
        ax3.set_title(f"{r_regtype} : knee thr = {r_knee_threshold:.3f}, gene count = {r_gene_count}")
        
        fig.subplots_adjust(wspace=0.25)
        plt.savefig(peaks_qc_path, dpi=300, format='png', bbox_inches='tight', pad_inches=0.5)
        print(f'[threshold_peaks_bit] Saving qc plot to {peaks_qc_path}') 
        plt.close()
        
    ## FINALLY, COMPILE THRESHOLDED PEAKS
    def _compile_peaks_bit(self, fov, bit, fdr_val, knee_threshold): 
        st = time.time()       
        sm_intensities_ = self._load_peaks(bit, fov, get_cell_ids = True)
        
        # threshold by knee threshold and remove spots not in cells
        select_knee_threshold = sm_intensities_['intensity'] > knee_threshold
        select_in_cell = sm_intensities_['cell_id'] != 0
        sm_intensities_ = sm_intensities_.loc[select_knee_threshold & select_in_cell]

        # pre calculate volume intden
        img_path = self.ws.get_postprocess_path(fov, bit)
        with h5py.File(img_path, 'r') as f: 
            norm_imgs = f['filtered_clipped_norm'][...]
            volume, intden = self._find_volume_intden(norm_imgs,
                sm_intensities_[['z', 'r', 'c']].values,
                knee_threshold)

        out_path = self.ws.get_compiled_peaks_path(bit, fov, fdr_val)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        et = time.time()
        print(f'[_compile_peaks_bit] writing compiled peaks to {out_path}, time taken before writing:{et-st:.3f}s.')
        with h5py.File(out_path, 'w') as f: 
            f.create_dataset('coords', data = sm_intensities_[['z', 'r', 'c']].values)
            f.create_dataset('intensities', data = sm_intensities_['intensity'].values)
            f.create_dataset('volume', data = volume)
            f.create_dataset('intden', data = intden)
            f.create_dataset('cell_id', data = sm_intensities_['cell_id'].values)   
    
    ## CALCULATE 3D-VOLUME
    def _find_volume_intden(self, 
                   norm_imgs,  
                   coords, 
                   low_threshold,):             
        '''
            Finds the area and integrated density using connected components. 
            Use watershed to split dense peaks      

            Parameters
            ----------
            norm_imgs: np.ndarray
                normalized image
            coords: npndarray
                np.ndarray of coordinates
            low_threshold: float
                initial threshold used to define bottom of the peak
                
        '''                                                                     
        coords = np.array(coords)                                       
        info_arr = np.zeros((len(coords), 2))
        norm_mask = norm_imgs.copy()        
        norm_mask[norm_imgs <= low_threshold] = 0
        norm_mask_2 = norm_mask.copy()   
        norm_mask[norm_imgs > low_threshold] = 1
                        
        mask = np.zeros(norm_mask.shape, dtype=bool)
        coords = coords.astype(int)
        mask[coords[:,0], coords[:,1], coords[:,2]] = True
        markers, _ = label(mask)
        labels = watershed(-norm_mask_2, markers, mask=norm_mask)
    
        for i in range(len(coords)):
            z = int(coords[i,0])
            x = int(coords[i,1])
            y = int(coords[i,2])
            spot_label = labels[z, x, y]

            if spot_label != 0:
                if z-5 < 0:
                    lower_bound = 0
                else:
                    lower_bound = z-5
                if z+5 > labels.shape[0]:
                    upper_bound = labels.shape[0]
                else:
                    upper_bound = z+5
                trial_crop = labels[lower_bound:upper_bound, x-10:x+10, y-10:y+10]
                
                stats_ind = np.where((trial_crop == spot_label))
                area = len(stats_ind[0])
                intden_bbox = norm_mask_2[lower_bound:upper_bound, x-10: x+10, y-10: y+10]              
                intden = np.sum(intden_bbox[stats_ind])
                info_arr[i,0] = area                                                                                                          
                info_arr[i,1] = intden                                                                                              
        return info_arr[:,0], info_arr[:,1]        
    ##################
    
    
    ##################
    def cleanup(self): 
        # removes the following:
        # filtered_images/postprocess
        # filtered_images/preprocess/hyb
        # {seg_model}/compiled_peaks
        # {seg_model}/peaks
        # {seg_model}/qc
        paths_to_delete = [
            Path(self.ws.params['existing_img_path']) / 'preprocess/hyb',
            Path(self.ws.params['existing_img_path']) / 'postprocess',
            Path(self.ws.params['main_output_path']) / f'{self.ws.args.seg_model}/peaks',
            Path(self.ws.params['main_output_path']) / f'{self.ws.args.seg_model}/compiled_peaks',
            Path(self.ws.params['main_output_path']) / f'{self.ws.args.seg_model}/qc',
            Path(self.ws.params['main_output_path']) / f'{self.ws.args.seg_model}/segmentation',
            Path(self.ws.params['main_output_path']) / f'{self.ws.args.seg_model}/stitched',
        ]

        for path in paths_to_delete:
            if path.exists():
                print(f'[cleanup] Removing {path}')
                shutil.rmtree(path)
    ##################


    ##################
    def cleanup_bit(self, bit): 
        paths = Path(self.ws.params['existing_img_path']).glob(f'preprocess/hyb/*/{bit}.h5')
        for path in paths:
            print(f'[cleanup_bit] preprocess: Removing {path}')
            shutil.rmtree(path)

        paths = Path(self.ws.params['existing_img_path']).glob(f'postprocess/*/{bit}.h5')
        for path in paths:
            print(f'[cleanup_bit] postprocess: Removing {path}')
            shutil.rmtree(path)

        path = self.ws.get_peaks_path(bit)
        if os.path.exists(path):
            print(f'[cleanup_bit] peaks: Removing {path}')
            shutil.rmtree(path)
        path = self.ws.get_peaks_qc_path(bit)
        if os.path.exists(path):
            print(f'[cleanup_bit] qc: Removing {path}')
            os.remove(path)
        path = self.ws.get_compiled_peaks_path(bit)
        if os.path.exists(path):
            print(f'[cleanup_bit] compiled_peaks: Removing {path}')
            shutil.rmtree(path)

        path = self.ws.get_stitched_dir() 
        for path_ in path.glob(f'*bit{bit}*'): 
            print(f'[cleanup_bit] stitched: Removing {path_}')
            path_.unlink()
    ##################


##################
##################   

def run_bit(args, bit): 
    print(f'[run_bit] Running for bit = {bit}')
    # parse control 
    if args.run_all_bit: 
        args.register_bit = True
        args.postprocess_bit = True
        args.clip_bit = True
        args.call_peaks_bit = True
        args.threshold_peaks_bit = True
        args.stitch_bit = False

    ws = Workspace(args) 

    runner = BaseRunner(
        ws, 
        fovs_to_process  = args.fovs_to_process
    )

    # register and filter hyb for given bit
    # output: filtered_images/preprocess/hyb
    if args.register_bit:
        runner.register_bit(bit, args.registration_mode)

    # run background subtraction
    # output: filtered_images/postprocess/{fov}/{bit}/background_removed
    if args.postprocess_bit:
        runner.postprocess_bit(bit)

    # run clipping
    # output: filtered_images/postprocess/{fov}/{bit}/filtered_clipped_norm
    if args.clip_bit:
        runner.clip_bit(bit)
    
    # run call peaks
    # output: {seg_model}/peaks 
    if args.call_peaks_bit:
        runner.call_peaks_bit(bit)

    # run threshold peaks
    # output: {seg_model}/compiled_peaks
    # output: {seg_model}/qc
    if args.threshold_peaks_bit:
        runner.threshold_peaks_bit(bit, fdr_values=[0.1, 0.05, 0.01], use_background_correction=True)

    # run threshold peaks in sliding mode
    if args.sliding_threshold_peaks_bit: 
        runner.sliding_threshold_peaks_bit(bit)

    # stitch
    # output: {seg_model}/stitched
    if args.stitch_bit:
        st = Stitcher(ws, fovs)
        st.stitch_bit_overlay(bit,'passed_filtered')
        
    if args.stitch_bit_celltype:
        st = Stitcher(ws, fovs)
        st.stitch_bit_celltype(bit,'mean')

    # cleanup /registered unless args.keep_hyb_rg
    if not ws.args.keep_hyb_rg: 
        for fov in args.fovs_to_process:
            path = ws.get_hyb_registered(fov, bit)
            if os.path.exists(path):
                os.remove(path)#shutil.rmtree(path)
            
            path = ws.get_postprocess_path(fov, bit)
            if os.path.exists(path):
                os.remove(path)

##################
##################            

def run_bit_force(args, bit): 
    try: 
        run_bit(args, bit)
    except Exception as e: 
        print(f'[run_bit] Failed execution on {bit} with {e}') 

##################
##################

if __name__ == "__main__": 
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_id', default = '1234')
    parser.add_argument('--data_keyword', default = '')
    parser.add_argument('-r', '--run_type', default = 'bandpass')
    parser.add_argument('-n', '--run_notes', default = 'normal')
    parser.add_argument('-s', '--seg_model', default = 'cellpose_modelD')
    parser.add_argument('--stitch_coordinates_pattern', default=None,
                        help="Optional glob (in the stitching-shift dir) for the "
                             "master coordinate array. Default: 'Master_coord_array*.npy'. "
                             "Use this to disambiguate if multiple shifts files exist.")
    parser.add_argument('-f', '--fov_file', default = 'fovs.txt')
    parser.add_argument('--flatfield_correction_folder', default = '/path/to/your/flatfield_correction_matrices')
    parser.add_argument('--Cy5_mat_file', default = 'Cy5_GB10.txt')
    parser.add_argument('--Cy3_mat_file', default = 'Cy3_GB10.txt')
    parser.add_argument('--Cy5_dn_file', default = None)
    parser.add_argument('--Cy3_dn_file', default = None)
    parser.add_argument('--Cy5_dark_bgd', default = 99)
    parser.add_argument('--Cy3_dark_bgd', default = 99)

    # parallelization, note that there are 2 loops of parallelization
    # and the total number of workers will be p*q 
    # if p > 1, q should probably be 1
    # because we should prefer to parallelize over fovs 
    parser.add_argument('-p', '--num_workers', default = 1, type = int)
    parser.add_argument('-q', '--num_bit_workers', default = 1, type = int)

    # control workflow
    parser.add_argument('--preprocess_background', action = 'store_true')
    
    parser.add_argument('--run_all_bit', action = 'store_true')
    parser.add_argument('--run_all_bit_bgd_correction', action = 'store_true')
    parser.add_argument('--save_intermediate_MIPs', action = 'store_true')
    parser.add_argument('--register_bit', action = 'store_true')
    parser.add_argument('--postprocess_bit', action = 'store_true')
    parser.add_argument('--clip_bit', action = 'store_true')
    parser.add_argument('--call_peaks_bit', action = 'store_true')
    parser.add_argument('--threshold_peaks_bit', action = 'store_true')
    parser.add_argument('--sliding_threshold_peaks_bit', action = 'store_true')
    parser.add_argument('--stitch_bit', action = 'store_true')
    parser.add_argument('--stitch_fov', action = 'store_true')
    parser.add_argument('--count_spots', action = 'store_true')
    parser.add_argument('--stitch_bit_celltype', action = 'store_true')
    parser.add_argument('--celltype_ab_1', action = 'store_true')
    parser.add_argument('--celltype_ab_2', action = 'store_true')
    parser.add_argument('--celltype_ab_expand_labels', action = 'store_true')
    parser.add_argument('--celltype_ab_raw', action = 'store_true')
    parser.add_argument('--celltype_ab_3', action = 'store_true')
    parser.add_argument('--celltype_ab_refined', action = 'store_true')
    parser.add_argument('--filter_seg', action = 'store_true') 
    
    parser.add_argument('--registration_mode', 
        choices = ['precomputed', 'register', 'none'], 
        default = 'precomputed')
    parser.add_argument('--bit')

    # control cleanup
    parser.add_argument('--cleanup', action = 'store_true')
    parser.add_argument('--cleanup_bit', action = 'store_true')
    parser.add_argument('--keep_hyb_rg', action = 'store_true')
    
    args = parser.parse_args()
    
    ws = Workspace(args)
    fovs = ws.load_fovs()
    args.fovs_to_process = fovs.flatten() 
    
    ### PIPELINE STEPS ARE DETAILED AS BELOW:
    ### Step 0: Create paths for preprocess background 
    if args.preprocess_background or args.cleanup or args.cleanup_bit :
        runner = BaseRunner(
            ws, 
            fovs_to_process  = args.fovs_to_process
        )
    
        if Path(ws.params_path).exists():
            print(f'[init] Found params_path at {ws.params_path}')
        else:
            runner.save_params() 
            print(f'[init] Saving params...')
            ws.load_params()
        print(f'[init] Making output paths...')
        ws.init_output_paths()
    ###
    
    ### Step 1: Preprocess background
    if args.preprocess_background:
        # register and filter prehyb
        # output: filtered_images/preprocess/prehyb
        # note that this is shared across bits and needs to be run before preprocess_hyb
        runner.preprocess_background(args.registration_mode)
    ###
    
    ### Step 2: NOT RUN HERE: RUN THE POST-PROCESS SEGMENTATION:
    ###e.g. python run_postprocess_segmentation.py --data_id 20231101 --seg_model cellpose_modelD -p 8
    ###
    
    ### Step 3: Run bit
    ### determine which bits to process
    if args.bit is None:
        pass
    elif args.bit == 'all': 
        bits = [bit for bit in ws.bits.index]
        print(f'[main] processing all bits = {bits}')
        Parallel(n_jobs = args.num_bit_workers)(
            delayed(run_bit)(args, bit) for bit in bits
        )
    else:
        bitstring = args.bit.split(',')
        print(f'[main] processing multiple bits: {bitstring}')
        for bit in bitstring:
            bit_to_run = int(bit)
            assert bit_to_run in ws.bits.index
            print(f'[main] processing bit = {bit_to_run}')
            run_bit(args, bit_to_run)
    ###        
    
    ### Step 4: Stitch fov
    if args.stitch_fov:
        st = Stitcher(ws, fovs)
        st.stitch_fov(filter_pad = 10) ## adding this because of the bandpass filter, should probably use 20...
        bits = [bit for bit in ws.bits.index]
        Parallel(n_jobs = args.num_bit_workers)(
            delayed(st.stitch_coords)(bit) for bit in bits)
    ###
    
    ### Step 5: Count spots        
    if args.count_spots:# can be done in jupyter notebook as well
        st = Stitcher(ws, fovs)
        fdr_vals = [0.1, 0.05, 0.01]
        for fdr_val in fdr_vals:
            num_spots, coords_ = st.count_spot("passed_stitched", fdr_val)
            st.filter_total_counts(num_spots, fdr_val)
            st.visualize_total_counts(num_spots, fdr_val)
    ###

    ### Step 6A: Do celltyping via antibody staining
    if args.celltype_ab_1:        
        ab = AbRegistration(ws, fovs)#
        ab.stitch_MIP_ab() 
        ab.get_percentiles_intensity()
    ###

    ### Step 6B: doing the celltyping without clipping and without expanding labels
    ### 
    if args.celltype_ab_raw:
        ab = AbRegistration(ws, fovs)
        ab(expand_labels=False, clip=False)
    ###
    
    ### adding in the possibility of expanding the labels for antibody celltyping; don't bother doing the celltyping for now.
    ### 
    if args.celltype_ab_expand_labels:
        ab = AbRegistration(ws, fovs)
        ab(expand_labels=True)
    ###
    
    ### For Clean-up:   
    if args.cleanup:
        runner.cleanup()
    ###
    
    ### For Clean-up by bit
    if args.cleanup_bit:
        if args.bit == 'all': 
            for bit in ws.bits.index:
                runner.cleanup_bit(args.bit)
        else:
            args.bit = int(args.bit)
            assert args.bit in ws.bits.index
            runner.cleanup_bit(args.bit)

