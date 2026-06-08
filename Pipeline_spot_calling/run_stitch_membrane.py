# stitch membrane for Joanito
# 20240828: incorporated latest changes from Lin Li before she left

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import skimage
import tifffile
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
from skimage.color import label2rgb, gray2rgb
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.segmentation import watershed
import cv2

from utils import _3D_translation, find_stitching_shifts
from run_postprocess_segmentation import regionprops_intensity_3D
from scipy.signal import argrelextrema

from workspace import Workspace

import argparse
import glob
import os

def clip_and_norm_image(zf, lp = 50, rp = 99.9): 
    zf = zf.copy()
    lq, rq = np.percentile(zf, [lp, rp])
    zf = (zf - lq) / (rq - lq)
    zf[zf < 0] = 0
    zf[zf > 1] = 1
    return(zf)

def clip_and_norm_image_intensity(zf, lc = 100, rc = 1000): 
    zf = zf.copy()
    zf = zf.astype(np.int16)
    zf = (zf - lc) / (rc - lc)
    zf[zf < 0] = 0
    zf[zf > 1] = 1
    return(zf)


class AbRegistration: 

    def __init__(self, 
        ws, 
        fovs_to_process): 

        self.ws = ws
        self.fovs_to_process = np.array(fovs_to_process)
        self.fovs_flat = self.fovs_to_process.flatten()
        self.fov_size = 2048 
        
        out_dir = ws.get_ab_intensity_dir()
        out_dir.mkdir(parents = True, exist_ok = True)
        
        # load shifts if available
        print(f'fovs_flat: {self.fovs_flat}')
        num_fovs = len(self.fovs_flat)-1
        fovs_valid  = [ elem for elem in self.fovs_flat  if elem != 'xxx']
            
        # --- globs for 'Master_coord_array*.npy' (override via --stitch_coordinates_pattern).
        #     Errors if multiple files match (stitching should produce one per dataset).
        shifts_path = find_stitching_shifts(
            self.ws.get_stitching_shift_dir(),
            pattern=getattr(self.ws.args, 'stitch_coordinates_pattern', None),
        )
        if shifts_path is not None:
            print(f'[AbRegistration] loading stitching shifts from {shifts_path.name}')
            self.shifts = np.load(shifts_path).astype(int)
            print(self.shifts.shape)
        else:
            print('[AbRegistration] no stitching shifts file found; continuing without')
        

    def __call__(self, expand_labels=False, clip=True): 
        print(f'[AbRegistration] Running with {self.ws.args.num_workers} workers')
        
        if self.ws.args.num_workers <= 1: 
            intensity_summary = []
            for fov in self.fovs_to_process: 
                if fov =='xxx' or fov=='xx':
                    continue
                props_table = self.run_fov_update(fov, expand_labels, clip) 
                intensity_summary.append(props_table)
        else:
            print('[AbRegistration] Running in parallel')
            intensity_summary = Parallel(n_jobs = self.ws.args.num_workers)(
                delayed(self.run_fov_update)(fov, expand_labels, clip) for fov in self.fovs_flat if fov!='xxx'and fov!='xx')
            
        intensity_summary = pd.concat(intensity_summary)
        ratio = 0.12755 *  0.12755 * 0.27
        intensity_summary['volume'] = intensity_summary['volume'] * ratio
        intensity_summary['surface_area'] = intensity_summary['surface_area'] * 0.12755 *  0.12755 
        intensity_summary['max_perimeter'] = intensity_summary['max_perimeter'] * 0.12755 *  0.12755 
        
        if expand_labels==False:
            if clip==True:
                out_path = self.ws.get_ab_intensity_dir() / 'ab_intensity_summary.csv'
            else:
                out_path = self.ws.get_ab_intensity_dir() / 'ab_intensity_raw_summary.csv'
        else:
            out_path = self.ws.get_ab_intensity_dir() / 'ab_intensity_summary_expand_labels.csv'
            
        print(f'[ab_intensity_summary] Writing output to {out_path}')        
        intensity_summary.to_csv(out_path)    

    def run_fov(self, fov): 
        dapi_prehyb_path = glob.glob(os.path.join(self.ws.data_path, f"dapi/prehyb*_F{fov}.tif"))[0]
        ab_path = glob.glob(os.path.join(self.ws.data_path, f"ab/*F{fov}.tif"))[0]

        dapi = skimage.io.imread(dapi_prehyb_path)[...,-1]
        ab = skimage.io.imread(ab_path)

        shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')
        z_max = 35 #shifts_['prehyb_Dapi_maxplane'].values[0]
        z_span = 5

        ms = []
        for ci in range(3): 
            shifts = shifts_.filter(regex = f"ab_fov_{fov}_cycle_.*_ch_{ci}", axis = 0)
            shifts = (int(shifts.z), int(shifts.y), int(shifts.x))
            m = _3D_translation(ab[...,ci], shifts).astype(np.uint16)
            m = m[z_max-z_span:z_max+z_span]
            ms.append(m)
        ms = np.stack(ms, axis = 0).sum(axis = 1).transpose(1, 2, 0)

        # also return the dapi for sanity checks and visualization
        dapi_ = dapi[z_max-z_span:z_max+z_span].sum(axis = 0)

        return dapi_, ms
    

    def run_fov_update(self, fov, expand_labels, clip=True): 
        print(f"registering FOV{fov}")
        ms = self.register_ab(fov)
        if expand_labels==False:
            return self.calculate_intensity(ms, fov, run_clip=clip)
        else:
            return self.calculate_intensity_expand_labels(ms, fov, run_clip=False)
    
    def register_ab(self, fov): 
        ab_path = glob.glob(os.path.join(self.ws.data_path, f"ab/*F{fov}.tif"))[0]
        ab = skimage.io.imread(ab_path)

        shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')

        ms = []
        for ci in range(3): 
            #0 should be cd45, 1 should be epcam, 2 should be vim, 3 should be dapi
            shifts_dapi = shifts_.filter(regex = f"ab_fov_{fov}_cycle_.*_ch_3", axis = 0)
            shifts = shifts_.filter(regex = f"ab_fov_{fov}_cycle_.*_ch_{ci}", axis = 0)
            shifts = (int(shifts_dapi.z), int(shifts.y), int(shifts.x))
            m = _3D_translation(ab[...,ci], shifts).astype(np.uint16)
            ms.append(m)
        ms = np.stack(ms, axis = -1)
        return ms
    
    def _canvas_shape(self, *extra_dims):
        """
        Compute ``(height, width, *extra_dims)`` for a stitched-output
        canvas large enough to fit every FOV after stitching shifts.

        The nominal size ``(fov_size*rows, fov_size*cols)`` is too small
        on minimal grids (e.g. a 1-column layout) when shifts are nonzero:
        each FOV is placed at canvas columns ``[j_shift, j_shift+fov_size)``,
        and with no extra columns to absorb a positive ``j_shift`` the
        rightmost FOV's right edge falls past the canvas. Numpy then
        silently truncates the assignment slice, and downstream ``np.stack``
        calls fail because the truncated canvas slice is narrower than the
        image slice it's being combined with.

        We size the canvas to ``max(nominal, max_shift + fov_size)`` on
        each axis: unchanged in the normal multi-column case (where the
        nominal size already has slack), grown by ``max_j_shift`` columns
        and/or ``max_i_shift`` rows when needed.
        """
        h = max(self.fov_size * self.fovs_to_process.shape[0],
                int(self.shifts[..., 1].max()) + self.fov_size)
        w = max(self.fov_size * self.fovs_to_process.shape[1],
                int(self.shifts[..., 0].max()) + self.fov_size)
        return (h, w, *extra_dims)
  
    def visualize(self, dapi, all_im_registered, fov, iz=35, smooth=False):
        clip = pd.read_csv(self.ws.get_stitched_dir() / 'clip_ab.csv',index_col=0)

        c = all_im_registered[...,2].copy()
        c = clip_and_norm_image_intensity(c, clip.loc[2,"clip_min"], clip.loc[2,"clip_max"]).max(axis=0)
        
        g = all_im_registered[...,1].copy()
        g = clip_and_norm_image_intensity(g, clip.loc[1,"clip_min"], clip.loc[1,"clip_max"]).max(axis=0)
        
        r = all_im_registered[...,0].copy()
        r = clip_and_norm_image_intensity(r, clip.loc[0,"clip_min"], clip.loc[0,"clip_max"]).max(axis=0)
       
        if smooth:
            c = gaussian_filter(c, sigma=11)*4
            g = gaussian_filter(g, sigma=11)*4
            r = gaussian_filter(r, sigma=11)*4
            
        c_ = gray2rgb(c)
        c_ = c_ * [0, 1, 1]
        
        g_ = gray2rgb(g)
        g_ = g_ * [0, 1, 0]
        
        r_ = gray2rgb(r)
        r_ = r_ * [1, 0, 0]
        
        # ------- dapi
        b = gray2rgb(clip_and_norm_image(dapi.max(axis=0)))
        b = b * [0, 0, 1]
        
        cb = c_ + b
        cb /= np.percentile(cb, 99.5)
        cb[cb > 1] = 1
        
        gb = g_ + b
        gb /= np.percentile(gb, 99.5)
        gb[gb > 1] = 1
        
        rb = r_ + b
        rb /= np.percentile(rb, 99.5)
        rb[rb > 1] = 1

        fig, ax = plt.subplots(1, 3, figsize = (24, 8))
        ax[0].imshow(cb)
        ax[1].imshow(gb)
        ax[2].imshow(rb)
        out_path = self.ws.get_ab_intensity_dir() / f'F{fov}_registered_ab_dapi_separate.png'
        if smooth:
            out_path = self.ws.get_ab_intensity_dir() / f'F{fov}_registered_ab_dapi_separate_smoothed.png'

        plt.savefig(out_path)        
        
    def visualize_ab_only(self, all_im_registered, fov, smooth=False, run_clip=False, raw_image=False):
        c = all_im_registered[...,2].copy()
        g = all_im_registered[...,1].copy()
        r = all_im_registered[...,0].copy()

        if run_clip == True:
            clip = pd.read_csv(self.ws.get_stitched_dir() / 'clip_ab.csv',index_col=0)
            c = clip_and_norm_image_intensity(c.max(axis=0), clip.loc[2,"clip_min"], clip.loc[2,"clip_max"])
            g = clip_and_norm_image_intensity(g.max(axis=0), clip.loc[1,"clip_min"], clip.loc[1,"clip_max"])
            r = clip_and_norm_image_intensity(r.max(axis=0), clip.loc[0,"clip_min"], clip.loc[0,"clip_max"])
        elif raw_image:
            c = (clip_and_norm_image_intensity(c.max(axis=0),lc=0,rc=np.max(c))*255).astype(np.uint8)
            g = (clip_and_norm_image_intensity(g.max(axis=0),lc=0,rc=np.max(g))*255).astype(np.uint8)           
            r = (clip_and_norm_image_intensity(r.max(axis=0),lc=0,rc=np.max(r))*255).astype(np.uint8)
      
        if smooth:
            c = gaussian_filter(c, sigma=11)*4
            g = gaussian_filter(g, sigma=11)*4
            r = gaussian_filter(r, sigma=11)*4
            
        c_ = gray2rgb(c)
        g_ = gray2rgb(g)
        r_ = gray2rgb(r)
    
        fig, ax = plt.subplots(1, 3, figsize = (24, 8))
        ax[2].imshow(c_)
        ax[1].imshow(g_)
        ax[0].imshow(r_)
        ax[0].set_title('CD45')
        ax[1].set_title('EPCAM')
        ax[2].set_title('VIM')
        
        print(f'Now plotting ab_separate_2 with smoothing:{smooth}')
        out_path = self.ws.get_ab_intensity_dir() / f'F{fov}_registered_ab_separate_2.png'
        if smooth:
            out_path = self.ws.get_ab_intensity_dir() / f'F{fov}_registered_ab_separate_smoothed_2.png'
        plt.savefig(out_path)     
        
        c_ = c_ * [0, 1, 1]
        g_ = g_ * [0, 1, 0]
        r_ = r_ * [1, 0, 0]
        fig, ax = plt.subplots(1, 3, figsize = (24, 8))
        ax[2].imshow(c_)
        ax[1].imshow(g_)
        ax[0].imshow(r_)
        ax[0].set_title('CD45')
        ax[1].set_title('EPCAM')
        ax[2].set_title('VIM')
        
        print(f'Now plotting ab_separate with smoothing:{smooth}')
        out_path = self.ws.get_ab_intensity_dir() / f'F{fov}_registered_ab_separate.png'
        if smooth:
            out_path = self.ws.get_ab_intensity_dir() / f'F{fov}_registered_ab_separate_smoothed.png'

        plt.savefig(out_path)   
        if False:
            fig, ax = plt.subplots(1,3,figsize = (11,4), tight_layout=True)
            ax[0].hist(r.flatten(),bins=255)        
            ax[1].hist(g.flatten(),bins=255)
            ax[2].hist(c.flatten(),bins=255)
            ax[0].set_title('CD45')
            ax[1].set_title('EPCAM')
            ax[2].set_title('VIM')
            for i in range(3):
                ax[i].set_xlabel('MIP intensity')
                ax[i].set_ylabel('count of pixels')
            out_path = self.ws.get_ab_intensity_dir() / f'F{fov} registered ab hist.png'
            plt.savefig(out_path)      
        
    
    def stitch_MIP_ab(self):
        canvas = np.zeros(self._canvas_shape(4)).astype('float32')
        print("stitch ab...")
        
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]): 
                fov = self.fovs_to_process[i][j]
                if fov=='xxx' or fov== 'xx' :
                    continue
                j_shift, i_shift, _ =  self.shifts[i,j]
                # -- load ab 
                ab_path = glob.glob(os.path.join(self.ws.data_path, f"ab/*F{fov}.tif"))[0]
                print(f'[stitch_MIP_ab]: loading {fov} Antibody image at {ab_path}.')
                ab = skimage.io.imread(ab_path)
                ab = ab.max(axis = 0)
                shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')
                ms = []
                for ci in range(4): 
                    #0 should be cd45, 1 should be epcam, 2 should be vim, 3 should be dapi
                    shifts = shifts_.filter(regex = f"ab_fov_{fov}_cycle_.*_ch_{ci}", axis = 0)
                    shifts = (int(shifts.y), int(shifts.x))#(int(shifts.z), int(shifts.y), int(shifts.x))
                    m = _3D_translation(ab[...,ci], shifts).astype('float32')
                    ms.append(m)
                ms = np.stack(ms, axis = -1)
                print(ms.shape)
                    
                canvas[canvas.shape[0]-i_shift-self.fov_size - min(shifts_.y):canvas.shape[0]-i_shift, \
                       j_shift- min(shifts_.x):j_shift+self.fov_size] = np.stack((
                    canvas[canvas.shape[0]-i_shift-self.fov_size - min(shifts_.y):canvas.shape[0]-i_shift,\
                           j_shift- min(shifts_.x):j_shift+self.fov_size], 
                    ms[-min(shifts_.y):,-min(shifts_.x):])).max(axis=0)
        
        canvas = np.transpose(canvas,(2,0,1))
        print(f'[stitch_MIP_ab] canvas shape: {canvas.shape}')
        out_path = self.ws.get_stitched_dir() / 'stitched_ab_MIP.tif'        
        tifffile.imwrite(out_path, canvas, metadata={'axes': 'CYX'}, imagej=True)
            
    
    def cal_std(self, bins, n,peak_idx):    
        mean = (bins[peak_idx]+bins[peak_idx+1])/2
        diffsqr = (np.repeat(mean,len(bins[peak_idx+1:]))-(bins[peak_idx:-1]+bins[peak_idx+1:])/2)**2
        std = np.sqrt(np.sum(diffsqr*n[peak_idx:])/np.sum(n[peak_idx:]))
        return mean,std             
    
    def get_percentiles_intensity(self):
        out_path = self.ws.get_stitched_dir() / 'stitched_ab_MIP.tif'
        canvas = tifffile.imread(out_path)
        canvas = np.moveaxis(canvas, 0, -1) ## 20240829 necessary for the subsequent steps, which assume z is the last axis.... 
        fig, ax = plt.subplots(1,3,figsize = (11,4), tight_layout=True)
        # print(f'[reading_canvas_for_get_percentiles_intensity]: canvas shape is {canvas.shape}')
        ax[0].hist(canvas[...,0].flatten(),bins=int(canvas[...,0].flatten().max()//10))
        ax[1].hist(canvas[...,1].flatten(),bins=int(canvas[...,1].flatten().max()//10))
        ax[2].hist(canvas[...,2].flatten(),bins=int(canvas[...,2].flatten().max()//10))
        ax[0].set_title('CD45')
        ax[1].set_title('EPCAM')
        ax[2].set_title('VIM')
        for i in range(3):
            ax[i].set_xlabel('MIP intensity')
            ax[i].set_ylabel('count of pixels')
        out_path = self.ws.get_stitched_dir() / 'stitched ab hist.png'
        plt.savefig(out_path)
        
        # calculate global norm max
        clip = pd.DataFrame(index = [0,1,2],columns = ['clip_min','clip_max'])
    
        fig, ax = plt.subplots(1,3,figsize = (11,4), tight_layout=True)
        data = canvas[...,0].flatten()[np.where(canvas[...,0].flatten()<np.percentile(canvas[...,0].flatten(),99.9))]
        n, bins, patches = ax[0].hist(data,bins=int(data.max()//10))
        # print('first peak: ',n.max())
        i = argrelextrema(n[n.argmax():], np.less)
        second_peak = np.max(n[i[0][0]+n.argmax():])
        # print('second peak: ',second_peak)
        peak = '2nd peak'
        peak_idx = np.argmax(n[i[0][0]+n.argmax():])+i[0][0]+n.argmax()
        if second_peak<n.max()*0.1:
            second_peak = n.max()
            peak = 'peak'
            peak_idx = n.argmax()
        
        i = argrelextrema(n[peak_idx:], np.less)
        
        third_peak = np.max(n[i[0][0]+peak_idx:])
        # print('third peak: ',third_peak)
        if third_peak > second_peak*0.1:
            second_peak = third_peak
            peak = '3rd peak'
            peak_idx = np.argmax(n[i[0][0]+n.argmax():])+i[0][0]+n.argmax()
    
        # print('final peak: ',second_peak)
        mean,std = self.cal_std(bins, n,peak_idx)
        
        
        ax[0].axhline(0.01*second_peak, c = 'k', linestyle = 'dashed')
        # +-1.5 Iqr
        clip.loc[0,"clip_min"]  = max(100,mean-4*std)#max(bins[np.where(n>0.001*second_peak)[0][0]],100)
        clip.loc[0,"clip_max"] = mean+4*std#bins[np.where(n>0.001*second_peak)[0][-1]]
        ax[0].axvline(clip.loc[0,"clip_min"], c = 'k', linestyle = 'dashed')
        ax[0].axvline( clip.loc[0,"clip_max"], c = 'k', linestyle = 'dashed')
        ax[0].text( clip.loc[0,"clip_max"]+10, second_peak*0.02, f'4*sigma')
        ax[0].text( clip.loc[0,"clip_min"]+10, second_peak*0.9, f'min clip:{round(clip.loc[0,"clip_min"])}')
        ax[0].text( clip.loc[0,"clip_max"]+10, second_peak*0.6, f'max clip:{round(clip.loc[0,"clip_max"])}')
        
        '''
        gmm = GMM(n_components = 2)
        data = data.reshape(-1, 1)
        # find useful parameters
        mean = gmm.fit(data).means_  
        covs  = gmm.fit(data).covariances_
        weights = gmm.fit(data).weights_
        y_axis0 = norm.pdf(bins, float(mean[0][0]), np.sqrt(float(covs[0][0][0])))*weights[0] # 1st gaussian
        y_axis1 = norm.pdf(bins, float(mean[1][0]), np.sqrt(float(covs[1][0][0])))*weights[0] # 2nd gaussian
    
        ax[0].plot(bins, y_axis0, lw=3, c='C0')
        ax[0].plot(bins, y_axis1, lw=3, c='C1')
        '''
        
        
        data = canvas[...,1].flatten()[np.where(canvas[...,1].flatten()<np.percentile(canvas[...,1].flatten(),99.9))]
        n, bins, patches = ax[1].hist(data,bins=int(data.max()//10))
        # print('first peak: ',n.max())
        i = argrelextrema(n[n.argmax():], np.less)
        second_peak = np.max(n[i[0][0]+n.argmax():])
        # print('second peak: ',second_peak)
        peak = '2nd peak'
        peak_idx = np.argmax(n[i[0][0]+n.argmax():])+i[0][0]+n.argmax()
        if second_peak<n.max()*0.1:
            second_peak = n.max()
            peak = 'peak'
            peak_idx = n.argmax()
        i = argrelextrema(n[peak_idx:], np.less)
        
        third_peak = np.max(n[i[0][0]+peak_idx:])
        # print('third peak: ',third_peak)
        if third_peak > second_peak*0.1:
            second_peak = third_peak
            peak = '3rd peak'
            peak_idx = np.argmax(n[i[0][0]+n.argmax():])+i[0][0]+n.argmax()
    
        # print('final peak: ',second_peak)
        mean,std = self.cal_std(bins, n,peak_idx)
        
        clip.loc[1,"clip_min"]  = max(100,mean-2.698*std)#max(bins[np.where(n>0.001*second_peak)[0][0]],100)
        clip.loc[1,"clip_max"] = mean+2.698*std# bins[np.where(n>0.001*second_peak)[0][-1]]
        ax[1].axhline(0.01*second_peak, c = 'k', linestyle = 'dashed')
        ax[1].axvline(clip.loc[1,"clip_min"], c = 'k', linestyle = 'dashed')
        ax[1].axvline( clip.loc[1,"clip_max"], c = 'k', linestyle = 'dashed')
        ax[1].text( clip.loc[1,"clip_max"]+10, second_peak*0.02, f'1.5*iqr/2.698*sigma')
        ax[1].text( clip.loc[1,"clip_min"]+10, second_peak*1.1, f'min clip:{round(clip.loc[1,"clip_min"])}')
        ax[1].text( clip.loc[1,"clip_max"]+10, second_peak*0.6, f'max clip:{round(clip.loc[1,"clip_max"])}')
        
        data = canvas[...,2].flatten()[np.where(canvas[...,2].flatten()<np.percentile(canvas[...,2].flatten(),99.9))]
        n, bins, patches = ax[2].hist(data,bins=int(data.max()//10))
        # print('first peak: ',n.max())
        i = argrelextrema(n[n.argmax():], np.less)
        second_peak = np.max(n[i[0][0]+n.argmax():])
        # print('second peak: ',second_peak)
        peak = '2nd peak'
        peak_idx = np.argmax(n[i[0][0]+n.argmax():])+i[0][0]+n.argmax()
        if second_peak<n.max()*0.1:
            second_peak = n.max()
            peak = 'peak'
            peak_idx = n.argmax()
        i = argrelextrema(n[peak_idx:], np.less)
        
        third_peak = np.max(n[i[0][0]+peak_idx:])
        # print('third peak: ',third_peak)
        if third_peak > second_peak*0.1:
            second_peak = third_peak
            peak = '3rd peak'
            peak_idx = np.argmax(n[i[0][0]+n.argmax():])+i[0][0]+n.argmax()
    
        # print('final peak: ',second_peak)
        mean,std = self.cal_std(bins, n,peak_idx)
    
        clip.loc[2,"clip_min"]  = max(100,mean-2.698*std)#max(bins[np.where(n>0.001*second_peak)[0][0]],100)
        clip.loc[2,"clip_max"] = mean+2.698*std#bins[np.where(n>0.001*second_peak)[0][-1]]
        ax[2].axhline(0.01*second_peak, c = 'k', linestyle = 'dashed')
        ax[2].axvline(clip.loc[2,"clip_min"], c = 'k', linestyle = 'dashed')
        ax[2].axvline( clip.loc[2,"clip_max"], c = 'k', linestyle = 'dashed')
        ax[2].text( clip.loc[2,"clip_max"]+10, second_peak*0.02, f'1.5*iqr/2.698*sigma')
        ax[2].text( clip.loc[2,"clip_min"]+10, second_peak*1.1, f'min clip:{round(clip.loc[2,"clip_min"])}')
        ax[2].text( clip.loc[2,"clip_max"]+10, second_peak*0.6, f'max clip:{round(clip.loc[2,"clip_max"])}')
        
        ax[0].set_title('CD45')
        ax[1].set_title('EPCAM')
        ax[2].set_title('VIM')
    
        for i in range(3):
            ax[i].set_xlabel('MIP intensity')
            ax[i].set_ylabel('count of pixels')
        print("saving hist ab... this may take awhile")
        out_path = self.ws.get_stitched_dir() / 'stitched ab hist clip value.png'
        plt.savefig(out_path)
        
        clip.to_csv(self.ws.get_stitched_dir() / 'clip_ab_raw_data.csv')
        
        print("now visualizing...")
        self.visualize_ab_only(canvas,'all')
    

    def watershed_image(self, clipped_normed_ints):
        
        image = (clipped_normed_ints[...,2]*255).astype(np.int32)
        #find an elevation map using the Sobel gradient of the image
        elevation_map = sobel(image)
        
        markers = np.zeros_like(image)
        markers[image < 20] = 1
        markers[image > 50] = 2
        vim = skimage.segmentation.watershed(elevation_map, markers)
        
        image = (clipped_normed_ints[...,1]*255).astype(np.int32)
        #find an elevation map using the Sobel gradient of the image
        elevation_map = sobel(image)
        
        markers = np.zeros_like(image)
        markers[image < 20] = 1
        markers[image > 50] = 2
        epcam = skimage.segmentation.watershed(elevation_map, markers)
        
        image = (clipped_normed_ints[...,0]*255).astype(np.int32)
        #find an elevation map using the Sobel gradient of the image
        elevation_map = sobel(image)
        
        markers = np.zeros_like(image)
        markers[image < 20] = 1
        markers[image > 50] = 2
        cd45 = skimage.segmentation.watershed(elevation_map, markers)
    
        return np.stack((cd45, epcam, vim),  axis = -1)
        
    def run_CLAHE(self, image, clipLimit):
        # The initial processing of the image
        image_bw = (clip_and_norm_image_intensity(image,lc=0,rc=np.max(image))*255).astype(np.uint8)         
        # The declaration of CLAHE
        # clipLimit -> Threshold for contrast limiting
        clahe = cv2.createCLAHE(clipLimit=clipLimit)
        final_img = clahe.apply(image_bw) 
         
        return final_img
    
    def run_equalizeHist(self, image):
        image_bw = (clip_and_norm_image_intensity(image,lc=0,rc=np.max(image))*255).astype(np.uint8)
        final_img = cv2.equalizeHist(image_bw)  
        return final_img
                
    def calculate_intensity(self, all_im_registered, fov, run_clip=False, run_CLAHE=False, run_HE=False):
        if run_clip==True:
            output_path = self.ws.get_ab_intensity_path(fov)
        else:
            output_path = self.ws.get_ab_intensity_raw_path(fov)
        #0 should be cd45, 1 should be epcam, 2 should be vim

        c = all_im_registered[...,2].copy().max(axis = 0)
        g = all_im_registered[...,1].copy().max(axis = 0)
        r = all_im_registered[...,0].copy().max(axis = 0)
        if run_clip:
            clip = pd.read_csv(self.ws.get_stitched_dir() / 'clip_ab.csv',index_col=0)
              
            c = clip_and_norm_image_intensity(c, clip.loc[2,"clip_min"], clip.loc[2,"clip_max"])# first try 4000 for 1109 and 1101, 0727, 2000 for 1121
            g = clip_and_norm_image_intensity(g, clip.loc[1,"clip_min"], clip.loc[1,"clip_max"])
            r = clip_and_norm_image_intensity(r, clip.loc[0,"clip_min"], clip.loc[0,"clip_max"])
            
        elif run_CLAHE:    
            #self.visualize_ab_only(all_im_registered,fov,smooth=False,run_clip=False)
            c = self.run_CLAHE(c,clipLimit=2)
            g = self.run_CLAHE(g,clipLimit=2)
            r = self.run_CLAHE(r,clipLimit=8)
            # Showing the two images
        elif run_HE:
            c = self.run_equalizeHist(c)
            g = self.run_equalizeHist(g)
            r = self.run_equalizeHist(r)
            # Showing the two images            
        image = np.stack((r,g,c),axis=-1)
        # self.visualize_ab_only(image,fov)
        
        '''            
        c_ = gray2rgb(c)
        g_ = gray2rgb(g)
        r_ = gray2rgb(r)
        
        c_ = c_ * [0, 1, 1]
        g_ = g_ * [0, 1, 0]
        r_ = r_ * [1, 0, 0]
        fig, ax = plt.subplots(1, 3, figsize = (24, 8))
        ax[2].imshow(c_)
        ax[1].imshow(g_)
        ax[0].imshow(r_)
        ax[0].set_title('CD45')
        ax[1].set_title('EPCAM')
        ax[2].set_title('VIM')
        
        plt.savefig(out_path)
        
        fig, ax = plt.subplots(1,3,figsize = (11,4), tight_layout=True)
        ax[0].hist(r.flatten(),bins=255)        
        ax[1].hist(g.flatten(),bins=255)
        ax[2].hist(c.flatten(),bins=255)
        ax[0].set_title('CD45')
        ax[1].set_title('EPCAM')
        ax[2].set_title('VIM')
        for i in range(3):
            ax[i].set_xlabel('MIP intensity')
            ax[i].set_ylabel('count of pixels')
        out_path = self.ws.get_ab_intensity_dir() / f'F{fov}_CLAHE ab hist.png'
        plt.savefig(out_path)
        '''
        
        clipped_normed_ints = np.stack((r, g, c),  axis = -1)
        clipped_normed_ints = np.repeat(clipped_normed_ints[None,...],75,0)  
        
        cp = self.ws.load_segmentation_masked(fov)
        
        rps = regionprops_intensity_3D(cp.astype(int), clipped_normed_ints)
        prop_names = ('volume', 'bbox_volume', 'extent', 'label', 'surface_area', 'max_perimeter','sphericity')
        props_table = []
        
        for rp in rps: 
            props = tuple((getattr(rp, prop_name) for prop_name in prop_names))
            
            centroid = getattr(rp, 'centroid')
            bbox = getattr(rp, 'bbox')
            bbox_z = (bbox[0], bbox[3]-1)
            props = props + centroid + bbox_z
            
            for attr in ['intensity_mean', 'intensity_min', 'intensity_max','intensity_percentile','intensity_median']:
                intensity_stats = getattr(rp, attr)
                props = props + tuple(intensity_stats)
                    
            props_table.append(props)    
        
        prop_names += ('centroid_z', 'centroid_r', 'centroid_c')
        prop_names += ('bbox_z_min', 'bbox_z_max')
        for attr in ['intensity_mean', 'intensity_min', 'intensity_max','intensity_percentile','intensity_median']:
            prop_names += tuple(f'{m}_{attr}' for m in ('CD45', 'EPCAM', 'VIM'))
            
        props_table = pd.DataFrame(props_table, columns = prop_names)
        props_table['fov'] = fov
        
        print(f'[calculate_intensity] Writing output to {output_path}')        
        props_table.to_csv(output_path)
        
        return props_table
    
    def calculate_intensity_expand_labels(self, all_im_registered, fov, run_clip=False, run_CLAHE=False, run_HE=False, expand_distance=10):
        output_path = self.ws.get_ab_intensity_expand_labels_path(fov, expand_distance)
        #0 should be cd45, 1 should be epcam, 2 should be vim

        c = all_im_registered[...,2].copy().max(axis = 0)
        g = all_im_registered[...,1].copy().max(axis = 0)
        r = all_im_registered[...,0].copy().max(axis = 0)
        if run_clip:
            clip = pd.read_csv(self.ws.get_stitched_dir() / 'clip_ab.csv',index_col=0)
              
            c = clip_and_norm_image_intensity(c, clip.loc[2,"clip_min"], clip.loc[2,"clip_max"])# first try 4000 for 1109 and 1101, 0727, 2000 for 1121
            g = clip_and_norm_image_intensity(g, clip.loc[1,"clip_min"], clip.loc[1,"clip_max"])
            r = clip_and_norm_image_intensity(r, clip.loc[0,"clip_min"], clip.loc[0,"clip_max"])
            
        elif run_CLAHE:    
            #self.visualize_ab_only(all_im_registered,fov,smooth=False,run_clip=False)
            c = self.run_CLAHE(c,clipLimit=2)
            g = self.run_CLAHE(g,clipLimit=2)
            r = self.run_CLAHE(r,clipLimit=8)
            # Showing the two images
        elif run_HE:
            c = self.run_equalizeHist(c)
            g = self.run_equalizeHist(g)
            r = self.run_equalizeHist(r)
            # Showing the two images            
        image = np.stack((r,g,c),axis=-1)
        #self.visualize_ab_only(image,fov)
        
        clipped_normed_ints = np.stack((r, g, c),  axis = -1)
        clipped_normed_ints = np.repeat(clipped_normed_ints[None,...],75,0)  
        
        ## here we expand labels, with an expansion distance: calculate by plane
        cp_raw = self.ws.load_segmentation_masked(fov)
        cp = np.zeros(cp_raw.shape, dtype='int')
        for plane in range(cp_raw.shape[0]):
            cp[plane, :, :] = expand_labels(cp_raw[plane, :, :], distance = expand_distance)
        
        
        rps = regionprops_intensity_3D(cp.astype(int), clipped_normed_ints)
        prop_names = ('volume', 'bbox_volume', 'extent', 'label', 'surface_area', 'max_perimeter','sphericity')
        props_table = []
        
        for rp in rps: 
            props = tuple((getattr(rp, prop_name) for prop_name in prop_names))
            
            centroid = getattr(rp, 'centroid')
            bbox = getattr(rp, 'bbox')
            bbox_z = (bbox[0], bbox[3]-1)
            props = props + centroid + bbox_z
            
            for attr in ['intensity_mean', 'intensity_min', 'intensity_max','intensity_percentile','intensity_median']:
                intensity_stats = getattr(rp, attr)
                props = props + tuple(intensity_stats)
                    
            props_table.append(props)    
        
        prop_names += ('centroid_z', 'centroid_r', 'centroid_c')
        prop_names += ('bbox_z_min', 'bbox_z_max')
        for attr in ['intensity_mean', 'intensity_min', 'intensity_max','intensity_percentile','intensity_median']:
            prop_names += tuple(f'{m}_{attr}' for m in ('CD45', 'EPCAM', 'VIM'))
            
        props_table = pd.DataFrame(props_table, columns = prop_names)
        props_table['fov'] = fov
        
        print(f'[calculate_intensity] Writing output to {output_path}')        
        props_table.to_csv(output_path)
        
        return props_table   

###############
###############
###############
        
if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', default = 'bandpass')
    parser.add_argument('--data_id', default = '1234')
    parser.add_argument('--data_keyword', default = '')
    parser.add_argument('--seg_model', default = 'cellpose_modelD')
    parser.add_argument('-p', '--num_workers', default = 8, type = int)
    parser.add_argument('-n', '--run_notes', default = 'run_notes cidre_120_multi_fdr')
    parser.add_argument('--stitch_coordinates_pattern', default=None,
                        help="Optional glob (in the stitching-shift dir) for the "
                             "master coordinate array. Default: 'Master_coord_array*.npy'. "
                             "Use this to disambiguate if multiple shifts files exist.")
    args = parser.parse_args() 

    # load params
    ws = Workspace(args)
    fovs = ws.load_fovs()
    args.fovs_to_process = fovs.flatten()

    ar = AbRegistration(ws, args.fovs_to_process)
    ar()
