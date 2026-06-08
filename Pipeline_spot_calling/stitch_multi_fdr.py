# automatically stitch results for visualization

import numpy as np
import pandas as pd
import zarr

from skimage.registration import phase_cross_correlation
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk
from skimage.color import gray2rgb, label2rgb 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors
import scipy.stats
from pandas.api.types import CategoricalDtype
from itertools import combinations
from scipy.spatial.distance import cdist, squareform

from workspace import Workspace
from utils import _3D_translation, find_stitching_shifts

import h5py
from pathlib import Path
import skimage
import glob

import argparse
import os
from skimage.io import imsave

def clip_and_norm_image(zf, lp = 50, rp = 99.9): 
    zf = zf.copy()
    lq, rq = np.percentile(zf, [lp, rp])
    zf = (zf - lq) / (rq - lq)
    zf[zf < 0] = 0
    zf[zf > 1] = 1
    return(zf)

def discrete_cmap(N, base_cmap = None): 
    """Create an N-bin discrete colormap from the specified input map"""

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

class Stitcher: 

    def __init__(self, 
        ws, 
        fovs_to_process
    ): 

        self.ws = ws
        self.fovs_to_process = np.array(fovs_to_process)
        self.fovs_flat = self.fovs_to_process.flatten()

        # -- check stitched directory
        out_dir = self.ws.get_stitched_dir()
        if not os.path.isdir(out_dir): 
            os.makedirs(out_dir)
        
        # -- fixed params
        self.fov_size = 2048 #1024
        self.margin = int(0.2*self.fov_size)
        self.shift_constant = int(0.095*self.fov_size)
        self.iz = 35 # should be automatically determined

        # -- plotting params
        self.cls = list(mcolors.TABLEAU_COLORS.values())*15
        
        # initialize cm for overlay 
        cm = discrete_cmap(7, base_cmap = 'YlOrRd')#autumn
        self.cm_dict = dict(zip(range(4,11), [cm(i)[:3] for i in range(7)]))
        self.cm_dict[0] = colors.to_rgba('darkblue')[:3]
        self.cm_dict[1] = colors.to_rgba('royalblue')[:3]
        self.cm_dict[2] = colors.to_rgba('green')[:3]
        self.cm_dict[3] = colors.to_rgba('mediumseagreen')[:3]

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
            print(f'[stitcher] loading stitching shifts from {shifts_path.name}')
            self.shifts = np.load(shifts_path).astype(int)
            print(self.shifts.shape)
        else:
            self.stitch_dapi()        

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

    def stitch_dapi(self): 
        # -- load dapi 
        dapis = {}
        for fov in self.fovs_flat: 
            dapis[fov] = self.ws.load_prehyb_registered(fov, 'dapi').max(axis = 0)

        # -- prepare canvas
        canvas = np.zeros(((
            self.fov_size*self.fovs_to_process.shape[0]), 
            self.fov_size*self.fovs_to_process.shape[1]))
        shifts = np.array([[(0, 0) for fov in fovs] for fovs in self.fovs_to_process])

        # -- stitch
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]): 

                if j == 0:
                    j_shift_ = (0, 0)
                    j_shift = 0
                else: 
                    reference_shift = shifts[i, j-1]
                    reference_image = canvas[
                        reference_shift[0]:reference_shift[0]+self.fov_size,
                        reference_shift[1]+self.fov_size-self.margin:reference_shift[1]+self.fov_size
                    ].copy() 
                    moving_image = dapis[self.fovs_to_process[i][j]][:,:self.margin].copy()
                    j_shift_, _, _ = phase_cross_correlation(
                        reference_image, 
                        moving_image, 
                    )
                    if np.abs(j_shift_[1] / self.fov_size - 0.1) > 0.01:
                        j_shift_ = (0, self.shift_constant)
                    j_shift_ = tuple(j_shift_)
                    j_shift = int(reference_shift[1] + self.fov_size - j_shift_[1])

                if i == 0:
                    i_shift_ = (0, 0)
                    i_shift = 0
                else:
                    reference_shift = shifts[i-1,j]
                    reference_image = canvas[
                        reference_shift[0]+self.fov_size-self.margin:reference_shift[0]+self.fov_size,
                        reference_shift[1]:reference_shift[1]+self.fov_size
                    ].copy()
                    moving_image = dapis[self.fovs_to_process[i][j]][:self.margin,:].copy()
                    i_shift_, _, _ = phase_cross_correlation(
                        reference_image, 
                        moving_image, 
                    )
                    i_shift_ = tuple(i_shift_)
                    if np.abs(i_shift_[0] / self.fov_size - 0.1) > 0.01: 
                        i_shift_ = (self.shift_constant, 0)
                    i_shift = int(reference_shift[0] + self.fov_size - i_shift_[0])

                # print(self.fovs_to_process[i,j], i, j, i_shift_, j_shift_, i_shift, j_shift)
                
                shifts[i,j] = (i_shift, j_shift)
                canvas[i_shift:i_shift+self.fov_size, j_shift:j_shift+self.fov_size] = np.stack((
                    canvas[i_shift:i_shift+self.fov_size,j_shift:j_shift+self.fov_size], 
                    dapis[self.fovs_to_process[i][j]])
                ).max(axis = 0)

        self.shifts = shifts # save shifts for other things
                
        fig, ax = plt.subplots(figsize = 
            (8*self.fovs_to_process.shape[1], 8*self.fovs_to_process.shape[0]))
        plt.imshow(canvas, cmap = plt.cm.gray)
        plt.axis('off')

        k = 0
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]): 
                i_shift, j_shift =  shifts[i,j]
                y = [i_shift, i_shift, i_shift+self.fov_size, i_shift+self.fov_size, i_shift]
                x = [j_shift, j_shift+self.fov_size, j_shift+self.fov_size, j_shift, j_shift]
                plt.plot(x, y, c = self.cls[k])

                fov = self.fovs_to_process[i][j]
                plt.text(j_shift+75, i_shift+150, fov, c = self.cls[k], size = 32)        
                k += 1
        
        out_path = self.ws.get_stitched_dir() / 'dapi.png'
        plt.savefig(out_path)
        
        with zarr.open(self.ws.get_stitched_dir() / 'shifts.zarr', 'w') as f: 
            f.create_dataset('shifts', data = shifts)

    def stitch_bit(self, bit): 
        # load images
        ims = {}
        for fov in self.fovs_flat: 
            cp = self.ws.load_segmentation_masked(fov)
            
            if len(cp.shape) == 3:
                cp = cp[self.iz,...]
                im = clip_and_norm_image(self.ws.load_hyb_registered(fov, bit, self.iz))
            elif len(cp.shape) == 2:
                im = clip_and_norm_image(self.ws.load_hyb_registered(fov, bit).max(axis = 0))
            else:
                raise NotImplementedError
            bd = find_boundaries(cp)
            bd = binary_dilation(bd, disk(1))
            im = gray2rgb(im)
            im[:,:,0] = np.where(bd, 1, im[:,:,0])
            im[:,:,1] = np.where(bd, 0, im[:,:,1])
            im[:,:,2] = np.where(bd, 0, im[:,:,2])
            ims[fov] = im

        self.stitch_images(ims, bit, f'bit{bit}_hyb', callout = 'o')

    def stitch_bit_overlay(self, bit,condition): 
        passed = pd.read_csv(self.ws.get_segmentation_qc_dir() / f'{condition}.csv', index_col = 0)

        ims = {}
        for fov in self.fovs_flat: 
            if fov == 'xxx':
                continue
            coords = self.ws.get_compiled_peaks(fov, bit)
            fov_passed = passed.loc[passed['fov'] == int(fov)]
            cells_passed = fov_passed.loc[fov_passed[condition]].label
            num_spots_per_cell = coords['cell_id'].value_counts().reindex(
                cells_passed).fillna(0).astype(int)

            param_idx = dict(zip(num_spots_per_cell.index, num_spots_per_cell.values))
            def relabel(x): 
                return param_idx[x] if x in param_idx else -1
            f = lambda x: relabel(x)
            vf = np.vectorize(f) 

            cp = self.ws.load_segmentation_masked(fov)
       
            if len(cp.shape) == 3:
                cp = cp[self.iz,...]
            bd = find_boundaries(cp)
            bd = binary_dilation(bd, disk(1))
    
            key_overlay = vf(cp)
            key_overlay[key_overlay > 9] = 9
            #cm_list = [self.cm_dict[lb] for lb in np.unique(key_overlay.flatten()) if lb != -1]
            cm_list = [self.cm_dict[lb] for lb in np.arange(10) if lb != -1]

            im = label2rgb(label = key_overlay, 
                    colors = cm_list, 
                    bg_label = -1, 
                    bg_color = 'black')
 
            im[:,:,0] = np.where(bd, 1, im[:,:,0])
            im[:,:,1] = np.where(bd, 1, im[:,:,1])
            im[:,:,2] = np.where(bd, 1, im[:,:,2])

            ims[fov] = im

        self.stitch_images(ims, bit, f'bit{bit}_overlay', callout = '+')
        
    def stitch_ab(self):
        # -- prepare canvas
        canvas = np.zeros(self._canvas_shape(4))
        print("stitch ab...")
        
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]): 
                fov = self.fovs_to_process[i][j]
                if fov=='xxx':
                    continue
                j_shift, i_shift, _ =  self.shifts[i,j]
                # -- load ab 
                ab_path = glob.glob(os.path.join(self.ws.data_path, f"ab/*F{fov}.tif"))[0]
                ab = skimage.io.imread(ab_path)
                shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')
                ms = []
                for ci in range(3): 
                    # 0 should be cd45, 1 should be epcam, 2 should be vim, 3 should be dapi
                    shifts = shifts_.filter(regex = f"ab_fov_{fov}_cycle_.*_ch_{ci}", axis = 0)
                    shifts = (int(shifts.z), int(shifts.y), int(shifts.x))
                    m = _3D_translation(ab[...,ci], shifts).astype(np.uint16)
                    ms.append(m.max(axis = 0))
                ms = np.stack(ms, axis = -1)
                    
                canvas[canvas.shape[0]-i_shift-self.fov_size - min(shifts_.y):canvas.shape[0]-i_shift, \
                       j_shift- min(shifts_.x):j_shift+self.fov_size] = np.stack((
                    canvas[canvas.shape[0]-i_shift-self.fov_size - min(shifts_.y):canvas.shape[0]-i_shift,\
                           j_shift- min(shifts_.x):j_shift+self.fov_size], 
                    ms[-min(shifts_.y):,-min(shifts_.x):])).max(axis=0)
        fig, ax = plt.subplots(figsize = 
            (8*self.fovs_to_process.shape[1], 8*self.fovs_to_process.shape[0]))
        plt.imshow(canvas, cmap = plt.cm.gray)
        plt.axis('off')
        
        out_path = self.ws.get_stitched_dir() / 'stitched ab.png'
        plt.savefig(out_path)                   
                           
        fig, ax = plt.subplots(1,3,figsize = (12,5), tight_layout=True)
        ax[0].hist(canvas[...,0].flatten(),bins=300)
        ax[1].hist(canvas[...,1].flatten(),bins=300)
        ax[2].hist(canvas[...,2].flatten(),bins=300)
        print("saving hist ab...")
        out_path = self.ws.get_stitched_dir() / 'stitched ab hist.png'
        plt.savefig(out_path)

    
    def stitch_bit_celltype(self, bit, metrics): 
        mean_intensities = pd.read_csv(self.ws.get_stitched_dir() / f'hierarchy_celltyping_{metrics}.csv')

        ims = {}
        for fov in self.fovs_flat: 
            if fov=='xxx':
                continue
            fov_mean_intensities = mean_intensities.loc[mean_intensities['fov'] == int(fov)]

            param_idx = dict(zip(fov_mean_intensities.label, fov_mean_intensities.celltype))
            def relabel(x): 
                return param_idx[x] if x in param_idx else -1
            f = lambda x: relabel(x)
            vf = np.vectorize(f) 

            cp = self.ws.load_segmentation_masked(fov)
       
            if len(cp.shape) == 3:
                cp = cp[self.iz,...]
            bd = find_boundaries(cp)
            bd = binary_dilation(bd, disk(1))
    
            key_overlay = vf(cp)
            
            #cm = discrete_cmap(8, base_cmap = 'jet')
            cm = plt.cm.get_cmap("tab10")
            cm_dict = dict(zip(range(1,10), [cm(i)[:3] for i in range(9)]))
            cm_dict[0] = colors.to_rgba('black')[:3]
            
            cm_list = [cm_dict[lb] for lb in np.unique(key_overlay.flatten()) if lb != -1]

            im = label2rgb(label = key_overlay, 
                    colors = cm_list, 
                    bg_label = -1, 
                    bg_color = 'black')
 
            im[:,:,0] = np.where(bd, 1, im[:,:,0])
            im[:,:,1] = np.where(bd, 1, im[:,:,1])
            im[:,:,2] = np.where(bd, 1, im[:,:,2])

            ims[fov] = im

        self.stitch_images(ims, bit, f'bit{bit}_celltype_{metrics}', callout = '+')

    def stitch_image_helper(self, ims): 
        canvas = np.zeros(self._canvas_shape())

        # assemble image
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]):  
                fov = self.fovs_to_process[i][j]
                if fov=='xxx':
                    continue
                j_shift, i_shift, _ = self.shifts[i, j]
                shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')
                canvas[canvas.shape[0]-i_shift-self.fov_size - min(shifts_.y):canvas.shape[0]-i_shift, \
                       j_shift- min(shifts_.x):j_shift+self.fov_size] = np.stack((
                    canvas[canvas.shape[0]-i_shift-self.fov_size - min(shifts_.y):canvas.shape[0]-i_shift,\
                           j_shift- min(shifts_.x):j_shift+self.fov_size], 
                    ims[self.fovs_to_process[i][j]][-min(shifts_.y):,-min(shifts_.x):]
                )).max(axis = 0)
        return canvas
    
    def stitch_images(self, ims, bit, out_name, plot_callouts = False, callout = 'o'): 
        canvas = np.zeros(self._canvas_shape(3))
        
        # assemble image
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]): 
                fov = self.fovs_to_process[i][j]
                if fov=='xxx':
                    continue
                j_shift, i_shift, _ = self.shifts[i, j]
                shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')
                canvas[canvas.shape[0]-i_shift-self.fov_size - min(shifts_.y):canvas.shape[0]-i_shift, \
                       j_shift- min(shifts_.x):j_shift+self.fov_size] = np.stack((
                    canvas[canvas.shape[0]-i_shift-self.fov_size - min(shifts_.y):canvas.shape[0]-i_shift,\
                           j_shift- min(shifts_.x):j_shift+self.fov_size], 
                    ims[self.fovs_to_process[i][j]][-min(shifts_.y):,-min(shifts_.x):]
                )).max(axis = 0)
        
        # mask out the extra canvas
        canvas[:,j_shift+self.fov_size:] = 1
        canvas[0:canvas.shape[0]-self.shifts[0, 0][1]-self.fov_size,:] = 1

        fig, ax = plt.subplots(figsize = 
            (8*self.fovs_to_process.shape[1], 8*self.fovs_to_process.shape[0]))
        plt.imshow(canvas, cmap = plt.cm.gray)
        plt.axis('off')

        # plot fovs
        k = 0
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]): 
                # plot boxes
                
                fov = self.fovs_to_process[i][j]
                if fov=='xxx':
                    continue
                j_shift, i_shift, _ =  self.shifts[i,j]
                shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')
                y = [canvas.shape[0]-i_shift, canvas.shape[0]-i_shift, \
                     canvas.shape[0]-i_shift-self.fov_size - min(shifts_.y), \
                         canvas.shape[0]-i_shift-self.fov_size- min(shifts_.y), canvas.shape[0]-i_shift]
                x = [j_shift- min(shifts_.x), j_shift+self.fov_size, j_shift+self.fov_size, \
                     j_shift-min(shifts_.x), j_shift - min(shifts_.x)]
                plt.plot(x, y, c = self.cls[k])

                fov = self.fovs_to_process[i][j]
                plt.text(j_shift+75, canvas.shape[0]-i_shift-1800, fov, c = self.cls[k], size = 32)        
                
                k += 1
        
        out_path = self.ws.get_stitched_dir() / f'{out_name}.png'
        plt.savefig(out_path)
        
        if plot_callouts:
            # plot callouts
            for i in range(self.fovs_to_process.shape[0]): 
                for j in range(self.fovs_to_process.shape[1]): 
                    fov = self.fovs_to_process[i][j]
                    if fov=='xxx':
                        continue
                    j_shift, i_shift, _ = self.shifts[i,j]
                    shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')
                    
                    # plot callouts
                    coords = self.ws.get_stitched_peaks(fov, bit)
                    if callout == 'o':
                        plt.scatter(coords['c'] + j_shift - min(shifts_.x), \
                                    canvas.shape[0] - (self.fov_size - coords['r'] + i_shift - min(shifts_.y)), 
                            facecolor = "none", edgecolor = 'b', s = 30)
                    elif callout == '+': 
                        plt.scatter(coords['c'] + j_shift - min(shifts_.x), \
                                    canvas.shape[0] - (self.fov_size - coords['r'] + i_shift - min(shifts_.y)), 
                            color = 'white', marker = '+')
            
            out_path = self.ws.get_stitched_dir() / f'{out_name}_callouts.png'
            plt.savefig(out_path)
        
    def plot_coords(self,bit,callout='o'):
        canvas = np.zeros(self._canvas_shape(3))
        
        fig, ax = plt.subplots(figsize = 
            (self.fovs_to_process.shape[1], self.fovs_to_process.shape[0]))
        plt.imshow(canvas, cmap = plt.cm.gray)
        plt.axis('off')
        
        # plot callouts
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]): 
                fov = self.fovs_to_process[i][j]
                if fov=='xxx':
                    continue
                j_shift, i_shift, _ = self.shifts[i,j]
                shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')
                # plot callouts
                coords = self.ws.get_stitched_peaks(fov, bit)
                #print(coords['r'])
                if callout == 'o':
                    plt.scatter(coords['c'] + j_shift - min(shifts_.x), \
                                canvas.shape[0] - (self.fov_size - coords['r'] + i_shift - min(shifts_.y)), 
                        facecolor = "none", edgecolor = 'r', s = 1)
                elif callout == '+': 
                    plt.scatter(coords['c'] + j_shift - min(shifts_.x), \
                                canvas.shape[0] - (self.fov_size - coords['r'] + i_shift - min(shifts_.y)), 
                        color = 'k', marker = '+')
        
        plt.show()
    
    def stitch_fov(self, filter_pad):
        passed = pd.read_csv(self.ws.get_segmentation_qc_dir() / 'passed.csv',index_col=0)
        passed['passed_top'] = True
        passed['passed_left'] = True
        passed['passed_bottom'] = True
        passed['passed_right'] = True
        passed['passed_stitched'] = True
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]): 
                fov = self.fovs_to_process[i][j]
                if fov=='xxx':
                    continue
                j_shift, i_shift, _ =  self.shifts[i,j]
                
                shifts_ = self.ws.load_precomputed_shifts(fov).set_index('tar')
                
                passed_sub = passed[passed.fov==int(fov)].copy()
                cell_info = pd.read_csv(self.ws.get_segmentation_qc_dir() / f"{fov}.csv",index_col=0)
            
                # Policy of stiching cells, for the completed cells in the overlapped region, always keep them at right/bottom FOVs, 
                # For imcompleted cells at the left/top boundary, keep the completed cells at left/ top FOVs
                # for each FOV:
                if i > 0: # The FOV not at the first row, there are the overlapped region with the FOV on its top
                    # Remove the incomplete cells at the top (cells touching the top boundary)
                    passed_sub['passed_top'] = cell_info.centroid_r - cell_info.bbox_r/2  > (-min(shifts_.y) + filter_pad)
                    
                if j > 0: # The FOV not at the first col, there are the overlapped region with the FOV on its left
                    # Remove the incomplete cells at the left (cells touching the left boundary)
                    passed_sub['passed_left'] = cell_info.centroid_c - cell_info.bbox_c/2  > (-min(shifts_.x) + filter_pad)
                
                cell_info.centroid_r = self.fov_size + i_shift - cell_info.centroid_r
                cell_info.centroid_c += j_shift
                    
                if i < self.fovs_to_process.shape[0]-1: # The FOV not at the last row, there are the overlapped region with the FOV on its bottom
                    # remove the completed cells of the top FOV at the overlapped region
                    _, i_shift_bottom, _ =  self.shifts[i+1,j]
                    passed_sub['passed_bottom'] = (cell_info.centroid_r + cell_info.bbox_r/2) > i_shift_bottom + self.fov_size + min(shifts_.y) - filter_pad
                    
                if j < self.fovs_to_process.shape[1]-1: # The FOV not at the lasst col, there are the overlapped region with the FOV on its right
                    # remove the completed cells of the left FOV at the overlapped region
                    j_shift_left, _ , _ =  self.shifts[i,j+1]
                    passed_sub['passed_right'] = (cell_info.centroid_c - cell_info.bbox_c/2) < j_shift_left - min(shifts_.x) + filter_pad
            
                passed_sub['passed_stitched'] = passed_sub[['passed','passed_top','passed_left','passed_bottom','passed_right']].all(axis = 1)
                passed.loc[passed.fov==int(fov),['passed','passed_top','passed_left','passed_bottom','passed_right','passed_stitched']] = \
                    passed_sub[['passed','passed_top','passed_left','passed_bottom','passed_right','passed_stitched']]
                        
        out_dir = self.ws.get_segmentation_qc_dir()
        passed.to_csv(out_dir / 'passed_stitched.csv')
        
    def stitch_coords(self, bit, fdr_vals = [0.1, 0.05, 0.01]):
        condition = 'passed_stitched'
        passed = pd.read_csv(self.ws.get_segmentation_qc_dir() / f'{condition}.csv', index_col = 0)
        for i in range(self.fovs_to_process.shape[0]): 
            for j in range(self.fovs_to_process.shape[1]):                 
                fov = self.fovs_to_process[i][j]
                if fov=='xxx':
                    continue
                fov_passed = passed.loc[passed['fov'] == int(fov)]
                
                cell_passed = fov_passed.label[fov_passed[condition]].values
                
                for fdr_val in fdr_vals:
                    coords = self.ws.get_compiled_peaks(fov, bit, fdr_val)
                    coords_ = coords[coords.cell_id.isin(cell_passed)]
                    
                    out_path = self.ws.get_stitched_peaks_path(bit, fov, fdr_val)
                    
                    out_path.parent.mkdir(parents = True, exist_ok = True)
                    print(f'[_stitch_peaks_bit] writing stitched peaks to {out_path}')
                    with h5py.File(out_path, 'w') as f: 
                        f.create_dataset('coords', data = coords_[['z', 'r', 'c']].values)
                        f.create_dataset('intensities', data = coords_['intensities'].values)
                        f.create_dataset('volume', data = coords_['volume'].values)
                        f.create_dataset('intden', data = coords_['intden'].values)
                        f.create_dataset('cell_id', data = coords_['cell_id'].values)
        
        
    def stitch_segmentation(self, fov, condition):
        passed = pd.read_csv(self.ws.get_segmentation_qc_dir() / f'{condition}.csv', index_col = 0)
        # load images 
        cp = self.ws.load_segmentation_masked(fov)
        
        fov_passed = passed.loc[passed['fov'] == int(fov)]
        
        remove_idx = dict(zip(fov_passed.label, ~fov_passed[condition]&fov_passed['passed']))
        remove_idx[0] = False

        f = lambda x: remove_idx[x]
        vf = np.vectorize(f)
        seg_mask = vf(cp)
        
        seg_stitched = np.where(seg_mask, 0, cp)
        output_path = self.ws.get_segmentation_stitched_path(fov)
        print(f'[stitch_segmentation] Writing output to {output_path}.')
        imsave(output_path, seg_stitched) 
        
    def count_spot(self,condition, fdr_val):
        passed = pd.read_csv(self.ws.get_segmentation_qc_dir() / f'{condition}.csv',index_col=0)

        num_spots = []
        coords_ = []

        for fov in self.fovs_flat: 
            if fov=='xxx':
                continue
            fov_passed = passed.loc[passed['fov'] == int(fov)]
            cells_passed = fov_passed.loc[fov_passed[condition]].label
            for bit in self.ws.bits.index:
                coords = self.ws.get_stitched_peaks(fov, bit, fdr_val)
                coords['spot_id'] = np.arange(1,len(coords)+1)
                coords_.append(coords)
                num_spots_per_cell = coords['cell_id'].value_counts().reindex(
                    cells_passed).fillna(0).astype(int)
                num_spots_per_cell = pd.DataFrame({'n': num_spots_per_cell, "bit": bit, "fov": fov})
                num_spots_per_cell['cell_id'] = num_spots_per_cell.index
                num_spots_per_cell.index = [f'FOV{fov}_X{c}' for c in num_spots_per_cell.index]
                num_spots.append(num_spots_per_cell)
            
        num_spots = pd.concat(num_spots)
        coords_ = pd.concat(coords_)
        coords_['bit_cell_spot'] = [f'{c.bit}_{c.cell_id}_{c.spot_id}' for _, c in coords_.iterrows()]
        
        return num_spots, coords_
    
    def visualize_total_counts(self, num_spots, fdr_val):
        total_num_spots = num_spots.groupby(['fov', 'cell_id'])['n'].sum()
        # print(total_num_spots.index)

        ims = {}

        for fov in self.fovs_flat: 
            if fov=='xxx':
                continue
            fov_total_num_spots = total_num_spots.loc[fov]
            param_idx = dict(zip(fov_total_num_spots.index, fov_total_num_spots.values))
            
            def relabel(x): 
                return param_idx[x] if x in param_idx else -1
            f = lambda x: relabel(x)
            vf = np.vectorize(f) 
            
            cp = self.ws.load_segmentation_masked(fov)
            if len(cp.shape) == 3:
                cp = cp[10,...]
            bd = find_boundaries(cp)
            bd = binary_dilation(bd, disk(1))
            
            key_overlay = vf(cp)
        
            ims[fov] = key_overlay
        
        stitched = self.stitch_image_helper(ims)

        fig, ax = plt.subplots(figsize = (self.fovs_to_process.shape[1], self.fovs_to_process.shape[0]))
        im = ax.imshow(stitched, vmax = 300)
        ax.axis('off')
        fig.colorbar(im, ax = ax, shrink = 0.9)
        plt.savefig(str(self.ws.get_stitched_dir()) + '/total number of spots_fdr_' + str(fdr_val) + '.png', dpi=300)
    
    def filter_total_counts(self, num_spots, fdr_val):
        probes = pd.read_csv(Path(self.ws.params['data_path']) / 'mCNV-FISH_gene inseq_Dec2023KMS.csv')
        gene_order = probes['Gene'].values
        cat_type = CategoricalDtype(categories=gene_order, ordered=True)
        num_spots_df = pd.merge(num_spots, self.ws.bits, left_on = 'bit', right_index = True)
        num_spots_df['genes'] = num_spots_df['genes'].astype(cat_type)
        num_spots_df['chrom_list'] = num_spots_df['chrom_list'].astype(str)
        
        total_num_spots = num_spots.groupby(['fov', 'cell_id'])['n'].sum()
        q1, q2, q3 = np.quantile(np.log10(total_num_spots+1), [0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_bound = 10**(q1-1.5*iqr)
        upper_bound = 10**(q3+1.5*iqr)
        
        mad = scipy.stats.median_abs_deviation(np.log10(total_num_spots+1))
        lower_bound = 10**(q2 - 3*mad)
        upper_bound = 10**(q2 + 3*mad)
        
        plt.figure(figsize = (3,3))
        plt.hist(total_num_spots, bins = 50)
        plt.axvline(lower_bound, c = 'k', linestyle = 'dashed')
        plt.axvline(upper_bound, c = 'k', linestyle = 'dashed')
        plt.xlabel('total counts per cell')
        plt.ylabel("# of cells")
        plt.tight_layout()
        plt.savefig(str(self.ws.get_stitched_dir()) + '/distribution and filtering of total number of spots_fdr_' + str(fdr_val) + '.png', dpi=300)
        
        select_cells = ((total_num_spots > lower_bound) & (total_num_spots < upper_bound))
        select_cells_ix = total_num_spots.loc[select_cells].index
        select_cells_ixn = [f'FOV{fov}_X{cell_id}' for fov, cell_id in select_cells_ix]
        num_spots_df['keep'] = [ci in select_cells_ixn for ci in num_spots_df.index]
        num_spots_df.to_csv(str(self.ws.get_stitched_dir()) + "/num_spots_fdr_" + str(fdr_val) + ".csv")
        
        passed = pd.read_csv(Path(self.ws.get_segmentation_qc_dir()) / 'passed_stitched.csv',index_col=0)       
        
        passed['passed_filtered'] = False

        for fov, cell_id in select_cells_ix:   
            passed.loc[(passed.fov==int(fov))&(passed.label==cell_id),['passed_filtered']] = True   
        
        out_dir = self.ws.get_segmentation_qc_dir()
        passed.to_csv(str(out_dir) + '/passed_filtered_fdr_' + str(fdr_val) + '.csv')
        
    def match_spot(self, coords_, fov, cell_id, cell_spots):
        matched_spots = []
        
        if cell_id%10 == 0:
            print(f'[Counting matched_spots] for {fov} and {cell_id}, downsampled randomly every 10 cells.')
            for bit_a, bit_b in combinations(self.ws.bits.index, 2): 
            
                spots_a = cell_spots.loc[cell_spots['bit'] == bit_a]
                spots_b = cell_spots.loc[cell_spots['bit'] == bit_b]
                
                cd = cdist(spots_a[['r_micron', 'c_micron', 'z_micron']], spots_b[['r_micron', 'c_micron', 'z_micron']])
                cd = pd.DataFrame(cd)
                cd.index = spots_a['bit_cell_spot'].values
                cd.columns = spots_b['bit_cell_spot'].values
                
                cd_ = cd.copy()
                
                while (cd_.shape[0] > 0) & (cd_.shape[1] > 0): 
                    ind = np.unravel_index(np.argmin(cd_.values, axis=None), cd_.shape)
                
                    v = cd_.iloc[ind[0], ind[1]]
                    spot_aa = cd_.index[ind[0]]
                    spot_bb = cd_.columns[ind[1]]
                
                    cd_ = cd_.drop(spot_aa).drop(spot_bb, axis = 1)
                
                    matched_spots.append((fov, cell_id, bit_a, bit_b, spot_aa, spot_bb, v))
            return pd.DataFrame(matched_spots)
                
###############
###############
###############

if __name__ == "__main__": 

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_id')
    parser.add_argument('--data_keyword', default = '')
    parser.add_argument('-r', '--run_type', default = 'bandpass')
    parser.add_argument('-s', '--seg_model')
    parser.add_argument('--stitch_coordinates_pattern', default=None,
                        help="Optional glob (in the stitching-shift dir) for the "
                             "master coordinate array. Default: 'Master_coord_array*.npy'. "
                             "Use this to disambiguate if multiple shifts files exist.")
    parser.add_argument('--stitch_bit', type = int)

    args = parser.parse_args()

    ws = Workspace(args)

    st = Stitcher(ws, fovs_to_process)
    if args.stitch_bit is not None:
        st.stitch_bit(args.stitch_bit)
        st.stitch_bit_overlay(args.stitch_bit)

