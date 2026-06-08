# postprocess segmentation

import numpy as np
import pandas as pd

import scipy.ndimage as spim 
from skimage.io import imsave
from skimage.measure import mesh_surface_area, marching_cubes, perimeter
from skimage.morphology import ball
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from workspace import Workspace

from pathlib import Path
from types import SimpleNamespace

import argparse
import os

class RegionProperties2D(RegionProperties): 

    @property
    def circularity(self): 
        return (4 * np.pi * self.area) / (self.perimeter**2) 

    @property
    def roundness(self): 
        return (4 * self.area) / (np.pi * self.axis_major_length**2)

    @property
    def background_intensity(self): 
        # defined as the mean up to the 50th percentile of pixels 
        it = self.image_intensity[self.image]
        # calculate for each channel
        assert len(it.shape) == 2 # expect pixels x channels
        b = []
        for i in range(it.shape[1]): # for each channel
            it_ = it[:,i]
            it_ = it_[it_ < np.median(it_)] # subset up to 50th percentile of pixels
            b_ = np.mean(it_) # get mean of remainder
            b.append(b_)
        return np.array(b)

def regionprops_intensity_2D(label_image, intensity_image): 
    results = regionprops(label_image, intensity_image)
    for i, obj in enumerate(results): 
        a = results[i]
        b = RegionProperties2D(a.slice, 
                               a.label, 
                               a._label_image, 
                               a._intensity_image, 
                               a._cache_active)
        results[i] = b
    return(results)

def summarize_2d(ws, fov): 
    output_path = ws.get_segmentation_qc_path(fov)
    if os.path.exists(output_path): 
        print(f'[summarize_2d] Summary exists at {output_path}. Loading...')
        return pd.read_csv(output_path, dtype={'fov': str})

    seg = ws.load_segmentation(fov) 

    # load prehybs
    prehyb_cy3_rg = ws.load_prehyb_registered(fov, 'Cy3')
    prehyb_cy5_rg = ws.load_prehyb_registered(fov, 'Cy5')
    signal = np.stack((prehyb_cy3_rg, prehyb_cy5_rg), axis = -1).max(axis = 0)

    # calculate properties
    prop_names = ('area', 'area_bbox', 'extent', 'label', 
        'perimeter', 'roundness', 'circularity')
    
    rps = regionprops_intensity_2D(
        seg.astype(int), 
        signal
    )

    props_table = []
    for rp in rps: 

        centroid = getattr(rp, 'centroid')
        bbox_shape = getattr(rp, 'image').shape
        intensity_mean = tuple(getattr(rp, 'intensity_mean'))

        props = tuple((getattr(rp, prop_name) for prop_name in prop_names))
        props = props + centroid + bbox_shape + intensity_mean

        props_table.append(props)

    prop_names = (prop_names +
        ('centroid_r', 'centroid_c') +
        ('bbox_r', 'bbox_c') + 
        ('cy3', 'cy5'))

    props_table = pd.DataFrame(props_table, columns = prop_names)
    props_table['fov'] = fov

    print(f'[summarize_2d] Writing output to {output_path}')
    props_table.to_csv(output_path)

    return props_table

def regionprops_intensity_3D(label_image, intensity_image): 
    results = regionprops(label_image, intensity_image)
    for i, obj in enumerate(results): 
        a = results[i]
        b = RegionProperties3D(a.slice, 
                               a.label, 
                               a._label_image,
                               a._intensity_image, 
                               a._cache_active)
        results[i] = b
    return(results)

def regionprops_3D(im): 
    results = regionprops(im)
    for i, obj in enumerate(results): 
        a = results[i]
        b = RegionProperties3D(a.slice, 
                               a.label, 
                               a._label_image, 
                               a._intensity_image, 
                               a._cache_active)
        results[i] = b
    return(results)

class RegionProperties3D(RegionProperties): 
    
    @property
    def mask(self): 
        return self.image
    
    @property
    def volume(self): 
        return(self.area)
    
    @property 
    def bbox_volume(self): 
        mask = self.mask
        return np.prod(mask.shape)
   
    @property
    def max_perimeter(self): 
        mask = self.mask
        max_perimeter = 0
        for i in range(mask.shape[0]): 
            max_perimeter = max(max_perimeter, perimeter(mask[i,:,:], 4))
        return max_perimeter

    @property
    def sphericity(self): 
        vol = self.volume
        r = (3 / 4 / np.pi * vol)**(1 / 3)
        a_equiv = 4 * np.pi * r**2
        a_region = self.surface_area
        return a_equiv / a_region 
    
    @property
    def surface_area(self): 
        mask = self.mask
        tmp = np.pad(np.atleast_3d(mask), pad_width = 1, mode = 'constant')
        tmp = spim.convolve(tmp, weights = ball(1)) / 5
        verts, faces, norms, vals = marching_cubes(volume = tmp, level = 0)
        self._surface_mesh_vertices = verts
        self._surface_mesh_simplices = faces 
        area = mesh_surface_area(verts, faces)
        return area
    
    @property
    def surface_mesh_vertices(self): 
        if not hasattr(self, '_surface_mesh_vertices'): 
            _ = self.surface_area
        return self._surface_mesh_vertices
    
    @property
    def surface_mesh_simplices(self): 
        if not hasattr(self, '_surface_mesh_simplices'): 
            _ = self.surface_area
        return self._surface_mesh_simplices 

    @property
    def background_intensity(self): 
        # mean intensity of the lower half of the pixel intensity distribution 
        it = self.image_intensity[self.image].flatten()
        th = np.quantile(it, 0.75) # spots don't take up the lower 75% of the cell
        it = it[it <= th]
        if len(it) == 0: 
            return 0
        else:
            return np.mean(it)
        
    @property
    def intensity_percentile(self): 
        return np.percentile(self.image_intensity[self.image], 90, axis = 0).astype(np.float64, copy=False)
    
    @property
    def intensity_median(self): 
        return np.median(self.image_intensity[self.image],axis=0).astype(np.float64, copy=False)

def get_cells_at_boundary(seg, keep_z = False): 
    if len(seg.shape) == 2: 
        b0 = np.unique(seg[0,:])
        b1 = np.unique(seg[-1,:])
        b2 = np.unique(seg[:,0])
        b3 = np.unique(seg[:,-1])
        cells_at_boundary = set(b0) | set(b1) | set(b2) | set(b3)
    elif len(seg.shape) == 3: 
        b0 = np.unique(seg[:,0,:])
        b1 = np.unique(seg[:,-1,:])
        b2 = np.unique(seg[:,:,0])
        b3 = np.unique(seg[:,:,-1])
        b4 = np.unique(seg[0,:,:])
        b5 = np.unique(seg[-1,:,:])
        if keep_z: 
            cells_at_boundary = set(b0) | set(b1) | set(b2) | set(b3)
        else:
            cells_at_boundary = set(b0) | set(b1) | set(b2) | set(b3) | set(b4) | set (b5)
    else:
        raise NotImplementedError
    cells_at_boundary.remove(0)    
    return cells_at_boundary

def summarize_cell_segmentation_depr(ws, fov, seg_model, seg_scaling, keep_z):
    seg = ws.load_segmentation(fov, seg_model, seg_scaling, return_original = True)
    cells_at_boundary = get_cells_at_boundary(seg, keep_z)

    props_table = []
    if len(seg.shape) == 2: #2d 
        prop_names = ('area', 'area_bbox', 'extent', 'label', 'perimeter', 'circularity')
        rps = regionprops_2D(seg.astype(int)) 
    
        for rp in rps: 
            props = tuple((getattr(rp, prop_name) for prop_name in prop_names))
            centroid = getattr(rp, 'centroid')
            props = props + centroid
            props_table.append(props)
        
        prop_names = prop_names + ('centroid_r', 'centroid_c')

    elif len(seg.shape) == 3: #3d    
        prop_names = ('volume', 'bbox_volume', 'extent', 'label', 
            'surface_area', 'sphericity', 'max_perimeter')
        rps = regionprops_3D(seg.astype(int))
        
        for rp in rps: 
            props = tuple((getattr(rp, prop_name) for prop_name in prop_names))
            centroid = getattr(rp, 'centroid')
            props = props + centroid
            bbox_shape = getattr(rp, 'mask').shape
            props = props + bbox_shape
            props_table.append(props)
 
        prop_names = (prop_names + 
            ('centroid_z', 'centroid_r', 'centroid_c') + 
            ('bbox_z', 'bbox_r', 'bbox_c'))
    else:
        raise NotImplementedError

    props_table = pd.DataFrame(props_table, columns = prop_names)    
    props_table['not_at_boundary'] = props_table['label'].apply(lambda c: c not in cells_at_boundary)
    print(f"Fraction of cells not at boundary: {props_table['not_at_boundary'].mean()}")
    print(f"Number of cells not at boundary: {props_table['not_at_boundary'].sum()}")

    #output_path = os.path.join(params['main_output_path'], f"reports/F{fov}_seg.csv")
    #props_table.to_csv(output_path, index = False)
    return props_table

def get_iqr_bounds(values): 
    q1, q3 = np.quantile(values, (0.25, 0.75))
    iqr = q3 - q1
    upper_bound = q3 + 1.5*iqr
    lower_bound = q1 - 1.5*iqr
    return lower_bound, upper_bound

def summarize_3d(ws, fov): 
    output_path = ws.get_segmentation_qc_path(fov)
    if os.path.exists(output_path): 
        print(f'[summarize_3d] Summary exists at {output_path}. Loading...')
        return pd.read_csv(output_path, dtype={'fov': str})

    # load segmentation
    print(f'[JA_insertion] Now loading segmentation for FOV_{fov}')
    seg = ws.load_segmentation(fov)

    # load prehybs
    prehyb_cy3_rg = ws.load_prehyb_registered(fov, 'Cy3')
    prehyb_cy5_rg = ws.load_prehyb_registered(fov, 'Cy5')
    signal = np.stack((prehyb_cy3_rg, prehyb_cy5_rg), axis = -1)

    # calculate properties
    prop_names = ('volume', 'bbox_volume', 'extent', 'label', 
        'surface_area', 'sphericity', 'max_perimeter')

    rps = regionprops_intensity_3D(
        seg.astype(int), 
        signal
    )

    props_table = []
    for rp in rps: 

        centroid = getattr(rp, 'centroid')
        bbox_shape = getattr(rp, 'mask').shape
        bbox = getattr(rp, 'bbox')
        bbox_z = (bbox[0], bbox[3]-1) ## bbox[0] is min z, bbox[3] is max z.
        intensity_mean = tuple(getattr(rp, 'intensity_mean'))
        
        props = tuple((getattr(rp, prop_name) for prop_name in prop_names))
        props = props + centroid + bbox_shape + bbox_z + intensity_mean  

        props_table.append(props) 

    prop_names = (prop_names + 
        ('centroid_z', 'centroid_r', 'centroid_c') + 
        ('bbox_z', 'bbox_r', 'bbox_c') +
        ('bbox_z_min', 'bbox_z_max') + 
        ('cy3', 'cy5'))
    props_table = pd.DataFrame(props_table, columns = prop_names)
    props_table['fov'] = fov
   
    print(f'[summarize_3d] Writing output to {output_path}')
    props_table.to_csv(output_path)

    return props_table

def threshold_cells(ws, seg_summary, threshold_params, ratio): 
    output_path = ws.get_segmentation_qc_dir() / 'passed.csv'
    if os.path.exists(output_path): 
        print(f'[threshold_cells] Passed exists at {output_path}. Loading...')
        passed = pd.read_csv(output_path)
   
    passed_vol = seg_summary['volume'] > 30000 * ratio

    fig, ax = plt.subplots(1, len(threshold_params), figsize = (3*len(threshold_params), 3))
    passed = {}
    for i, param in enumerate(threshold_params): 
        values = np.log10(seg_summary.loc[passed_vol][param].values+1)
        lower_bound, upper_bound = get_iqr_bounds(values)
        ax[i].hist(values, bins = 50)
        ax[i].axvline(lower_bound, c = 'k', linestyle = 'dashed')
        ax[i].axvline(upper_bound, c = 'k', linestyle = 'dashed')
        ax[i].set_title(f'{param}\n10**{lower_bound:.2f}, 10**{upper_bound:.2f}')

        values = seg_summary[param].values
        if param == 'volume': 
            lower_bound = 30000 * ratio
            
        elif param == 'cy5' or param == 'cy3':
                lower_bound = 0
        else:
            lower_bound = 10**lower_bound-1
        upper_bound = 10**upper_bound-1
        passed[f'passed_{param}'] = (values < upper_bound) & (values > lower_bound)
        passed['label'] = seg_summary['label']
        passed['fov'] = seg_summary['fov']

    passed = pd.DataFrame(passed)
    passed['passed'] = passed[[f'passed_{param}' for param in threshold_params]].all(axis = 1)

    out_dir = ws.get_segmentation_qc_dir()
    plt.savefig(out_dir / 'summary.png')
    passed.to_csv(out_dir / 'passed.csv')

    return passed

def mask_cells(ws, fov, passed):
    output_path = ws.get_segmentation_masked_path(fov)
    if os.path.exists(output_path): 
        print(f'[mask_cells] Masked exists at {output_path}. Loading...')

    seg = ws.load_segmentation(fov)

    remove_idx = dict(zip(passed.label, ~passed['passed']))
    remove_idx[0] = False

    f = lambda x: remove_idx[x]
    vf = np.vectorize(f)
    seg_mask = vf(seg)
    
    seg_masked = np.where(seg_mask, 0, seg)

    print(f'[mask_cells] Writing output to {output_path}.')
    imsave(output_path, seg_masked) 

###############
###############
###############

if __name__ == "__main__": 

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_id', default = '1234')
    parser.add_argument('--data_keyword', default = '')
    parser.add_argument('-r', '--run_type', default = 'bandpass')
    parser.add_argument('-n', '--run_notes', default = 'normal')
    parser.add_argument('-s', '--seg_model', default = 'cellpose_modelD')
    parser.add_argument('-m', '--mode', default = '3D')
    parser.add_argument('-p', '--num_workers', default = 1, type = int)
    parser.add_argument('-f', '--fov_file', default = 'fovs.txt')
    args = parser.parse_args()

    ws = Workspace(args)
    fovs = ws.load_fovs()
    args.fovs_to_process = fovs.flatten()
    print(f'[main] fovs_to_process = {args.fovs_to_process}')
   
    # make directory if needed
    output_dir = ws.get_segmentation_qc_dir()
    Path(output_dir).mkdir(parents = True, exist_ok = True)

    if args.mode.upper() == '3D': 
        summarize = summarize_3d
        threshold_params = ['cy3', 'cy5', 'volume', 'surface_area', 'max_perimeter']
    elif args.mode.upper() == '2D':
        summarize = summarize_2d
        threshold_params = ['cy3', 'cy5', 'area', 'perimeter', 'circularity']
    else:
        raise NotImplementedError
    print(f'[main] filtering on threshold_params {threshold_params}')

    if args.num_workers <= 1: 
        seg_summary = []
        for fov in args.fovs_to_process: 
            if fov == 'xxx':
                continue
            props_table = summarize(ws, fov) 
            seg_summary.append(props_table)
    else: 
        seg_summary = Parallel(n_jobs = args.num_workers)(
            delayed(summarize)(ws, fov) for fov in args.fovs_to_process if fov != 'xxx'
        )
    seg_summary = pd.concat(seg_summary)
    
    # transfer the pixel into um*3, for x y: 2048 pixels is 261.224um, 1z is 0.27 um
    ratio = 0.12755 *  0.12755 * 0.27
    seg_summary['volume'] = seg_summary['volume'] * ratio
    seg_summary['surface_area'] = seg_summary['surface_area'] * 0.12755 *  0.12755 
    seg_summary['max_perimeter'] = seg_summary['max_perimeter'] * 0.12755 *  0.12755 


    passed = threshold_cells(ws, seg_summary, threshold_params, ratio)
    

    # then for each fov write final segmentation to file 
    if args.num_workers <= 1:
        for fov, fov_passed in passed.groupby('fov'): 
            mask_cells(ws, fov, fov_passed)
    else:
        Parallel(n_jobs = args.num_workers)(
            delayed(mask_cells)(ws, fov, fov_passed) 
            for fov, fov_passed in passed.groupby('fov')
        )
        

