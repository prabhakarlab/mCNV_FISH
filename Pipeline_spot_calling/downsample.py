import argparse
import glob
import os
import time

import numpy as np
import cv2
from scipy.ndimage import median_filter
import tifffile as tif

#################
#################

def _read_ims(im, channel):
    """
    Read a multi-channel tiff and return the requested channel as a
    3D (Z, Y, X) array.

    Uses ``tifffile.TiffFile(path).series[0].axes`` to identify the channel
    axis explicitly, instead of guessing "channel = smallest axis" -- that
    heuristic fails silently on single-z-plane or very thin stacks (where
    Z can equal or be smaller than the channel count).
    """
    st = time.time()
    with tif.TiffFile(im) as tf:
        axes = tf.series[0].axes  # e.g. 'ZYXC', 'CZYX', 'ZYX'
        shape = tf.series[0].shape
    image = tif.imread(im)
    et = time.time()
    print(f'imread: {et-st:.3f}s  shape={shape}  axes={axes!r}')

    if 'C' in axes:
        c_axis = axes.index('C')
        image = np.take(image, channel, axis=c_axis)
        print(f'  selected channel {channel} along axis {c_axis} (axes={axes!r})')
    else:
        print(f'  no channel axis present; using full volume as-is')
    return image

#################
#################

def _run_scale_MF_ds(file_to_process, args, dtype):
    if dtype=='npy':
        raw = np.load(file_to_process)
        #print(raw.shape, raw.shape.index(min(raw.shape)))
        if raw.shape.index(min(raw.shape))==(len(raw.shape)-1):
            raw = np.moveaxis(raw, -1, 0)
            print(f'shifted axis: new raw shape is {raw.shape}')
        raw = raw.astype(int)
    
    if dtype=='tif':
        raw = _read_ims(file_to_process, args.channel)
        
    if args.medianfilter:
        st = time.time()
        image = median_filter(raw, size=[1,3,3]) # NOTE: 1,3,3, because z, x,y
        et = time.time()
        print(f'Median filter: {et-st} seconds')
    else:
        image = raw
    
    image = image.astype('float32')
    ds_z = args.ds_z
    ds_xy = args.ds_xy
    ds_add = args.ds_add
            
    image_S_MF_DS = np.zeros((int(np.floor(image.shape[0]/ds_z)), int(np.floor(image.shape[1]/ds_xy)), int(np.floor(image.shape[2]/ds_xy)))) 
    
    ## do resizing as specified, unless ds_xy is 1.
    for i in range(image_S_MF_DS.shape[0]):
        if ds_add>0:
            planes = np.sum(image[(ds_z*i):(ds_z*i+ds_add+1), :, :], axis=0, dtype='uint16') ## must be an integer for cv2 resize to work...
        else:
            planes = image[ds_z*i, :, :]
            
        if ds_xy!=1:
            image_S_MF_DS[i, :, :] = cv2.resize(planes, (image_S_MF_DS.shape[2], image_S_MF_DS.shape[1]), interpolation = cv2.INTER_CUBIC)
        else:
            image_S_MF_DS[i, :, :] = planes
    
    ## clipping    
    z, y, x = np.where(image_S_MF_DS > 4095)
    image_S_MF_DS[z, y, x] = 4095
    z, y, x = np.where(image_S_MF_DS < 0)
    image_S_MF_DS[z, y, x] = 0    
    print(f'After resizing, new shape: {image_S_MF_DS.shape}')
    
    image_S_MF_DS = image_S_MF_DS.astype('uint16') 
    ## saving    
    tif.imwrite(os.path.join(args.dest, file_to_process[:-4] + '_MF_' + ('T' if args.medianfilter else 'F') + '_DS_z_y_x_add_' + str(ds_z) + '_' + str(ds_xy) + '_' + str(ds_add) + '.tif'), image_S_MF_DS, photometric='minisblack')
    print(f'Scaled, median-filtered and downsampled image saved.')    
    
    del raw, image, image_S_MF_DS   


##############
##############
                    
def run_scale_MF_ds(args):
    if os.path.exists(args.source) and os.path.exists(args.dest):        
        print(f'Source and destination directories found.')            
        
        os.chdir(args.source)
        #print(os.getcwd())
        files_to_process = glob.glob('*' + args.keyword + '*') 
        #print(files_to_process)
            
        for fileid, file_to_process in enumerate(files_to_process):
            print(f'now doing: {fileid}, {file_to_process}, {file_to_process[-3:]}')
            if file_to_process[-3:]=='tif':
                _run_scale_MF_ds(file_to_process, args, dtype = 'tif')
            elif file_to_process[-3:]=='npy':
                _run_scale_MF_ds(file_to_process, args, dtype = 'npy')
                

##############
##############
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default = '')
    parser.add_argument('--keyword', default = '.tif')
    parser.add_argument('--dest', default = '')
    
    parser.add_argument('--channel', default=2, type=int, help='Channel index to extract (default 2 = DAPI for prehyb).')
    parser.add_argument('--medianfilter', action='store_true')
    parser.add_argument('--ds_xy', default=1, type=int)
    parser.add_argument('--ds_z', default=1, type=int)
    parser.add_argument('--ds_add', default=0, type=int)

    args = parser.parse_args() 
    run_scale_MF_ds(args)

##############
##############




