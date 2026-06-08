"""
ImageData: holds raw and processed images and parameters for these images

Modified from imagedataClasses
for use with functional style processing

30 Oct 19
"""

import os
import h5py
import copy
import csv
import re
import pprint as pp
import time

import warnings
from collections import OrderedDict
from typing import Union, List, Tuple
import numpy as np

import skimage.io  # for loading multi-frame ome-tif files
import scipy.ndimage as sp
import tifffile



class ImageData(object):
    """
    Class for holding image data from a single FOV
    including all registered, and filtered outputs
    also has attributes that provide info on fovs, hybs, image dimensions

    this should be used as a context manager,
    to ensure that all references to its arrays are deleted after the FOV is processed

    dimensions of the typical arrays are:
    (frames, y_pix, x_pix, number_of_hybs)
    """


    def __init__(self,
                 iteration: int,
                 fov: str,
                 output_path: str,
                 existing_img_path: str,
                 y_pix: int = 1024,
                 x_pix: int = 1024,
                 z_pix: int = 1,
				 z_slice: tuple = None,
                 first_roi:bool=True,
                 roi: int = 1,
                 num_bits: int = 1,
                 datatype: np.dtype = np.float64,
                 smfish_callout_method: str = 'peak_3D',	
                 microscope_type: str = 'confocal',
                 ) -> None:
        """
        Parameters
        ----------
        fov: str
            FOV reference for the ImageData object
        y_pix, x_pix: int
            image dimensions
        frames: int
            number of frames / image dimension in z
        num_bits: int
            number of bits
        dtype: numpy datatype (default float64)
            datatype in which all operations will take place
        border_padding: int

            add extra padding on borders of image
            added during registration step when borders
            are calcuated from misalignment
        dropped_bits: list of integers
            bits that you intend to drop
            should only be defined at ImageData intialization
        """

        # main parameters
        # ---------------

        self.iteration = iteration
        self.fov = fov
        self.output_path = output_path
        self.existing_img_path = existing_img_path        
        self.y_pix = y_pix
        self.x_pix = x_pix
        self.z_slice = z_slice
        self.frames = z_pix
        self.num_bits = num_bits
        self.datatype = datatype	
        self.smfish_callout_method = smfish_callout_method
        self.microscope_type = microscope_type
        self.first_roi = first_roi
        self.roi = roi


        # Lists to record colours associated with each bit
        # ------------------------------------------------
        # for storing the colour (e.g. "Cy5") of the image associated with each bit

        self.colour_list = None

        self.data = OrderedDict()


        # NOTE: all flags should be boolean
        #       all arrays are numpy arrays with dimension:
        #       (frames, y_pix, x_pix, num_bits)

        # self.printStagesStatus(f"at initialization")


    def __enter__(self):
        """
        create a h5py File object and save as an attribute (h5)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        
        pass

    #
    #                            Print info on status for all stages
    # -------------------------------------------------------------------------------------------
    #
    #
    
    def printStagesStatus(self,
                          text: str,
                          ) -> str:
        """
        print out the dictionary of flags and arrays
        """
        dotted_line = "-" * 60

        status_str = (dotted_line +
                      f"\nFOV {self.fov} Data and flags {text}:\n" +
                      dotted_line + "\n\n")

        for stage_num, stage in enumerate(self.data):

            stage_title_str = f"Stage {stage_num} : {stage}\n"
            status_str += stage_title_str + "-" * len(stage_title_str)

            # flag
            # ----
            status_str += f"\n\tFlag  : {self.data[stage]['flag']}"

            # array
            # -----
            if self.data[stage]['array'] is None:
                array_str = "-"
            if isinstance(self.data[stage]['array'], np.ndarray):
                array_str = f"{self.data[stage]['array'].shape}"
            status_str += f"\n\tArray : {array_str}"

            # info
            # ----
            status_str += f"\n\tInfo  : {self.data[stage]['info']}\n\n"

        status_str += dotted_line + "\n"

        print(status_str)

        return status_str


    #
    #                     Reading of images (from list of files or HDF5)
    # ---------------------------------------------------------------------------------
    #
    #			
			
    def readFiles3D(self,
                  data_path: str,
                  img_list: list,				  
				  num_chns: int=3,
                  verbose: bool = False,
				  background:bool=False,
				  chrom_array:list=None,
                  gene_list: list=[],
                  subtract_chn: list=[],
                  dims: tuple = (75,2048,2048)
                  ) -> None:
        """
        Same as readFiles but in 3D

        """

        num_imgs = len(img_list)
			
        for ind, img_num in enumerate(img_list):               
            if img_num == 0:
                continue
            if background:
                if img_num[1] not in subtract_chn:
                    continue

                raw_array = self.readImages(
                [img_num[:3]], data_path,      
    			num_chns = num_chns,
                verbose=verbose,		
            )
                h5_filepath = os.path.join(
                self.existing_img_path,
                f"prebleach_{fov}_imagedata_{img_num[1]}.hdf5"   
                ) 
                
                
            else:
                h5_filepath = os.path.join(
                self.existing_img_path,
                f"FOV_{self.fov}_imagedata_bit{img_num[4]}.hdf5"
                )

                with h5py.File(h5_filepath, "a") as f:                        
                    f.attrs.create('chn', img_num[1])
                    f.attrs.create('bit', img_num[4])           
                    f.attrs.create('hyb', img_num[3])
                    f.attrs.create('gene', gene_list[img_num[4]])                    
                    if chrom_array != None:
                        f.attrs.create('chr', chrom_array[ind])
                    else:
                        f.attrs.create('chr', 0) 


    def checkImages(self,img_input,
                prebleach: bool=False,
               checkArray: str=None,
               ) -> dict:
    
        files = os.listdir(self.existing_img_path)
        img_dict = img_input.copy()
        print(f'Checking files for {checkArray}!')
        for ind, i in enumerate(img_input):
            if prebleach:
                file_pattern = f'prebleach_{self.fov}_imagedata_{i[1]}.hdf5'
            else:
                bit = i[4]
                file_pattern = f'FOV_{self.fov}_imagedata_bit{bit}.hdf5'
                
            if file_pattern in files:
                with h5py.File(os.path.join(self.existing_img_path, file_pattern), 'r') as f:
                    if prebleach:
                        if self.fov in f.keys():
                            if checkArray in f[self.fov].keys():
                                img_dict[ind] = 0
                    else:
                        if checkArray in f.attrs.keys():
                            img_dict[ind] = 0
    
    
        return img_dict


def readImages_v2(img_list:List[tuple],
                   data_path:str,
                   z_slice:tuple=None,
                   smfish_callout_method: str = 'peak_3D',
                   read_method: str = None,
                   existing_img_path:str=None,
                   ):
        """
        Read images from specified microscope image format.
        Input is a list of tuples providing:
         filename, colour channel and other information
    
        Parameters
        ----------
        :param img_list:
        :param data_path:
        """

        start_time = time.time()
    
        num_imgs = len(img_list)
        
        if read_method == None:
    
            for img_num in range(num_imgs):
                # filename / path of the image is the first entry of the tuple
                img_fullpath = os.path.join(
                    data_path, img_list[img_num][0]
                )
        
                #
                # Read image file using reader for specific format
                # ------------------------------------------------
                #
        
                with warnings.catch_warnings():
        
                        # filter out 'not an ome-tiff master file' UserWarning
                        warnings.simplefilter("ignore", category=UserWarning)
    
                        tiff_frame = img_list[img_num][-1]
                        tiff_img = tifffile.imread(img_fullpath)
        				
                        if tiff_img.ndim == 4:
                            #print('Image shape: ', tiff_img.shape)
                            tiff_img = np.moveaxis(tiff_img, np.argmin(tiff_img.shape), -1)
        							
                            if z_slice == None:                    
                                    z_slice = (0,tiff_img.shape[0])						
                            else: 
                                print('Using z slices: ', z_slice)	
                                    
                            temp_image = tiff_img[z_slice[0]:z_slice[1], ..., tiff_frame]
                            
                            if smfish_callout_method == 'peak_2D':  
                                sum_image = np.sum(temp_image, axis=0)
                                temp_image = np.max(temp_image, axis=0)
                            else:
                                sum_image = None

        elif read_method == 'dump':
            print(img_list)
            # filename / path of the image is the first entry of the tuple
            img_fullpath = os.path.join(
                data_path, img_list[0][1][0]
            )
            print(f'Rushing raw files: {img_list[0][1][0]}')
            
            
            hdf5_filename = f'FOV_{img_list[0][0]}_imagedata_bit{img_list[0][1][-1]}.hdf5'
            
            if hdf5_filename in os.listdir(existing_img_path):
                temp_image = None
                return

            with warnings.catch_warnings():
        
                        # filter out 'not an ome-tiff master file' UserWarning
                        warnings.simplefilter("ignore", category=UserWarning)
                        tiff_img = tifffile.imread(img_fullpath)                    
                        print('Files read!')
                        
            for img_num in range(num_imgs):
                    
                tiff_frame = img_list[img_num][1][-3]
                print('Channel: ', tiff_frame)            				
                if tiff_img.ndim == 4:
                    print('Image shape: ', tiff_img.shape)
                    tiff_img = np.moveaxis(tiff_img, np.argmin(tiff_img.shape), -1)
        							
                    if z_slice == None and smfish_callout_method == 'peak_2D':                    
                            z_slice = (0,tiff_img.shape[0])
                            
                    temp_image = tiff_img[z_slice[0]:z_slice[1], ..., tiff_frame]
                    
                    if smfish_callout_method == 'peak_2D':
                        temp_image = np.max(temp_image, axis=0)
                
                bit = img_list[img_num][1][-1]
                hdf5_filename = f'FOV_{img_list[img_num][0]}_imagedata_bit{bit}.hdf5'
                with h5py.File(os.path.join(existing_img_path, hdf5_filename), 'a') as f:
                    f.create_dataset('raw', data=temp_image)

        #print(f'Reading raw image done: {time.time()-start_time}')
			
        return temp_image, sum_image

