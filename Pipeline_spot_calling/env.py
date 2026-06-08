# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:00:09 2023

@author: linli
"""

# working directory
BASE_PATH = "/path/to/your/directory/" # path to base directory where the pipeline expects to find _data and _processed_data 

# parallel processing for dask
SCHEDULER_PORT = 8586 # port for dask
NUM_WORKERS = 8 # number of workers for dask

