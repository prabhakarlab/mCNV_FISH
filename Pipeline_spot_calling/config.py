# -*- coding: utf-8 -*-
"""
Pipeline configuration.

If you ever need to support a different microscope, the right move
is to add a second module like this one (e.g. `config_dory.py`) and
pick which to import at the top of the orchestrator, rather than
reintroducing a runtime lookup table.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Defaults  (previously sm_Defaults('Confocal').findDefaults()
#                    + .advanced() + .adaptive_threshold(),
#            merged into one flat dict.)
#
# Anything the orchestrator overrides per-run still lives in the orchestrator;
# this dict is just the starting point.
# ---------------------------------------------------------------------------

DEFAULTS = {
    # --- pipeline identity / I/O ---
    "name":              "confocal_default",
    "microscope_type":   "confocal",
    "codebook_path":     "fpkm_data.txt",
    "calibration_path":  "calibration.txt",

    # --- acquisition / geometry ---
    "stage_pixel_matrix": 8 * np.array([[0, -1], [-1, 0]]),
    "roi":                None,        # confocal has no ROI concept
    "z_slice":            None,
    "grid_height":        5,
    "grid_length":        5,
    "fovs_to_process":    [],

    # --- channels (filename code -> canonical name) ---
    #     The values are what appear in the _metadata.txt Wavelength= field.
    #     Add entries here if a new dye is introduced; nothing else needs to change.
    "type_to_colour": {
        "Cy3": "600",
        "Cy5": "700",
        "Cy7": "785",
    },

    # --- filtering ---
    "low_cut":         100,
    "high_cut":        300,
    "bw_filter_order": 2,

    # --- background / prebleach ---
    "background":      True,
    "background_type": "prebleach",
    "subtract_chn":    ["Cy3"],

    # --- adaptive thresholding ---
    "fdr_percent":           [0.01, 0.01, 0.01],
    "smfish_callout_method": "peak_3D",

    # --- advanced (rarely toggled) ---
    "correct_field":      False,
    "correct_distortion": False,
    "subtract_background": False,
    "drop_genes":         [],
    "num_iterations":     1,
    "bits_to_drop":       [],

    # --- runtime ---
    "imgcolour_to_maskcolour": None,
    "multithread":             False,
}


# ---------------------------------------------------------------------------
# File-naming conventions  
#
# These are the *only* per-installation values that genuinely have to vary
# with how your acquisition software names files. Keep them here so it's the
# one place to edit when a naming convention changes.
# ---------------------------------------------------------------------------

FILE_PATTERNS = {
    # Top-level subdirectories that contain each image type:
    "hyb_dir":    "hyb",
    "prehyb_dir": "prehyb",
    "dapi_dir":   "dapi",

    # Tiff filename patterns. Two are needed because hyb-0 files
    # historically didn't carry an explicit hyb number.
    "tiff":        r"([a-zA-Z0-9_-]+)_(\d+)_F(\d+).tif",   # name_HYB_FFOV.tif
    "tiff_hyb0":   r"([a-zA-Z0-9_-]+)_F(\d+).tif",         # name_FFOV.tif  (assumed hyb=0)

    # Sidecar metadata text-file pattern (only used for grid + wavelength info;
    # image dimensions are read directly from the tiff itself).
    "metadata":    r"([a-zA-Z0-9_-]+)_metadata.txt",
}
