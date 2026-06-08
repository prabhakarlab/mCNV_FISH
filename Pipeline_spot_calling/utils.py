"""
Shared utility functions used across the pipeline.

Currently holds:
    _3D_translation            -- integer-pixel shifting of a 3D image.
    cellpose_mask_target_name  -- derive the pipeline's expected mask filename.
    rename_cellpose_mask       -- perform that rename on disk.
    find_stitching_shifts      -- locate the master-coord-array shifts file.

Replaces processFunctions3D_v4.py, which originally bundled the translation
function with several large dead helpers (image_process / dapi_process /
background_process) that were removed earlier. Anything genuinely shared
and stateless belongs here; domain-specific code should stay home.
"""

import os
import re
from pathlib import Path

import numpy as np


def _3D_translation(img: np.ndarray, shifts) -> np.ndarray:
    """
    Translate a 3D image by integer-pixel shifts along (z, y, x).

    Positive shifts move content toward higher indices and fill the
    freed edge with zeros; negative shifts do the reverse. Same shape
    as input is preserved.
    """
    initial_shape = img.shape
    for di, i in enumerate(shifts):
        if i > 0:
            img = np.moveaxis(img, di, 0)
            z = np.zeros((int(abs(i)),) + img.shape[1:])
            img = np.concatenate((z, img), axis=0)
            img = img[:initial_shape[di], ...]
        elif i < 0:
            img = np.moveaxis(img, di, 0)
            z = np.zeros((int(abs(i)),) + img.shape[1:])
            img = np.concatenate((img, z), axis=0)
            img = img[-initial_shape[di]:, ...]
        else:
            continue
        img = np.moveaxis(img, 0, di)

    assert img.shape == initial_shape
    return img


def cellpose_mask_target_name(filename: str) -> str:
    """
    Derive the pipeline-expected ``F<fov>_cp_masks.tif`` name from a
    Cellpose-output filename.

    Cellpose appends ``_<channel>_<channel>_<diameter>_cp_masks.tif`` to the
    input image's full name. The pipeline's ``Workspace.load_segmentation``
    expects just ``F<fov>_cp_masks.tif`` in ``segmentation/<seg_model>/``.

    Example::

        hyb_637_Cy5_CF40_Sona 1_561_RFP_CF40_Sona 1_1_F080_1_1_0_cp_masks.tif
        ->  F080_cp_masks.tif

    The FOV identifier is found as the first ``_F<digits>_`` pattern in the
    filename. Raises ValueError if no such pattern exists.
    """
    m = re.search(r'_(F\d+)_', filename)
    if m is None:
        raise ValueError(
            f"No '_F<digits>_' FOV pattern found in {filename!r}"
        )
    return f'{m.group(1)}_cp_masks.tif'


def rename_cellpose_mask(path: str) -> str:
    """
    Rename a single Cellpose mask file in place to the pipeline-expected
    ``F<fov>_cp_masks.tif`` form, leaving it in its original directory.
    Returns the new basename (not the full path).

    See ``cellpose_mask_target_name`` for the naming rule. To rename a
    whole directory of masks, just loop and call this on each file::

        from utils import rename_cellpose_mask
        for f in os.listdir(seg_dir):
            if f.endswith('_cp_masks.tif'):
                rename_cellpose_mask(os.path.join(seg_dir, f))
    """
    new_name = cellpose_mask_target_name(os.path.basename(path))
    os.rename(path, os.path.join(os.path.dirname(path), new_name))
    return new_name



def find_stitching_shifts(directory, pattern: str = None):
    """
    Locate the (single) stitching-shifts ``.npy`` file in `directory`.

    By default, globs for ``Master_coord_array*.npy`` (the historical
    naming, irrespective of the FOV range encoded in the filename).
    Pass a custom ``pattern`` -- typically via the CLI flag
    ``--stitch_coordinates_pattern`` -- to override.

    Returns:
        pathlib.Path of the single matching file, OR
        None if no file matches (caller decides what to do; the historical
        behavior was to silently skip and continue without loading shifts).

    Raises:
        RuntimeError if more than one file matches. Stitching should
        produce one shifts file per dataset, so multiple matches usually
        indicate a stale leftover. The error message guides the user to
        narrow the pattern via ``--stitch_coordinates_pattern`` or to
        delete the stale file.
    """
    pat = pattern if pattern is not None else "Master_coord_array*.npy"
    matches = sorted(Path(directory).glob(pat))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        listing = "\n".join(f"  {m.name}" for m in matches)
        raise RuntimeError(
            f"Multiple stitching-shift files match {pat!r} in {directory}:\n"
            f"{listing}\n"
            f"Stitching should produce a single shifts file per dataset. "
            f"Pass --stitch_coordinates_pattern '<unique-pattern>' to "
            f"select one, or remove stale files from the directory."
        )
    return matches[0]
