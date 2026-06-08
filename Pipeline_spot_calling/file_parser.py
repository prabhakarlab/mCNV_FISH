# -*- coding: utf-8 -*-
"""
Confocal file parser.

Key change from the previous version: image dimensions (ydim, xdim,
zdim) are read *directly from the tiff file* via tifffile.shape,
not parsed out of `_metadata.txt` via a `Width=` regex.

This means:
  - There's no longer a microscope-specific JSON-key lookup table
    (filedefaults / sm_Defaults['type_to_colour']) for dimensions.
  - When images are manually merged or have non-standard channel
    counts, dimensions stay correct because they come from the
    actual array.
  - The sidecar `_metadata.txt` is still read, but only for the two
    things that *aren't* in the tiff itself: the channel/wavelength
    list and the acquisition grid (Rows= / Columns=).

Public surface (kept identical to what the orchestrator already consumes):

    parser = ConfocalParser(data_path, background=False)
    parser.parseDirectory()            # populates parser.files_df
    img, bg, dapi, first_roi, gr, gc = parser.dfToDict(
        fovs_to_process=[...],
        bits_list=..., hyb_list=..., type_list=...,
        background=False, verbose=False,
    )
    # parser.files_df['ydim','xdim','frames'] are populated from the tiff itself.
"""

import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile

from config import FILE_PATTERNS


# Module-level lookup. The previous code did this inline inside readMetadata
# via a chain of `if i == '700': color_list[n] = "Cy5"` statements. Lifting
# it out makes it the obvious place to edit if a new wavelength is added.
WAVELENGTH_TO_CHANNEL = {
    "700": "Cy5",
    "600": "Cy3",
    "785": "Cy7",
    "450": "dapi",
}


# ---------------------------------------------------------------------------

class ConfocalParser:
    """
    Walks the data directory, discovers hyb/prehyb/dapi tiff files,
    and exposes:
       .files_df              — one row per (file, channel) pair
       .dfToDict(...)         — per-FOV ordered file lists for the orchestrator
    """

    # Columns of files_df. Same set as before so downstream code is unchanged.
    _DF_COLUMNS = [
        "file_name",
        "type", "hyb", "fov", "frames",
        "ydim", "xdim",
        "ygrid", "xgrid",
        "tiff_frame",
        "zpos", "ypos", "xpos",
        "z_slice", "roi",
        "file_type",
    ]

    def __init__(self, data_path: str, background: bool = False) -> None:
        assert data_path is not None, "data_path not provided"
        self.data_path = data_path
        self.background = background
        self.files_df: pd.DataFrame = pd.DataFrame(columns=self._DF_COLUMNS)
        self.first_roi = None  # confocal has no ROI; kept for API parity

    # ---------------------------------------------------------------------
    # Public: directory scan
    # ---------------------------------------------------------------------

    def parseDirectory(self, verbose: bool = False) -> pd.DataFrame:
        """Discover all hyb/prehyb/dapi tiffs and populate self.files_df."""
        tiff_re      = re.compile(FILE_PATTERNS["tiff"])
        tiff_hyb0_re = re.compile(FILE_PATTERNS["tiff_hyb0"])
        meta_re      = re.compile(FILE_PATTERNS["metadata"])

        rows = []
        for filetype, subdir in [("hyb",    FILE_PATTERNS["hyb_dir"]),
                                 ("prehyb", FILE_PATTERNS["prehyb_dir"]),
                                 ("dapi",   FILE_PATTERNS["dapi_dir"])]:
            folder = os.path.join(self.data_path, subdir)
            if not os.path.isdir(folder):
                warnings.warn(f"{filetype} folder not found at {folder}; skipping.")
                continue

            filelist = os.listdir(folder)

            # Sidecar metadata: channel list + grid only.
            colors, grid_rows, grid_cols = self._read_sidecar_metadata(
                filelist, meta_re, folder,
            )
            fov_strlen = len(str(grid_rows * grid_cols))

            for fname in filelist:
                m  = tiff_re.search(fname)
                m0 = tiff_hyb0_re.search(fname) if m is None else None
                if m is None and m0 is None:
                    continue

                if m is not None:
                    fov = m.group(3)
                    hyb = int(m.group(2)) if filetype == "hyb" else -1
                else:
                    fov = m0.group(2)
                    hyb = 0 if filetype == "hyb" else -1

                # ---- THE KEY CHANGE: dimensions from the tiff itself ----
                fullpath = os.path.join(folder, fname)
                ydim, xdim, frames = self._read_dims_from_tiff(fullpath)

                # One row per channel, matching the old layout.
                for ch_idx, ch in enumerate(colors):
                    rows.append({
                        "file_name":  f"{filetype}/{fname}",
                        "type":       ch,
                        "hyb":        hyb,
                        "fov":        str(fov).zfill(fov_strlen),
                        "frames":     frames,
                        "ydim":       ydim,
                        "xdim":       xdim,
                        "ygrid":      grid_rows,
                        "xgrid":      grid_cols,
                        "tiff_frame": ch_idx,
                        "zpos":       np.nan, "ypos": np.nan, "xpos": np.nan,
                        "z_slice":    np.nan,
                        "roi":        0,
                        "file_type":  filetype,
                    })

        self.files_df = pd.DataFrame(rows, columns=self._DF_COLUMNS)
        # Numeric columns (same coercion as before).
        for col in ["hyb", "frames", "ydim", "xdim"]:
            self.files_df[col] = pd.to_numeric(
                self.files_df[col], errors="coerce", downcast="unsigned",
            )
        self.files_df.sort_values(by=["type", "hyb", "fov"], inplace=True)

        if verbose:
            print(self.files_df)
        return self.files_df

    # ---------------------------------------------------------------------
    # Helper: read dims straight from the tiff, using ITS OWN AXIS LABELS.
    # Returns (ydim, xdim, zdim).
    # ---------------------------------------------------------------------

    @staticmethod
    def _read_dims_from_tiff(path: str) -> Tuple[int, int, int]:
        """
        Read (ydim, xdim, zdim) from the tiff using its declared axis order
        (``tifffile.TiffFile(path).series[0].axes``), NOT a magnitude
        heuristic on ``shape``.

        Using ``series.axes`` ('YX', 'ZYX', 'ZYXC', 'CZYX', etc.) tells us
        unambiguously which axis is which.
        """
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            shape = series.shape
            axes  = series.axes

        if 'Y' not in axes or 'X' not in axes:
            raise ValueError(
                f"Tiff {path} has no Y or X axis "
                f"(axes={axes!r}, shape={shape}); cannot determine dims."
            )

        dim = {ax: int(sz) for ax, sz in zip(axes, shape)}
        return dim['Y'], dim['X'], dim.get('Z', 1)

    # ---------------------------------------------------------------------
    # Sidecar: ONLY channels + grid. (No more Width=, StepCount= regexes.)
    # ---------------------------------------------------------------------

    @staticmethod
    def _read_sidecar_metadata(filelist, meta_re, folder
                               ) -> Tuple[List[str], int, int]:
        color_pat    = re.compile(r"Wavelength=(\d+)", re.IGNORECASE)
        grid_row_pat = re.compile(r"(?<=Rows=)(\d+)")
        grid_col_pat = re.compile(r"(?<=Columns=)(\d+)")

        for fname in filelist:
            if not meta_re.search(fname):
                continue
            with open(os.path.join(folder, fname)) as f:
                text = f.read()
            wavelengths = re.findall(color_pat, text)
            colors = [WAVELENGTH_TO_CHANNEL.get(w, w) for w in wavelengths]
            grid_rows = int(re.findall(grid_row_pat, text)[-1])
            grid_cols = int(re.findall(grid_col_pat, text)[-1])
            return colors, grid_rows, grid_cols

        raise FileNotFoundError(f"No _metadata.txt file found in {folder}")

    # ---------------------------------------------------------------------
    # Public: per-FOV grouping.
    # ---------------------------------------------------------------------

    def dfToDict(self,
                 fovs_to_process: list,
                 bits_list: list,
                 hyb_list: list,
                 type_list: list,
                 roi: int = None,           # ignored; kept for API parity
                 background: bool = False,
                 verbose: bool = False,
                 ) -> Tuple[Dict, Dict, Dict, int, int, int]:
        """
        Build, per FOV:
          img_files[fov]        : list of (filename, channel, tiff_frame, hyb, bit)
          background_files[fov] : list of (filename, channel, tiff_frame, hyb)  [if background]
          dapi_files[fov]       : list of (filename, channel, tiff_frame)

        Returns: (img_files, background_files, dapi_files,
                  first_roi, grid_row, grid_col)
        """
        assert not self.files_df.empty, "files_df is empty — call parseDirectory() first"

        df = self.files_df
        grid_row = int(df["ygrid"].iloc[0])
        grid_col = int(df["xgrid"].iloc[0])
        fov_strlen = len(str(grid_row * grid_col))

        img_files, background_files, dapi_files = {}, {}, {}

        for fov in fovs_to_process:
            if fov == "xxx":
                continue
            fov = str(fov).zfill(fov_strlen)
            img_files[fov] = []

            # --- hyb bits in codebook order ---
            for bit in bits_list:
                mask = ((df["fov"].astype(str) == fov)
                        & (df["type"] == type_list[bit])
                        & (df["hyb"]  == hyb_list[bit]))
                hits = df[mask]
                tag = (f"bit {bit} (FOV {fov}, hyb {hyb_list[bit]}, "
                       f"type {type_list[bit]})")
                assert not hits.empty, f"Entry for {tag} not found!"
                if len(hits) > 1:
                    warnings.warn(f"Multiple files for {tag}; using first.")
                row = hits.iloc[0]
                img_files[fov].append((
                    row["file_name"], type_list[bit],
                    row["tiff_frame"], row["hyb"], bit,
                ))

            # --- dapi ---
            dapi_files[fov] = [
                (r["file_name"], r["type"], r["tiff_frame"])
                for _, r in df[(df["fov"] == fov)
                               & (df["file_type"] == "dapi")].iterrows()
            ]

            # --- background (prehyb), only if requested ---
            if background:
                background_files[fov] = [
                    (r["file_name"], r["type"], r["tiff_frame"], r["hyb"])
                    for _, r in df[(df["fov"] == fov)
                                   & (df["file_type"] == "prehyb")].iterrows()
                ]

        if verbose:
            print("\nFiles by FOV:")
            for fov, entries in img_files.items():
                print(f" FOV {fov}: {len(entries)} bits")

        return img_files, background_files, dapi_files, self.first_roi, grid_row, grid_col
