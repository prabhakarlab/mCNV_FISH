"""
3D stitching pipeline (prehyb DAPI channel).

Reads the prehyb DAPI z-stacks for each FOV, treats them as tiles in a
serpentine (boustrophedon) grid of master_nrow x master_ncol, then:
  1. xy-registers each pair of stacked tiles within a column (register_mip_xy)
  2. xy-registers the column canvases to each other (register_mip_xy_2d)
  3. z-registers within columns (register_mip_z)
  4. z-registers across columns (register_mip_2d_z)
  5. emits the master coordinate array for downstream stitching.

This file mirrors run__finalUTD_to_use__qc_registration.py but for DAPI only
and at the inter-FOV (rather than inter-cycle) level.
"""
import argparse
import os
import time
from collections import Counter

import numpy as np
from scipy.ndimage import median_filter
from skimage.registration import phase_cross_correlation

import qc_registration_reg_functions as reg
import qc_registration_utils as utils


# Sentinel matching reg_functions: 999 means "could not register".
FAILED_SHIFT = 999


##############
##############

def _layout_from_serpentine(allfovs, nrow, ncol):
    """Default layout: serpentine (boustrophedon) based on 0-indexed FOV order.

    Raw FOV numbers are first remapped to 0-indexed grid positions so a partial
    run (e.g. minimal example with FOVs 80, 81) lays out from (0, 0) onwards
    instead of from the FOV's original full-slide position. For a full-slide
    run with FOVs 0..nrow*ncol-1, the remap is the identity and behaviour is
    unchanged.
    """
    raw_fovs = [int(f) for f in allfovs['fov']]
    sorted_unique = sorted(set(raw_fovs))
    fov_to_idx = {f: i for i, f in enumerate(sorted_unique)}

    if len(sorted_unique) > nrow * ncol:
        raise ValueError(
            f'{len(sorted_unique)} FOVs supplied but grid is '
            f'{nrow}x{ncol}={nrow*ncol}. Increase --master_nrow / --master_ncol, '
            f'or use --fov_layout to specify the layout explicitly.'
        )

    fov_to_coord = {}
    for fov in raw_fovs:
        idx = fov_to_idx[fov]
        xcoor = int(np.floor(idx / nrow))
        ycoor = idx % (2 * nrow)
        if ycoor >= nrow:
            ycoor = nrow - ycoor % nrow - 1
        fov_to_coord[fov] = (xcoor, ycoor)
    return nrow, ncol, fov_to_coord


def _layout_from_file(path, allfovs):
    """Parse an explicit layout file.

    Format: one row per line, top-to-bottom (first line = top of grid).
    Within a line, FOV numbers are comma-separated, left-to-right
    (first value = leftmost column). Whitespace around values is ignored.

    Example file contents for a 2x2 grid with FOVs 76, 77, 80, 81:
        077,080
        076,081
    This places 077 at top-left, 080 at top-right, 076 at bottom-left,
    081 at bottom-right.

    Returns (nrow, ncol, fov_to_coord) where ycoor=0 is the bottom row
    (matching the convention used by serpentine layouts).

    Validates:
    - All rows have the same number of columns.
    - Every FOV loaded into allfovs appears in the layout.
    Layout-only FOVs (i.e. positions you've left blank) are warned about
    but don't otherwise fail.
    """
    with open(path) as fh:
        lines = [ln.rstrip('\n') for ln in fh]

    rows = []
    for ln_idx, ln in enumerate(lines):
        stripped = ln.strip()
        if not stripped:
            continue
        row = [cell.strip() for cell in stripped.split(',')]
        if not all(row):
            raise ValueError(
                f'{path}: line {ln_idx+1} has an empty cell; '
                f'every grid position must hold a FOV number.'
            )
        rows.append(row)

    if not rows:
        raise ValueError(f'{path}: layout file is empty.')

    ncol = len(rows[0])
    for r_idx, row in enumerate(rows):
        if len(row) != ncol:
            raise ValueError(
                f'{path}: row {r_idx+1} has {len(row)} columns but row 1 has '
                f'{ncol}; all rows must have the same number of columns.'
            )

    nrow = len(rows)
    fov_to_coord = {}
    for r_idx, row in enumerate(rows):
        ycoor = nrow - r_idx - 1   # file's first line is the top → highest ycoor
        for c_idx, cell in enumerate(row):
            try:
                fov = int(cell)
            except ValueError:
                raise ValueError(f'{path}: row {r_idx+1} col {c_idx+1}: '
                                 f'{cell!r} is not an integer FOV number.')
            if fov in fov_to_coord:
                raise ValueError(f'{path}: FOV {fov} appears more than once.')
            fov_to_coord[fov] = (c_idx, ycoor)

    loaded = set(int(f) for f in allfovs['fov'])
    layout = set(fov_to_coord.keys())
    missing = loaded - layout
    if missing:
        raise ValueError(
            f'{path}: FOV(s) {sorted(missing)} are loaded into the pipeline '
            f'but not present in the layout file.'
        )
    extra = layout - loaded
    if extra:
        print(f'Note: {path} lists FOV(s) {sorted(extra)} that are not loaded; '
              f'those grid positions will remain empty.')

    return nrow, ncol, fov_to_coord


def generate_coords(allfovs, nrow, ncol, layout_file=''):
    """Lay the FOVs out in a grid of shape (nrow, ncol).

    If `layout_file` is a non-empty path, the grid is read from that file
    (see `_layout_from_file` for the format) and `nrow`/`ncol` are overridden
    by what the file specifies. Otherwise the default serpentine layout
    (see `_layout_from_serpentine`) is used.
    """
    if layout_file:
        file_nrow, file_ncol, fov_to_coord = _layout_from_file(layout_file, allfovs)
        if (nrow > 0 and ncol > 0) and (nrow, ncol) != (file_nrow, file_ncol):
            print(f'Layout file specifies {file_nrow}x{file_ncol}; '
                  f'overriding --master_nrow={nrow} --master_ncol={ncol}.')
        nrow, ncol = file_nrow, file_ncol
    else:
        nrow, ncol, fov_to_coord = _layout_from_serpentine(allfovs, nrow, ncol)

    masterimgarray = np.zeros((nrow, ncol))
    shiftxcol = np.zeros((nrow, ncol, 5))
    shiftyrow = np.zeros((nrow, ncol, 5))

    coords = []
    for fov_raw in allfovs['fov']:
        c = fov_to_coord[int(fov_raw)]
        coords.append(c)
        masterimgarray[c[1], c[0]] = 1

    allfovs['coords'] = coords
    print(masterimgarray)
    return shiftxcol, shiftyrow, masterimgarray, allfovs


##############
##############

def _pick_robust_center(vec):
    """Return the element of `vec` with smallest sum-of-squared-differences to
    all other elements. (Used as a robust replacement for the simple mean.)"""
    vec = np.asarray(vec)
    diffs = np.zeros_like(vec)
    for ind, val in enumerate(vec.tolist()):
        diffs[ind] = np.sum((val - vec) ** 2)
    return vec[np.argmin(diffs)]


def _resolve_shiftz(shiftz, pccz, diff_max_z):
    """If the masterz vote returned 'NA', fall back to the average of PCC and
    max-diff estimates. Used after both `register_mip_z` and `register_mip_2d_z`."""
    if shiftz == 'NA':
        return int(0.5 * pccz + 0.5 * diff_max_z)
    return shiftz


##############
##############

def xyreg(refname, tarname, refm, tarm, refm2, tarm2):
    """Stack-reg + PCC consensus xy-shift, with FAILED_SHIFT sentinels on disagreement."""
    sr_x, sr_y, sr_x_y_time = reg.register_xy_sr(refname, tarname, refm, tarm)
    sr_mf_x, sr_mf_y, sr_mf_x_y_time = reg.register_xy_sr(refname, tarname, refm2, tarm2)
    pcc_mip_x, pcc_mip_y, pcc_mip_time = reg.register_xy_pccmip(refm, tarm)
    pcc_mf_mip_x, pcc_mf_mip_y, pcc_mf_mip_time = reg.register_xy_pccmip(refm2, tarm2)

    xvec = np.array([sr_x, sr_mf_x, pcc_mip_x, pcc_mf_mip_x])
    yvec = np.array([sr_y, sr_mf_y, pcc_mip_y, pcc_mf_mip_y])

    mean_x_est = np.median(xvec)
    sd_x_est = max(5, 0.1 * np.std(xvec))
    close_x = np.array([abs(xvec - mean_x_est) <= sd_x_est])
    # print(f'sr_x, sr_mf_x, pcc_mip_x, pcc_mf_mip_x: {xvec}, mean:{mean_x_est}, std:{sd_x_est}, conf:{close_x}')
    conf_x = np.sum(np.array(close_x).astype(int))

    mean_y_est = np.median(yvec)
    sd_y_est = max(5, 0.1 * np.std(yvec))
    close_y = np.array([abs(yvec - mean_y_est) <= sd_y_est])
    # print(f'sr_y, sr_mf_y, pcc_mip_y, pcc_mf_mip_y: {yvec}, mean:{mean_y_est}, std:{sd_y_est}, conf:{close_y}')
    conf_y = np.sum(np.array(close_y).astype(int))

    if conf_x >= 3 and conf_y >= 3:
        shiftx_final = int(np.round(np.sum(xvec * close_x) / np.sum(close_x)))
        shifty_final = int(np.round(np.sum(yvec * close_y) / np.sum(close_y)))
        return shiftx_final, shifty_final, xvec, yvec
    return FAILED_SHIFT, FAILED_SHIFT, xvec, yvec


##############
##############

'''
Overall design: for a specific set of FOVs, we find whether there are more
columns than rows, or vice versa, present in that set of FOVs. We then choose
to link the dimension with more elements. For example, if the FOV set has 3
rows and 2 columns, we start by gluing each of the rows in the 2 columns
together, and then we glue the two columnar strips together. This averages
out the error in the column direction.

Because it's possible to glue rows first and then columns, or vice versa, we
create two separate arrays: shiftxcol and shiftyrow. Each array is a
nrow*ncol*2 array. A coordinate [x,y,0] denotes the xshift (i.e. number of
pixels in the columnar direction), while a coordinate [x,y,1] denotes the
yshift (i.e. number of pixels in the row direction).

If we glue rows first and then columns, the shiftyrow gets filled first. In
the second step, we fill the shiftxcol with values from the columns, so every
entry in a particular column will have the same coordinate shift. Vice versa
if we glue columns first and then rows.
'''


def register_mip_xy(dest, allfovs, masterimgarray, shiftxcol, shiftyrow, tileshape, padding, overlapwidth, threshold):
    coords = np.array(allfovs['coords'])
    tempcanvases = {'x_col_SI': [], 'x_col_EI': [], 'y_row_SI': [], 'y_row_EI': [], 'canvas': []}
    sortedvals = np.unique(np.sort(coords[:, 0]))

    if masterimgarray.shape[0] == 1:
        # Single-row grid: each "column" has exactly one FOV, so there's no
        # within-column pair to register. Mark FOVs as trivially registered
        # (masterimgarray=2) so the tempcanvas-build below paints them.
        print('Single-row grid: skipping within-column xy registration.')
        for c in allfovs['coords']:
            masterimgarray[c[1], c[0]] = 2
    else:
        print('processing cols first')  # the microscope moves in the column direction → less x-error
        failed = {'start_coord': [], 'end_coord': [], 'reg_results': []}
        succeeded = {'start_coord': [], 'end_coord': [], 'reg_x': [], 'reg_y': []}
        print(shiftyrow[:, :, 0], shiftyrow[:, :, 1])

        for x, xcoord in enumerate(list(sortedvals)):
            xcoords = np.array(coords[coords[:, 0] == xcoord])  # all coords in same column
            xcoords = xcoords[xcoords[:, 1].argsort()[::-1]]    # reverse sort: higher y on top
            print(f'xcoords are {xcoords}.')

            for y, ycoord in enumerate(list(xcoords)):
                if y == 0:
                    continue
                ycoordup = xcoords[y - 1]
                print(f'{y},{ycoord},{y-1}, {ycoordup}')
                img = allfovs['mip'][allfovs['coords'].index((ycoordup[0], ycoordup[1]))]  # top
                imgn = allfovs['mip'][allfovs['coords'].index((ycoord[0], ycoord[1]))]     # bottom

                imgS = img[int(np.round(7 / 8 * imgn.shape[0])):, :]  # by definition row 0 is on the bottom
                imgnS = imgn[:int(np.round(1 / 8 * img.shape[0])), :]

                shiftx, shifty, sx_raw, sy_raw = xyreg(
                    str(allfovs['fov'][y - 1]), str(allfovs['fov'][y]),
                    imgS, imgnS, median_filter(imgS, [3, 3]), median_filter(imgnS, [3, 3]))

                if shiftx != FAILED_SHIFT and shifty != FAILED_SHIFT:
                    masterimgarray[ycoordup[1], ycoordup[0]] = 2
                    masterimgarray[ycoord[1], ycoord[0]] = 2
                    shiftyrow[ycoord[1], ycoord[0], 0] = shiftx
                    shiftyrow[ycoord[1], ycoord[0], 1] = shifty
                    succeeded['start_coord'].append((ycoord[1], ycoord[0]))
                    succeeded['end_coord'].append((ycoordup[1], ycoordup[0]))
                    succeeded['reg_x'].append(shiftx)
                    succeeded['reg_y'].append(shifty)
                else:
                    failed['start_coord'].append((ycoord[1], ycoord[0]))
                    failed['end_coord'].append((ycoordup[1], ycoordup[0]))
                    failed['reg_results'].append(np.stack((sx_raw, sy_raw), axis=0))

        print(f'Failed registrations: {failed}')
        print(f'Succeeded registration: {succeeded}')

        ############## Clean-up: recheck all failed registrations to see if any
        ## of the values were close to the actual (usually the stack-reg is off).
        ## Then force a value (usually the average of the PCC values).

        successx = np.array(succeeded['reg_x'])
        successy = np.array(succeeded['reg_y'])
        sd_x_est = min(3, np.std(successx))  # 6 Mar 2024 - was originally max!
        sd_y_est = min(3, np.std(successy))
        print(successx, successy, sd_x_est, sd_y_est, np.mean(successx), np.mean(successy))

        bestx = _pick_robust_center(successx)
        besty = _pick_robust_center(successy)

        ## check if any successfully-registered shift is actually off the average, and correct
        falsesuccessx = abs(successx - bestx) >= 3 * sd_x_est
        falsesuccessy = abs(successy - besty) >= 3 * sd_y_est
        print(f'Checking for false successfully registered stacks: x:{falsesuccessx}, y:{falsesuccessy}')

        for index, val in enumerate(falsesuccessx):
            if val:
                print('Now updating for:', succeeded['start_coord'][index][0],
                      ' replacing ', succeeded['reg_x'][index], ' with ', bestx)
                shiftyrow[succeeded['start_coord'][index][0], succeeded['start_coord'][index][1], 0] = bestx

        for index, val in enumerate(falsesuccessy):
            if val:
                print('Now updating for:', succeeded['start_coord'][index][1],
                      ' replacing ', succeeded['reg_y'][index], ' with ', besty)
                shiftyrow[succeeded['start_coord'][index][0], succeeded['start_coord'][index][1], 1] = besty

        ## check if any failed registration had values that were correct but contradicted by the other algo.
        for x, xcoord in enumerate(failed['start_coord']):
            failedresults = failed['reg_results'][x]
            print(f'x: {failedresults[0,:]} ; y: {failedresults[1,:]}')

            close_x = np.array([abs(failedresults[0, :] - np.mean(successx)) <= threshold])
            close_y = np.array([abs(failedresults[1, :] - np.mean(successy)) <= threshold])
            conf_x = np.sum(np.array(close_x).astype(int))
            conf_y = np.sum(np.array(close_y).astype(int))

            print(f'stats: {close_x}, {close_y}, {conf_x}, {conf_y}')

            if conf_x >= 2 and conf_y >= 2:
                shiftx = int(np.round(np.sum(failedresults[0, :] * close_x) / np.sum(close_x)))
                shifty = int(np.round(np.sum(failedresults[1, :] * close_y) / np.sum(close_y)))
            else:
                shiftx = int(np.mean(successx))
                shifty = int(np.mean(successy))
            print('Now updating:', failed['start_coord'][x][0], failed['start_coord'][x][1])
            masterimgarray[failed['start_coord'][x][0], failed['start_coord'][x][1]] = 2
            masterimgarray[failed['end_coord'][x][0], failed['end_coord'][x][1]] = 2
            shiftyrow[failed['start_coord'][x][0], failed['start_coord'][x][1], 0] = shiftx
            shiftyrow[failed['start_coord'][x][0], failed['start_coord'][x][1], 1] = shifty

        print(f'After clean-up, masterimgarray is: {masterimgarray}')
        print(f'After clean-up, shifts are: {shiftyrow[:,:,0]}')
        print(f'After clean-up, shifts are: {shiftyrow[:,:,1]}')

    ############## Generate temp canvases.
    for x, xcoord in enumerate(list(sortedvals)):
        xcoords = np.array(coords[coords[:, 0] == xcoord])
        xcoords = xcoords[xcoords[:, 1].argsort()[::-1]]

        # Canvas y-dim now includes `padding` so the initial 1/4*padding offset
        # fits even when there is only one row per column. For multi-row cases
        # this just adds a small empty strip at the bottom; downstream slicing
        # uses absolute (startingxcoord, startingycoord) offsets so it is
        # unaffected by the extra room.
        tempcanvas = np.zeros(((max(xcoords[:, 1]) - min(xcoords[:, 1]) + 1) * tileshape[0] + padding,
                               tileshape[0] + padding))
        startingxcoord = int(1 / 4 * padding)
        startingycoord = int(1 / 4 * padding)
        for y, ycoord in enumerate(list(xcoords)):
            if masterimgarray[ycoord[1], ycoord[0]] != 2:
                break
            shift = [int(shiftyrow[ycoord[1], ycoord[0], 0]),
                     int(shiftyrow[ycoord[1], ycoord[0], 1])]
            print(shift, ycoord)
            tempimg = allfovs['mip'][allfovs['coords'].index((ycoord[0], ycoord[1]))]
            if y == 0:
                tempcanvas[startingycoord:startingycoord + tempimg.shape[0],
                           startingxcoord:startingxcoord + tempimg.shape[1]] = tempimg
                startingycoord = startingycoord + tileshape[0]
            else:
                tempcanvas[startingycoord - overlapwidth + shift[1]:startingycoord - overlapwidth + shift[1] + tempimg.shape[0],
                           startingxcoord + shift[0]:startingxcoord + shift[0] + tempimg.shape[1]] = tempimg
                startingxcoord = startingxcoord + shift[0]
                startingycoord = startingycoord - overlapwidth + shift[1] + tempimg.shape[0]
            print(f'new startingx:{startingxcoord}, startingy:{startingycoord}')

        tempcanvases['x_col_SI'].append(xcoord)
        tempcanvases['x_col_EI'].append(xcoord)
        tempcanvases['y_row_SI'].append(max(xcoords[:, 1]))
        tempcanvases['y_row_EI'].append(min(xcoords[:, 1]))
        tempcanvases['canvas'].append(tempcanvas)

        print(masterimgarray)
        print(tempcanvases['x_col_SI'], tempcanvases['x_col_EI'],
              tempcanvases['y_row_SI'], tempcanvases['y_row_EI'])
        print(shiftyrow[:, :, 0])
        print(shiftyrow[:, :, 1])

    return masterimgarray, shiftxcol, shiftyrow, tempcanvases


##############
##############

def register_mip_xy_2d(dest, allfovs, masterimgarray, shiftxcol, shiftyrow, tempcanvases, tileshape, padding, overlapwidth):
    for i, image in enumerate(tempcanvases['canvas']):
        print(tempcanvases['x_col_SI'][i], tempcanvases['x_col_EI'][i],
              tempcanvases['y_row_SI'][i], tempcanvases['y_row_EI'][i], image.shape)

    # Single-column grid: nothing to register column-to-column.
    if masterimgarray.shape[1] == 1:
        print('Single-column grid: skipping column-to-column xy registration.')
        return masterimgarray, shiftxcol, shiftyrow

    # If shiftxcol has been touched already, the column-to-column registration
    # has been done elsewhere - nothing to do here.
    if np.max(shiftxcol) != 0:
        return masterimgarray, shiftxcol, shiftyrow

    startingxcoord = int(1 / 4 * padding)
    startingycoord = int(1 / 4 * padding)
    succeeded = {'start_coord': [], 'end_coord': [], 'reg_x': [], 'reg_y': []}

    print('need to join columns')
    xcoords = np.unique(np.array(tempcanvases['x_col_SI']))
    for x, xcoord in enumerate(list(xcoords)):
        if x == 0:
            continue
        xcoordup = xcoords[x - 1]
        print(f'{x},{xcoord},{x-1}, {xcoordup}')

        img = tempcanvases['canvas'][x - 1]   # left image
        imgn = tempcanvases['canvas'][x]      # right image

        # First image in a column always starts at (1/4*padding, 1/4*padding).
        numtiles = int(min(tempcanvases['y_row_SI'][x], tempcanvases['y_row_SI'][x - 1])
                       - max(tempcanvases['y_row_EI'][x], tempcanvases['y_row_EI'][x - 1]) + 1)
        print(numtiles)

        # To avoid div-by-zero, register two tiles at a time and find the average of the closest values.
        regvalues = {'shiftx': [], 'shifty': []}
        for t in np.arange(0, numtiles, 1):
            imgS = img[int(startingycoord + t * tileshape[0]):int(startingycoord + (t + 1) * tileshape[0]),
                       startingxcoord + int(7 / 8 * tileshape[1]):startingxcoord + tileshape[1]]
            imgnS = imgn[int(startingycoord + t * tileshape[0]):int(startingycoord + (t + 1) * tileshape[0]),
                         startingxcoord:startingxcoord + int(1 / 8 * tileshape[1])]
            imgSF = median_filter(imgS, [3, 3])
            imgnSF = median_filter(imgnS, [3, 3])

            print(f'shapes of canvases: {imgS.shape}, {imgnS.shape}')
            shiftx, shifty, _, _ = xyreg('ref', 'tar', imgS, imgnS, imgSF, imgnSF)
            print(imgS.shape, imgnS.shape, shiftx, shifty)
            regvalues['shiftx'].append(shiftx)
            regvalues['shifty'].append(shifty)

        xvec = np.array(regvalues['shiftx'])
        yvec = np.array(regvalues['shifty'])

        xvec = np.array(xvec[xvec != FAILED_SHIFT])
        yvec = np.array(yvec[yvec != FAILED_SHIFT])
        print(f'xvec:{xvec}, yvec:{yvec}')

        if len(xvec) > 0:
            succeeded['reg_x'] += xvec.tolist()
            succeeded['reg_y'] += yvec.tolist()
            succeeded['start_coord'] += [x - 1] * len(xvec)
            succeeded['end_coord'] += [x] * len(xvec)

        print(f'Now printing succeeded dict for {x} and {x-1}: {succeeded}')

    ############## Clean-up: recheck all failed registration to see if any
    ## of the values were close to the actual; force a value if so.
    successx = np.array(succeeded['reg_x'])
    successy = np.array(succeeded['reg_y'])
    sd_x_est = min(3, np.std(successx))  # 6 Mar 2024 - was originally max!
    sd_y_est = min(3, np.std(successy))

    avgx = _pick_robust_center(successx)
    avgy = _pick_robust_center(successy)

    print(successx, successy, sd_x_est, sd_y_est, avgx, avgy)

    ## column-by-column: extract column x/y shifts, find values near global avgx/avgy.
    ## If none, replace with avgx/avgy; else use the average of those values.
    for x, xcoord in enumerate(list(xcoords)):
        if x == 0:
            continue
        subsetx = [succeeded['reg_x'][i] for i, val in enumerate(succeeded['end_coord']) if val == x]
        subsety = [succeeded['reg_y'][i] for i, val in enumerate(succeeded['end_coord']) if val == x]

        print(f'subsetx:{subsetx}, subsety:{subsety}')

        close_x = np.array([abs(subsetx - avgx) <= 2 * sd_x_est])
        close_y = np.array([abs(subsety - avgy) <= 2 * sd_y_est])

        print(f'close_x:{close_x}, close_y:{close_y}')

        shift_x = int(avgx) if np.sum(close_x) < 1 else int(np.round(np.sum(subsetx * close_x) / np.sum(close_x)))
        shift_y = int(avgy) if np.sum(close_y) < 1 else int(np.round(np.sum(subsety * close_y) / np.sum(close_y)))

        print(f'shiftx: {shift_x}, shifty: {shift_y}')
        y_lo = int(tempcanvases['y_row_EI'][0])
        y_hi = int(tempcanvases['y_row_SI'][0] + 1)
        masterimgarray[y_lo:y_hi, xcoord - 1:xcoord + 1] = 3
        shiftxcol[y_lo:y_hi, xcoord, 0] = shift_x
        shiftxcol[y_lo:y_hi, xcoord, 1] = shift_y
        print(masterimgarray)
        print(shiftyrow[:, :, 0])
        print(shiftyrow[:, :, 1])
        print(shiftxcol[:, :, 0])
        print(shiftxcol[:, :, 1])

    return masterimgarray, shiftxcol, shiftyrow


##############
##############

def register_mip_z(allfovs, masterimgarray, shiftxcol, shiftyrow, tileshape, padding, overlapwidth):
    # Single-row grid: each column has only one FOV, no within-column z-pair to register.
    if masterimgarray.shape[0] == 1:
        print('Single-row grid: skipping within-column z registration.')
        return masterimgarray, shiftxcol, shiftyrow

    coords = np.array(allfovs['coords'])
    sortedvals = np.unique(np.sort(coords[:, 0]))

    for x, xcoord in enumerate(list(sortedvals)):
        xcoords = np.array(coords[coords[:, 0] == xcoord])
        xcoords = xcoords[xcoords[:, 1].argsort()[::-1]]
        print(f'xcoords are {xcoords}.')

        for y, ycoord in enumerate(list(xcoords)):
            if y == 0:
                continue
            ycoordup = xcoords[y - 1]
            print(f'{y},{ycoord},{y-1}, {ycoordup}')

            shiftx = int(shiftyrow[ycoord[1], ycoord[0], 0])
            shifty = int(shiftyrow[ycoord[1], ycoord[0], 1])
            print(f'shiftx:{shiftx}, shifty:{shifty}')

            img3dS = allfovs['image'][allfovs['coords'].index((ycoordup[0], ycoordup[1]))][:, int(np.round(7 / 8 * tileshape[0])):, :]
            img3dnS = allfovs['image'][allfovs['coords'].index((ycoord[0], ycoord[1]))][:, :int(np.round(1 / 8 * tileshape[0])), :]
            print(f'Now examining 3D: shapes: {img3dS.shape}, {img3dnS.shape}')

            img3dSm, img3dnSm, img3dS, img3dnS = reg.clip(np.max(img3dS, axis=0), np.max(img3dnS, axis=0),
                                                          img3dS, img3dnS, shiftx, shifty)
            print(f'After 3D clipping, shapes are: {img3dS.shape}, {img3dnS.shape}')

            shiftz, pccz, diff_max_z = zreg(img3dS, img3dnS, fccutoff=0.5,
                                            numrowtiles=1, numcoltiles=3,
                                            rowdivisor=1, coldivisor=2,
                                            linespread=10, dsz=3, divisor=3)
            shiftz = _resolve_shiftz(shiftz, pccz, diff_max_z)

            shiftyrow[ycoord[1], ycoord[0], 2] = shiftz
            shiftyrow[ycoord[1], ycoord[0], 3] = pccz
            shiftyrow[ycoord[1], ycoord[0], 4] = diff_max_z

    return masterimgarray, shiftxcol, shiftyrow


##############
##############

def zreg(img3dS, img3dnS, fccutoff=0.5, numrowtiles=1, numcoltiles=3, rowdivisor=1, coldivisor=2,
         linespread=10, dsz=3, divisor=3):
    startr, endr, startc, endc = find_roi_uneven(np.max(img3dS, axis=0), np.max(img3dnS, axis=0),
                                                  numrowtiles=numrowtiles, numcoltiles=numcoltiles,
                                                  rowdivisor=rowdivisor, coldivisor=coldivisor)
    img3dS = img3dS[:, startr:endr, startc:endc]
    img3dnS = img3dnS[:, startr:endr, startc:endc]
    return _zreg(img3dS, img3dnS, fccutoff=fccutoff, linespread=linespread, dsz=dsz, divisor=divisor)


##############
##############

def _zreg(reffz3, tarfz3, fccutoff=0.5, linespread=10, dsz=3, divisor=3):
    st = time.time()

    reffc = reg.get_fluor_curve(reffz3)
    tarfc = reg.get_fluor_curve(tarfz3)
    diff_max_z = int(np.where(reffc == np.max(reffc))[0] - np.where(tarfc == np.max(tarfc))[0])

    shift_pcc, _, _ = phase_cross_correlation(reffz3, tarfz3, normalization=None)
    pccz = shift_pcc[0]

    refind = [i for i, val in enumerate(reffc) if (val > fccutoff * max(reffc) and i % divisor == 0)]
    tarind = [i for i, val in enumerate(tarfc) if (val > fccutoff * max(tarfc) and i % divisor == 0)]
    print(f'Searching ref z in:{refind}, searching tar z in:{tarind}')

    masterz = {'i': [], 'j': [], 'shiftzest': []}
    rowvec = np.arange(10, reffz3.shape[1], linespread)
    colvec = np.arange(10, reffz3.shape[2], linespread)

    masterz = get_z_est_rows(rowvec, reffz3, tarfz3, refind, tarind, masterz,
                             window=8 * dsz, shiftzest=0, cutoff=99, zclipnorm='True')
    masterz = get_z_est_cols(colvec, reffz3, tarfz3, refind, tarind, masterz,
                             window=8 * dsz, shiftzest=0, cutoff=99, zclipnorm='True')

    et = time.time()
    if len(masterz.get('shiftzest')) > 0:
        _ests = Counter(masterz.get('shiftzest'))
        print('Masterz_get shiftzest:', masterz.get('shiftzest'))
        print(f'Estimated values from Counter: {_ests}')
        mostcommon = _ests.most_common()
        print(f'Most common final values from Counter: {mostcommon}')
        shiftz, _ = reg.getmode(mostcommon, 1)
    else:
        shiftz = 'NA'

    zfinaltime = et - st
    print('ENDING Z_REGISTRATION: final:', str(shiftz), 'planes', str(int(zfinaltime)), ' seconds',
          'pcc:', str(pccz), 'planes ', ' diff_max_z:', str(diff_max_z), 'planes')
    return shiftz, pccz, diff_max_z


##############
##############

def _get_z_est_along_vec(vec, ref3, tar3, refind, tarind, masterzest, axis,
                        window=8, shiftzest=0, cutoff=99, zclipnorm='True'):
    """Shared body for get_z_est_rows (axis=1) and get_z_est_cols (axis=2).

    Samples a window of `window` planes along z, takes the slice along `vec`
    on the given axis, stacks the per-vec-element slabs vertically, and runs
    phase-cross-correlation between ref and tar. The (i, j) pair with the
    smallest correlation error is recorded into masterzest.
    """
    for i in refind:
        _mins = {'i': [], 'j': [], 'shift': [], 'error': []}

        for j in tarind:
            if not (i + window < ref3.shape[0] and j + window < tar3.shape[0]
                    and (i - j) <= (shiftzest + cutoff)
                    and (i - j) >= (shiftzest - cutoff)):
                continue

            if axis == 1:
                refm2 = ref3[i:window + i, vec, :]
                newm2 = tar3[j:window + j, vec, :]
                nvec, other = refm2.shape[1], refm2.shape[2]
            else:  # axis == 2
                refm2 = ref3[i:window + i, :, vec]
                newm2 = tar3[j:window + j, :, vec]
                nvec, other = refm2.shape[2], refm2.shape[1]

            refm3 = np.zeros((refm2.shape[0] * nvec, other))
            newm3 = np.zeros((newm2.shape[0] * nvec, other))
            for k in range(nvec):
                if axis == 1:
                    refm3[k * refm2.shape[0]:(k + 1) * refm2.shape[0], :] = refm2[:, k, :]
                    newm3[k * newm2.shape[0]:(k + 1) * newm2.shape[0], :] = newm2[:, k, :]
                else:
                    refm3[k * refm2.shape[0]:(k + 1) * refm2.shape[0], :] = refm2[:, :, k]
                    newm3[k * newm2.shape[0]:(k + 1) * newm2.shape[0], :] = newm2[:, :, k]

            if zclipnorm == 'True':
                refm3 = np.clip(refm3, np.percentile(refm3, 1), np.percentile(refm3, 98))
                refm3n = reg.normalize_image(refm3, np.min(refm3), np.max(refm3))
                newm3 = np.clip(newm3, np.percentile(newm3, 1), np.percentile(newm3, 98))
                newm3n = reg.normalize_image(newm3, np.min(newm3), np.max(newm3))

            if zclipnorm == 'False':
                shift, error, _ = phase_cross_correlation(refm3, newm3, normalization=None)
            else:
                shift, error, _ = phase_cross_correlation(refm3n, newm3n, normalization=None)

            _mins['i'].append(i)
            _mins['j'].append(j)
            _mins['shift'].append(shift)
            _mins['error'].append(error)

        if len(_mins.get('error')) > 0:
            minind = np.array(_mins.get('error')).argmin()
            masterzest['i'].append(_mins.get('i')[minind])
            masterzest['j'].append(_mins.get('j')[minind])
            masterzest['shiftzest'].append(_mins.get('i')[minind] - _mins.get('j')[minind])

    return masterzest


def get_z_est_cols(colvec, ref3, tar3, refind, tarind, masterzest,
                   window=8, shiftzest=0, cutoff=99, zclipnorm='True'):
    return _get_z_est_along_vec(colvec, ref3, tar3, refind, tarind, masterzest, axis=2,
                                window=window, shiftzest=shiftzest, cutoff=cutoff, zclipnorm=zclipnorm)


def get_z_est_rows(rowvec, ref3, tar3, refind, tarind, masterzest,
                   window=8, shiftzest=0, cutoff=99, zclipnorm='True'):
    return _get_z_est_along_vec(rowvec, ref3, tar3, refind, tarind, masterzest, axis=1,
                                window=window, shiftzest=shiftzest, cutoff=cutoff, zclipnorm=zclipnorm)


##############
##############

def find_roi_uneven(refmcn, tarmcn, numrowtiles=3, numcoltiles=3, rowdivisor=2, coldivisor=2):
    """Find the (numrowtiles x numcoltiles)-tile sub-region of (refmcn, tarmcn)
    with the highest combined variance."""
    rsubsetvar = np.zeros(numrowtiles * numcoltiles)
    tsubsetvar = np.zeros(numrowtiles * numcoltiles)

    lengthr = int(np.round(refmcn.shape[0] / rowdivisor))
    lengthc = int(np.round(refmcn.shape[1] / coldivisor))

    jumplengthr = int(np.floor((refmcn.shape[0] - lengthr) / (numrowtiles - 1))) if numrowtiles > 1 else 0
    jumplengthc = int(np.floor((refmcn.shape[1] - lengthc) / (numcoltiles - 1))) if numcoltiles > 1 else 0

    print(f'Looking at: {refmcn.shape}, {tarmcn.shape}, {numrowtiles}, {numcoltiles},  {lengthr}, {lengthc}.')

    for i in range(numrowtiles):
        for j in range(numcoltiles):
            rsubset = refmcn[i * jumplengthr:i * jumplengthr + lengthr,
                             j * jumplengthc:j * jumplengthc + lengthc]
            tsubset = tarmcn[i * jumplengthr:i * jumplengthr + lengthr,
                             j * jumplengthc:j * jumplengthc + lengthc]
            rsubsetvar[i * numcoltiles + j] = np.var(rsubset)
            tsubsetvar[i * numcoltiles + j] = np.var(tsubset)

    totalvar = np.add(rsubsetvar / np.sum(rsubsetvar), tsubsetvar / np.sum(tsubsetvar))
    maxvar = totalvar.argmax()
    subseti = int(np.floor(maxvar / numcoltiles))
    subsetj = maxvar % numcoltiles

    startr = subseti * jumplengthr
    endr = subseti * jumplengthr + lengthr
    startc = subsetj * jumplengthc
    endc = subsetj * jumplengthc + lengthc

    print(f'Looking at {totalvar}, {maxvar}, {subseti}, {subsetj}; region: {startr}:{endr}, {startc}:{endc}.')

    return startr, endr, startc, endc


##############
##############

'''
We do z-registration block by block within the two columns, i.e. (11,3) is
registered to (10,3), (12,4) is registered to (11,4) and so on. If a column
has n rows, we should do at least 3n/4 comparisons. Z shifts are then averaged.

Consider a starting block, i.e. (x+1, 0) registered to (x, 0). The initial
shift, we define as C0. Now consider any pair of blocks (x+1, k) registering
to (x, k) with k>0. This shift, we call D. Then, letting the Z shift between
(x,0) and (x,k) be Z1, and (x+1,0) and (x+1,k) be Z2, the following relation
holds: C - Z1 - D + Z2 = 0.

For each i in {0,n}, we calculate D, Z1 and Z2, and calculate Ci = D + Z1 - Z2.
In this way, we average over all values of i, to determine the average
difference across the column.
'''


def register_mip_2d_z(allfovs, masterimgarray, shiftxcol, shiftyrow, tileshape, padding, overlapwidth):
    # Single-column grid: nothing to register column-to-column.
    if masterimgarray.shape[1] == 1:
        print('Single-column grid: skipping column-to-column z registration.')
        return shiftxcol

    coords = np.array(allfovs['coords'])
    sortedvals = np.unique(np.sort(coords[:, 0]))
    print(sortedvals)
    print(shiftyrow[:, :, 0])
    print(shiftyrow[:, :, 1])
    print(shiftyrow[:, :, 2])

    # Start from the 2nd column, registering to the left 1st column.
    for x, xcoord in enumerate(list(sortedvals)[1:]):
        xcoords = np.array(coords[coords[:, 0] == xcoord])
        xcoords = xcoords[xcoords[:, 1].argsort()[::-1]]
        print(f'xcoords are {xcoords}.')

        xcoordsleft = np.array(coords[coords[:, 0] == xcoord - 1])
        xcoordsleft = xcoordsleft[xcoordsleft[:, 1].argsort()[::-1]]
        print(f'xcoordsleft are {xcoordsleft}.')

        print(np.flip(xcoords)[0])
        rowbyrowshifts = []
        masterzshifts = []

        for y, ycoord in enumerate(list(xcoords)):
            ycoordleft = xcoordsleft[y]
            print(f'{y},{ycoord},{y}, {ycoordleft}')

            tempimgleft = allfovs['image'][allfovs['coords'].index((ycoordleft[0], ycoordleft[1]))]
            tempimg = allfovs['image'][allfovs['coords'].index((ycoord[0], ycoord[1]))]

            zshiftleft = int(np.sum(shiftyrow[ycoordleft[1]:, ycoordleft[0], 2]))
            zshift = int(np.sum(shiftyrow[ycoord[1]:, ycoord[0], 2]))
            print(f'zshiftleft: {zshiftleft}, zshift: {zshift}')

            # These shifts are columnar:
            xshift = int(shiftxcol[ycoord[1], ycoord[0], 0])
            yshift = int(shiftxcol[ycoord[1], ycoord[0], 1])

            tempimgleft = tempimgleft[:, :, int(7 / 8 * tileshape[1]):]
            tempimg = tempimg[:, :, :int(1 / 8 * tileshape[1])]
            print(f'x y shift for this column is {xshift}, {yshift}; shapes:{tempimgleft.shape}, {tempimg.shape}.')

            _, _, tempimgleftC, tempimgC = reg.clip(np.max(tempimgleft, axis=0), np.max(tempimg, axis=0),
                                                    tempimgleft, tempimg, xshift, yshift)
            print(f'after clipping, shapes:{tempimgleftC.shape}, {tempimgC.shape}.')
            shiftz, pccz, diff_max_z = zreg(tempimgleftC, tempimgC,
                                            fccutoff=0.4, numrowtiles=3, numcoltiles=1,
                                            rowdivisor=2, coldivisor=1,
                                            linespread=10, dsz=3, divisor=3)
            shiftz = _resolve_shiftz(shiftz, pccz, diff_max_z)
            print({shiftz}, {pccz}, {diff_max_z})

            rowbyrowshifts.append(shiftz)
            if y == 0:
                masterzshifts.append(shiftz)
            else:
                newval = zshiftleft + shiftz - zshift
                masterzshifts.append(newval)

        colzshift = int(np.mean(masterzshifts))
        shiftxcol[min(xcoords[:, 1]):max(xcoords[:, 1]) + 1, xcoord, 2] = colzshift

        print(f'final values: RbyRshifts:{rowbyrowshifts}, masterzshifts:{masterzshifts}')

    return shiftxcol


##############
##############

def generate_coord_numbers_3d(dest, allfovs, masterimgarray, shiftxcol, shiftyrow, tileshape, padding, overlapwidth):
    mastercoordarray = np.zeros((masterimgarray.shape[0], masterimgarray.shape[1], 3))

    print(allfovs['coords'])
    coordarray = np.array(allfovs['coords'])
    startcoordind = allfovs['coords'].index((min(coordarray[:, 0]), max(coordarray[:, 1])))
    print(startcoordind, allfovs['coords'][startcoordind])

    xcoords = list(np.unique(coordarray[:, 0]))
    ycoords = list(np.unique(coordarray[:, 1]))
    ycoords.reverse()

    startingxcoordnew = int(1 / 4 * padding)
    startingycoordnew = int(1 / 4 * padding)
    startingzcoordnew = 0

    for x, xcoord in enumerate(xcoords):
        # assemble per-column first
        startingxcoordtemp = 0
        startingycoordtemp = 0
        startingzcoordtemp = 0

        if x > 0:  # extract the columnar shift values
            colxshift = int(shiftxcol[np.max(ycoords), xcoord, 0])
            rowyshift = int(shiftxcol[np.max(ycoords), xcoord, 1])
            zshift = int(shiftxcol[np.max(ycoords), xcoord, 2])
            print(f'existing shifts are:\n startingxcoordtemp:{startingxcoordtemp}, '
                  f'startingycoordtemp:{startingycoordtemp}, startingzcoordtemp:{startingzcoordtemp}\n '
                  f'colxshift is: {colxshift}, rowyshift is: {rowyshift}, zshift is:{zshift}')
            startingxcoordnew = startingxcoordnew + 7 / 8 * tileshape[1] + colxshift
            startingycoordnew = startingycoordnew + rowyshift
            startingzcoordnew = startingzcoordnew + zshift  # 9 Jan 2024: added the `startingzcoordnew +`

        for y, ycoord in enumerate(ycoords):
            shift = [int(shiftyrow[ycoord, xcoord, 0]),
                     int(shiftyrow[ycoord, xcoord, 1]),
                     int(shiftyrow[ycoord, xcoord, 2])]
            print(shift, xcoord, ycoord, y)

            if y == 0:
                startingxcoordtemp = int(startingxcoordnew + shift[0])
                startingycoordtemp = int(startingycoordnew + shift[1])
                startingzcoordtemp = int(startingzcoordnew + shift[2])
            else:
                startingxcoordtemp = int(startingxcoordtemp + shift[0])
                startingycoordtemp = int(startingycoordtemp + shift[1])
                startingzcoordtemp = int(startingzcoordtemp + shift[2])

            print(f'y={y}, xcoord: {startingxcoordtemp}, ycoord: {startingycoordtemp}, zcoord: {startingzcoordtemp}')
            mastercoordarray[ycoord, xcoord, 0] = startingxcoordtemp
            mastercoordarray[ycoord, xcoord, 1] = startingycoordtemp
            mastercoordarray[ycoord, xcoord, 2] = startingzcoordtemp

            startingycoordtemp = int(startingycoordtemp + 7 / 8 * tileshape[0])

    print(mastercoordarray[:, :, 0])
    print(mastercoordarray[:, :, 1])
    print(mastercoordarray[:, :, 2])

    return mastercoordarray


##############
##############

def run_stitching_pipeline(args):
    if not (os.path.exists(args.source) and os.path.exists(args.dest)):
        return
    print('Source and destination directories found.')

    layout_file = getattr(args, 'fov_layout', '') or ''
    nrow = int(args.master_nrow) if args.master_nrow else 0
    ncol = int(args.master_ncol) if args.master_ncol else 0

    if not layout_file and (nrow == 0 or ncol == 0):
        print('Error: Please specify the total number of rows and columns in the stitched image '
              'using --master_nrow and --master_ncol, or pass --fov_layout to point to an '
              'explicit grid-layout text file.')
        return

    allfovs = None
    if args.fovs != 'all':
        fovs = args.fovs.split(',')
        print(fovs)
        if len(fovs) > 0:
            for fov in fovs:
                allfovs = utils.parse_directory_stitching(args.source, fov, allfovs)
        allfovs, tile = utils._read_ims_stitching(allfovs, fovstoread=[])
    else:
        allfovs = utils.parse_directory_stitching(args.source, 999, allfovs)
        allfovs, tile = utils._read_ims_stitching(allfovs, fovstoread=[])

    tileshape = (tile.shape[0], tile.shape[1])
    padding = int(1 / 16 * tileshape[0])
    overlapwidth = int(1 / 8 * tileshape[0])

    shiftxcol, shiftyrow, masterimgarray, allfovs = generate_coords(
        allfovs, nrow, ncol, layout_file=layout_file)
    nrow_grid, ncol_grid = masterimgarray.shape
    print(f'Grid is {nrow_grid} rows x {ncol_grid} columns.')

    # register_mip_xy always runs (it also builds tempcanvases). It internally
    # skips the within-column reg loop if nrow_grid == 1.
    masterimgarray, shiftxcol, shiftyrow, tempcanvases = register_mip_xy(
        args.dest, allfovs, masterimgarray, shiftxcol, shiftyrow,
        tileshape, padding, overlapwidth, int(args.threshold))

    # Column-to-column xy registration: only relevant when there's more than one column.
    if ncol_grid > 1:
        masterimgarray, shiftxcol, shiftyrow = register_mip_xy_2d(
            args.dest, allfovs, masterimgarray, shiftxcol, shiftyrow, tempcanvases,
            tileshape, padding, overlapwidth)

    if args.do_3D == 'True':
        # Within-column z-registration: only relevant when there's more than one row.
        if nrow_grid > 1:
            masterimgarray, shiftxcol, shiftyrow = register_mip_z(
                allfovs, masterimgarray, shiftxcol, shiftyrow, tileshape, padding, overlapwidth)
        # Column-to-column z-registration: only relevant when there's more than one column.
        if ncol_grid > 1:
            shiftxcol = register_mip_2d_z(
                allfovs, masterimgarray, shiftxcol, shiftyrow, tileshape, padding, overlapwidth)

    mastercoordarray3d = generate_coord_numbers_3d(
        args.dest, allfovs, masterimgarray, shiftxcol, shiftyrow, tileshape, padding, overlapwidth)

    fmin = str(min(allfovs['fov']))
    fmax = str(max(allfovs['fov']))
    np.save(os.path.join(args.dest, f'Master_coord_array_3D_Dapi_F{fmin}-{fmax}'), mastercoordarray3d)
    np.save(os.path.join(args.dest, f'Master_shiftyrow_3D_Dapi_F{fmin}-{fmax}'), shiftyrow)
    np.save(os.path.join(args.dest, f'Master_shiftxcol_3D_Dapi_F{fmin}-{fmax}'), shiftxcol)


##############
##############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='')
    parser.add_argument('--dest', default='')
    parser.add_argument('--master_nrow', default=0)
    parser.add_argument('--master_ncol', default=0)
    parser.add_argument('--fov_layout', default='',
                        help='Optional path to a text file specifying the FOV grid layout. '
                             'Each line is one row top-to-bottom; within a line, FOVs are '
                             'comma-separated left-to-right. Overrides --master_nrow/--master_ncol.')
    parser.add_argument('--do_3D', default='True')
    parser.add_argument('--fovs', default='all')
    parser.add_argument('--threshold', default=10)
    args = parser.parse_args()

    run_stitching_pipeline(args)

##############
##############
