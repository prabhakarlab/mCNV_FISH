"""
Post-processing of the per-target shifts produced by qc_registration:
chains the shifts back to the prehyb DAPI reference, picks best-aligned
planes per channel, and produces before/after MIP overlay plots for QC.
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import qc_registration_reg_functions as reg
import qc_registration_utils as utils


# Sentinel matching the one in reg_functions: 999 means "could not register".
FAILED_SHIFT = 999

##############
##############

# Colourmaps used for the two-channel red/green overlays.
_CM_RED = LinearSegmentedColormap.from_list("Custom", [(0, 0, 0), (1, 0, 0)], N=256)
_CM_GREEN = LinearSegmentedColormap.from_list("Custom", [(0, 0, 0), (0, 1, 0)], N=256)


def _overlay(ax, refp, tarp, title):
    ax.imshow(refp, cmap=_CM_RED, vmin=np.percentile(refp, 5),
              vmax=np.percentile(refp, 98), alpha=0.5)
    ax.imshow(tarp, cmap=_CM_GREEN, vmin=np.percentile(tarp, 5),
              vmax=np.percentile(tarp, 98), alpha=0.4)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


##############
##############

def plot_array_aligned(imagedict, dest, fov, keyword, subplotwidth=4, subplotheight=4,
                      hspace=0.17, wspace=0.17):
    nummips = len(imagedict.get('imagename'))
    numcol = int(np.ceil(np.sqrt(nummips)))
    numrow = int(np.ceil(nummips / numcol))

    if not (numcol > 1 and numrow > 1):
        print('Not enough registered bits to generate aligned image. Skipping.')
        return

    figure, axis = plt.subplots(nrows=numrow, ncols=numcol,
                                figsize=(subplotheight * numcol, subplotwidth * numrow))

    for i in range(numrow):
        for j in range(numcol):
            idx = i * numcol + j
            if idx >= nummips:
                axis[i, j].axis('off')
                continue

            flag = imagedict.get('flag')[idx]
            name = imagedict.get('imagename')[idx]
            refp = imagedict.get('refplane')[idx]
            tarp = imagedict.get('tarplane')[idx]

            if flag == 'OK':
                _overlay(axis[i, j], refp, tarp, f'Prehyb_DAPI_(r)\n {name}_(g)')
            elif flag == 'DND':
                axis[i, j].set_title(f'Prehyb_DAPI_(r)\n {name}_(g)\n NOT REG. DUE TO FLAG', fontsize=10)
                axis[i, j].axis('off')

    figure.subplots_adjust(hspace=hspace, wspace=wspace)
    figure.savefig(os.path.join(dest, 'Post_reg_alignment_FOV_' + fov + '_' + keyword + '.png'),
                   dpi=300, format='png', bbox_inches='tight')
    plt.close(figure)

##############
##############

def plot_array_prealigned(imagedict, dest, fov, keyword, subplotwidth=4, subplotheight=4,
                         hspace=0.17, wspace=0.17):
    nummips = len(imagedict.get('imagename'))
    numcol = int(np.ceil(np.sqrt(nummips)))
    numrow = int(np.ceil(nummips / numcol))
    figure, axis = plt.subplots(nrows=numrow, ncols=numcol,
                                figsize=(subplotheight * numcol, subplotwidth * numrow))

    for i in range(numrow):
        for j in range(numcol):
            idx = i * numcol + j
            if idx >= nummips:
                axis[i, j].axis('off')
                continue
            refp = imagedict.get('refplane')[idx]
            tarp = imagedict.get('tarplane')[idx]
            name = imagedict.get('imagename')[idx]
            _overlay(axis[i, j], refp, tarp, f'Prehyb_DAPI_(r)\n {name}_(g)')

    figure.subplots_adjust(hspace=hspace, wspace=wspace)
    figure.savefig(os.path.join(dest, 'Pre_reg_alignment_FOV_' + fov + '_' + keyword + '.png'),
                   dpi=300, format='png', bbox_inches='tight')
    plt.close(figure)

##############
##############

def _find_ref_id(allchs):
    """Return the index of the prehyb DAPI (cycle 99, ch 2) entry in allchs."""
    return [i for i, val in enumerate(allchs.get('imagename')) if 'cycle_99_ch_2' in val][0]


def _build_target_plane(tarstack, tarname):
    """MIP of the target stack, with extra clipping for antibody VIM staining."""
    tarplane = np.max(tarstack, axis=0)
    if 'cycle_88_ch_2' in tarname:
        tarplane = np.clip(tarplane, np.percentile(tarplane, 1), np.percentile(tarplane, 90))
    return tarplane


def _run_pre_mip_alignment(fov, allchs, df, dest, keyword):
    refid = _find_ref_id(allchs)
    refstack = allchs['image'][refid]
    refplane = np.max(refstack, axis=0)

    aligned = {'imagename': [], 'refplane': [], 'tarplane': []}
    for tarname in df.get('tar'):
        tarid = [i for i, val in enumerate(allchs.get('imagename')) if tarname in val][0]
        tarstack = allchs['image'][tarid]
        aligned['imagename'].append(tarname)
        aligned['refplane'].append(refplane)
        aligned['tarplane'].append(_build_target_plane(tarstack, tarname))

    plot_array_prealigned(imagedict=aligned, dest=dest, fov=fov, keyword=keyword)

##############
##############

def _run_post_alignment(fov, allchs, df, dest, keyword):
    refid = _find_ref_id(allchs)
    refstack = allchs['image'][refid]

    aligned = {'imagename': [], 'refplane': [], 'tarplane': [], 'flag': []}
    for tid, tarname in enumerate(df.get('tar')):
        # Recompute tarid - allchs also contains prehyb_dapi
        tarid = [i for i, val in enumerate(allchs.get('imagename')) if tarname in val][0]
        tarstack = allchs['image'][tarid]
        shiftx = int(df.get('x')[tid])
        shifty = int(df.get('y')[tid])

        refplane = np.max(refstack, axis=0)
        tarplane = _build_target_plane(tarstack, tarname)

        if shiftx != FAILED_SHIFT:
            refp, tarp, _, _ = reg.clip(refplane, tarplane, refstack, tarstack, shiftx, shifty)
            aligned['imagename'].append(tarname)
            aligned['refplane'].append(refp)
            aligned['tarplane'].append(tarp)
            aligned['flag'].append('OK')
        else:
            aligned['imagename'].append(tarname)
            aligned['refplane'].append(refplane)
            aligned['tarplane'].append(tarplane)
            aligned['flag'].append('DND')

    plot_array_aligned(imagedict=aligned, dest=dest, fov=fov, keyword=keyword)

##############
##############

def seqadd(df, currvals, tar, ref, xcolname, ycolname, zcolname):
    """Accumulate the shift from `tar` to its reference. Used iteratively to
    walk back to the prehyb DAPI."""
    row = df[df['tarname'] == tar].index[0]
    rowref = df['refname'][row]

    z_raw = df[zcolname][row]
    if isinstance(z_raw, str):
        z = int(z_raw[1:-1])
    elif isinstance(z_raw, list):
        z = int(z_raw[0])
    else:
        z = int(z_raw)

    newvals = np.array((df[xcolname][row], df[ycolname][row], z))
    return (newvals if currvals is None else np.add(currvals, newvals)), rowref

##############
##############

def _walk_to_dapi(fovtouse, element, xcolname, ycolname, zcolname):
    """Walk from `element` back to cycle_99_ch_2 (prehyb DAPI) accumulating shifts.

    Returns (currvals, rowref) where currvals is (x, y, z) total shift and
    rowref is the final reference name (will contain 'cycle_99_ch_2').
    """
    row = fovtouse[fovtouse['tarname'] == element].index[0]
    rowref = fovtouse['refname'][row]
    currvals, rowref = seqadd(fovtouse, None, element, rowref, xcolname, ycolname, zcolname)
    while 'cycle_99_ch_2' not in rowref:
        currvals, rowref = seqadd(fovtouse, currvals, rowref, 'cycle_99_ch_2',
                                  xcolname, ycolname, zcolname)
    return currvals, rowref


def _compute_refind_tarind(zshift, tarfc, reffc):
    """Best-aligned ref/tar plane indices given a z-shift between them.

    Multiplies the z-shifted target curve by the reference and picks the peak.
    """
    newtarfc = np.zeros(tarfc.shape[0])
    if zshift < 0:
        newtarfc[:zshift] = tarfc[0:(tarfc.shape[0] + zshift)]
    elif zshift > 0:
        newtarfc[zshift:] = tarfc[zshift:]
    mul = np.multiply(reffc, newtarfc)
    refind = mul.argmax()
    tarind = refind - zshift
    return refind, tarind


def _parse_fc_string(fc_string):
    """Parse a fluor_curve numpy-array string from a CSV cell."""
    return np.array([float(x) for x in fc_string[1:-1].split()])

##############
##############

def _run_qc_registration_post_processing(dest, allchs, failedregdata, regdata, regattempts,
                                         qcparams, fov, xcolname, ycolname, zcolname, saveword):
    """Post-processing variant that takes in-memory qcparams (fluor_curves as np arrays)
    and inserts FAILED_SHIFT placeholders for any target with no registration row.
    """
    fovdata = pd.DataFrame(regdata)
    qcdata = pd.DataFrame(qcparams)

    fovtouse = fovdata[[xcolname, ycolname, zcolname, 'refname', 'tarname']]
    qctouse = qcdata[['imagename', 'fluor_curve']]

    # Precompute the prehyb-DAPI peak plane.
    dapi_idx = qctouse[qctouse['imagename'].str.contains('cycle_99_ch_2')].index[0]
    dapimaxp = np.argmax(qctouse['fluor_curve'][dapi_idx])

    df = {'x': [], 'y': [], 'z': [], 'ref': [], 'tar': [],
          'refind': [], 'tarind': [], 'prehyb_Dapi_maxplane': []}

    for element in qctouse['imagename']:
        if 'cycle_99_ch_2' in element:
            continue

        tarexists = fovtouse[fovtouse['tarname'].str.contains(element)].index
        if len(tarexists) == 0:
            # No reference was found for this target - record a placeholder.
            utils._update_data(df, {
                'x': FAILED_SHIFT, 'y': FAILED_SHIFT, 'z': FAILED_SHIFT,
                'ref': 'No reference', 'tar': element,
                'refind': FAILED_SHIFT, 'tarind': FAILED_SHIFT,
                'prehyb_Dapi_maxplane': dapimaxp,
            })
            continue

        currvals, rowref = _walk_to_dapi(fovtouse, element, xcolname, ycolname, zcolname)
        zshift = int(currvals[2])

        tarrow = qctouse[qctouse['imagename'] == element].index[0]
        refrow = qctouse[qctouse['imagename'].str.contains('cycle_99_ch_2')].index[0]
        tarfc = qctouse['fluor_curve'][tarrow]
        reffc = qctouse['fluor_curve'][refrow]
        refind, tarind = _compute_refind_tarind(zshift, tarfc, reffc)

        utils._update_data(df, {
            'x': currvals[0], 'y': currvals[1], 'z': currvals[2],
            'ref': rowref, 'tar': element,
            'refind': refind, 'tarind': tarind,
            'prehyb_Dapi_maxplane': dapimaxp,
        })

    dfpd = pd.DataFrame(df)
    dfpd.to_csv(os.path.join(dest, 'FINAL_Reg_fov_' + fov + '_' + saveword + '.csv'),
                float_format='%.4f')

    st = time.time()
    _run_pre_mip_alignment(fov, allchs, df, dest, saveword)
    _run_post_alignment(fov, allchs, df, dest, saveword)
    print(f'time taken: {time.time() - st} seconds')

##############
##############

def run_qc_registration_post_processing(args):
    """CSV-driven variant: reads Reg_report and QC_Rpt CSVs from disk. Skips
    any target that has no row in the reg report (no FAILED_SHIFT placeholder).
    """
    if not (os.path.exists(args.source) and os.path.exists(args.sourceim)):
        return
    print('Source and source-image directories found.')

    os.chdir(args.source)
    xcolname, ycolname, zcolname = args.xcoltouse, args.ycoltouse, args.zcoltouse

    Failed = glob.glob('FAILED' + '*.csv')
    Attempts = glob.glob('Reg_attempts' + '*.csv')
    QC = glob.glob('QC_Rpt' + '*.csv')
    Reg = glob.glob('Reg_report' + '*.csv')

    # Each saveword corresponds to one run/configuration.
    difftries = []
    for val in Reg:
        saveword = val.split('_')[-1][:-4]
        if saveword not in difftries:
            difftries.append(saveword)

    for saveword in difftries:
        RegF = sorted(v for v in Reg if saveword in v)
        print(f'saveword: {saveword}, {RegF}')

        for val in RegF:
            fov = val.split('_')[3]
            qc = [v2 for v2 in QC if '_' + fov + '.csv' in v2][0]
            fovdata = pd.read_csv(os.path.join(args.source, val))
            qcdata = pd.read_csv(os.path.join(args.source, qc))

            fovtouse = fovdata[[xcolname, ycolname, zcolname, 'refname', 'tarname']]
            qctouse = qcdata[['imagename', 'fluor_curve']]

            df = {'x': [], 'y': [], 'z': [], 'ref': [], 'tar': [], 'refind': [], 'tarind': []}

            for element in qctouse['imagename']:
                if 'cycle_99_ch_2' in element:
                    continue
                tarexists = fovtouse[fovtouse['tarname'].str.contains(element)].index
                if len(tarexists) == 0:
                    continue  # CSV path skips rather than inserting placeholders.

                currvals, rowref = _walk_to_dapi(fovtouse, element, xcolname, ycolname, zcolname)
                zshift = int(currvals[2])

                tarrow = qctouse[qctouse['imagename'] == element].index[0]
                refrow = qctouse[qctouse['imagename'].str.contains('cycle_99_ch_2')].index[0]
                # CSVs store fluor curves as the string-repr of the array.
                tarfc = _parse_fc_string(qctouse['fluor_curve'][tarrow])
                reffc = _parse_fc_string(qctouse['fluor_curve'][refrow])
                refind, tarind = _compute_refind_tarind(zshift, tarfc, reffc)

                utils._update_data(df, {
                    'x': currvals[0], 'y': currvals[1], 'z': currvals[2],
                    'ref': rowref, 'tar': element,
                    'refind': refind, 'tarind': tarind,
                })

            dfpd = pd.DataFrame(df)
            dfpd.to_csv(os.path.join(args.source,
                        'FINAL_Reg_fov_' + fov + '_' + saveword + args.keyword + '.csv'),
                        float_format='%.4f')

            print(f'{args.source}, {fov}')
            st = time.time()
            allchs = utils.parse_directory(args.sourceim, fov)
            allchs = utils._read_ims(allchs)
            _run_pre_mip_alignment(fov, allchs, df, args.source, args.keyword)
            _run_post_alignment(fov, allchs, df, args.source, args.keyword)
            print(f'time taken: {time.time() - st} seconds')

##############
##############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='')
    parser.add_argument('--sourceim', default='')
    parser.add_argument('--xcoltouse', default='shiftx_final')
    parser.add_argument('--ycoltouse', default='shifty_final')
    parser.add_argument('--zcoltouse', default='shiftz_final')
    parser.add_argument('--keyword', default='')
    args = parser.parse_args()

    run_qc_registration_post_processing(args)
