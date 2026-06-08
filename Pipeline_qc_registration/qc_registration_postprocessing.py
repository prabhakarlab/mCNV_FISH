"""
Standalone postprocess pipeline (testing/with-updates variant).

Reads per-FOV Reg_report CSVs and the corresponding QC CSVs, chains
target-to-reference shifts back to prehyb DAPI for each target, applies
cross-FOV outlier correction on x/y, and emits FINAL_Reg_fov_*.csv files.
Optionally draws pre/post-alignment MIPs from the imaged stacks.

This is a more recent / more aggressive variant of postprocess_functions.py;
the two are kept separate because they emit slightly different column sets
and have different missing-target semantics.
"""
import argparse
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import qc_registration_reg_functions as reg
import qc_registration_utils as utils


# Sentinel matching reg_functions: 999 means "could not register".
FAILED_SHIFT = 999

# Colourmaps used for the red/green two-channel overlays.
_CM_RED = LinearSegmentedColormap.from_list("Custom", [(0, 0, 0), (1, 0, 0)], N=256)
_CM_GREEN = LinearSegmentedColormap.from_list("Custom", [(0, 0, 0), (0, 1, 0)], N=256)


def _overlay(ax, refp, tarp, title, title_fontsize=10):
    """Render a red/green dual-imshow panel."""
    ax.imshow(refp, cmap=_CM_RED, vmin=np.percentile(refp, 5),
              vmax=np.percentile(refp, 98), alpha=0.5)
    ax.imshow(tarp, cmap=_CM_GREEN, vmin=np.percentile(tarp, 5),
              vmax=np.percentile(tarp, 98), alpha=0.4)
    ax.set_title(title, fontsize=title_fontsize)
    ax.axis('off')


def _vim_clip(plane):
    """Antibody-VIM intensity clip used by the alignment plotters."""
    return np.clip(plane, np.percentile(plane, 1), np.percentile(plane, 90))


def _fov_from_filename(filename):
    """Extract FOV string from Reg_*_fov_<X>_*.csv. Strips a trailing '.csv'
    if the saveword was missing during registration."""
    fov = filename.split('_')[3]
    if '.csv' in fov:
        fov = fov[:-4]
    return fov


def _parse_fc_array(fc):
    """Coerce a fluor_curve cell to np.ndarray. Handles already-array input
    and the bracketed string form '[1.0 2.0 3.0]' that pandas reads back."""
    if isinstance(fc, np.ndarray):
        return fc
    return np.array([float(x) for x in fc[1:-1].split()])


##############
##############

def plot_array_aligned(imagedict, dest, fov, keyword, subplotwidth=4, subplotheight=4,
                       hspace=0.17, wspace=0.17):
    nummips = len(imagedict.get('imagename'))
    numcol = np.ceil(np.sqrt(nummips)).astype(int)
    numrow = np.ceil(nummips / numcol).astype(int)
    if not (numcol > 1 and numrow > 1):
        print('Not enough registered bits to generate aligned image. Skipping.')
        return

    figure, axis = plt.subplots(nrows=numrow, ncols=numcol,
                                figsize=(subplotheight * numcol, subplotwidth * numrow))

    for i in range(numrow):
        for j in range(numcol):
            ind = i * numcol + j
            if ind >= nummips:
                axis[i, j].axis('off')
                continue
            name = imagedict.get('imagename')[ind]
            flag = imagedict.get('flag')[ind]
            if flag == 'OK':
                _overlay(axis[i, j],
                         imagedict.get('refplane')[ind],
                         imagedict.get('tarplane')[ind],
                         f'Prehyb_DAPI_(r)\n {name}_(g)')
            if flag == 'DND':
                axis[i, j].set_title(f'Prehyb_DAPI_(r)\n {name}_(g)\n NOT REG. DUE TO FLAG',
                                     fontsize=10)
                axis[i, j].axis('off')

    figure.subplots_adjust(hspace=hspace, wspace=wspace)
    figure.savefig(os.path.join(dest, f'Post_reg_alignment_FOV_{fov}_{keyword}.png'),
                   dpi=300, format='png', bbox_inches='tight')
    plt.close(figure)


##############
##############

def plot_array_aligned_z(zcolname1, zcolname2, zcolname3, imagedict, dest, fov, keyword,
                         subplotwidth=3.3, subplotheight=3.3, hspace=0.15, wspace=0.1):
    numrow = len(imagedict.get('imagename'))
    numcol = 4
    if numrow <= 0:
        print('Not enough registered bits to generate aligned image. Skipping.')
        return

    # squeeze=False keeps axis 2D for numrow==1, collapsing the original's two-way branching.
    figure, axis = plt.subplots(nrows=numrow, ncols=numcol,
                                figsize=(subplotheight * numcol, subplotwidth * numrow),
                                squeeze=False)

    for i in range(numrow):
        flag = imagedict.get('flag')[i]
        if flag != 'OK':
            continue
        refp = imagedict.get('refplane')[i]
        tarps = [imagedict.get(f'tarplane{k}')[i] for k in range(4)]
        alignedplanes = imagedict.get('planes')[i]
        name = imagedict.get('imagename')[i]
        znames = ['avg', zcolname1, zcolname2, zcolname3]

        for j in range(numcol):
            title = (f'Prehyb_DAPI(r)_p_{alignedplanes[0]}\n'
                     f'{znames[j]}_shift_{alignedplanes[0] - alignedplanes[j + 1]}\n'
                     f'{name}_p_{alignedplanes[j + 1]}')
            _overlay(axis[i, j], refp, tarps[j], title, title_fontsize=8)

    figure.subplots_adjust(hspace=hspace, wspace=wspace)
    figure.savefig(os.path.join(dest, f'Post_reg_alignment_z_FOV_{fov}_{keyword}.png'),
                   dpi=200, format='png', bbox_inches='tight')
    plt.close(figure)


##############
##############

def plot_array_aligned_z_mC(zcolname1, zcolname2, zcolname3, imagedict, dest, fov, keyword,
                            subplotwidth=3, subplotheight=3, hspace=0.15, wspace=0.1):
    numrow = len(imagedict.get('imagename'))
    numcol = 5
    if numrow <= 0:
        print('Not enough registered bits to generate aligned image. Skipping.')
        return

    figure, axis = plt.subplots(nrows=numrow, ncols=numcol,
                                figsize=(subplotheight * numcol, subplotwidth * numrow),
                                squeeze=False)

    for i in range(numrow):
        flag = imagedict.get('flag')[i]
        if flag != 'OK':
            continue
        refp = imagedict.get('refplane')[i]
        tarps = [imagedict.get(f'tarplane{k}')[i] for k in range(4)]
        imagelist = [refp] + tarps
        alignedplanes = imagedict.get('planes')[i]
        name = imagedict.get('imagename')[i]
        znames = ['avg', zcolname1, zcolname2, zcolname3]

        for j in range(numcol):
            im = imagelist[j]
            axis[i, j].imshow(im, vmin=np.percentile(im, 2), vmax=np.percentile(im, 99),
                              cmap='gray')
            axis[i, j].axis('off')
            if j == 0:
                axis[i, j].set_title(f'Prehyb_DAPI_plane_{alignedplanes[0]}', fontsize=8)
            else:
                axis[i, j].set_title(f'{znames[j - 1]}_shift_{alignedplanes[0] - alignedplanes[j]}\n'
                                     f'{name}_p_{alignedplanes[j]}',
                                     fontsize=8)

    figure.subplots_adjust(hspace=hspace, wspace=wspace)
    figure.savefig(os.path.join(dest, f'Post_reg_alignment_z_mC_FOV_{fov}_{keyword}.png'),
                   dpi=200, format='png', bbox_inches='tight')
    plt.close(figure)


##############
##############

def plot_array_prealigned(imagedict, dest, fov, keyword, subplotwidth=4, subplotheight=4,
                          hspace=0.17, wspace=0.17):
    nummips = len(imagedict.get('imagename'))
    numcol = np.ceil(np.sqrt(nummips)).astype(int)
    numrow = np.ceil(nummips / numcol).astype(int)
    figure, axis = plt.subplots(nrows=numrow, ncols=numcol,
                                figsize=(subplotheight * numcol, subplotwidth * numrow))

    for i in range(numrow):
        for j in range(numcol):
            ind = i * numcol + j
            if ind >= nummips:
                axis[i, j].axis('off')
                continue
            name = imagedict.get('imagename')[ind]
            _overlay(axis[i, j],
                     imagedict.get('refplane')[ind],
                     imagedict.get('tarplane')[ind],
                     f'Prehyb_DAPI_(r)\n {name}_(g)')

    figure.subplots_adjust(hspace=hspace, wspace=wspace)
    figure.savefig(os.path.join(dest, f'Pre_reg_alignment_FOV_{fov}_{keyword}.png'),
                   dpi=300, format='png', bbox_inches='tight')
    plt.close(figure)


##############
##############

def _run_pre_mip_alignment(fov, allchs, df, dest, keyword):
    """Generate pre-alignment MIP overlays. Currently never called from the
    main pipeline (the invocation site at the bottom is commented out) but
    preserved for parity with the original."""
    aligned = {'imagename': [], 'refplane': [], 'tarplane': []}

    refid = [i for i, val in enumerate(allchs.get('imagename')) if 'cycle_99_ch_2' in val][0]
    refstack = allchs['image'][refid]

    for tarname in df.get('tar'):
        tarid = [i for i, val in enumerate(allchs.get('imagename')) if tarname in val][0]
        tarstack = allchs['image'][tarid]
        refplane = np.max(refstack, axis=0)
        tarplane = np.max(tarstack, axis=0)
        if 'cycle_88_ch_2' in tarname:  # antibody VIM staining
            tarplane = _vim_clip(tarplane)
        aligned = utils._update_data(aligned, {
            'imagename': tarname, 'refplane': refplane, 'tarplane': tarplane,
        })

    plot_array_prealigned(imagedict=aligned, dest=dest, fov=fov, keyword=keyword)


##############
##############

def _run_post_alignment(fov, allchs, df, dest, keyword):
    aligned = {'imagename': [], 'refplane': [], 'tarplane': [], 'flag': []}

    refid = [i for i, val in enumerate(allchs.get('imagename')) if 'cycle_99_ch_2' in val][0]
    refstack = allchs['image'][refid]

    for tid, tarname in enumerate(df.get('tar')):
        tarid = [i for i, val in enumerate(allchs.get('imagename')) if tarname in val][0]
        tarstack = allchs['image'][tarid]
        shiftx = int(df.get('x')[tid])
        shifty = int(df.get('y')[tid])

        refplane = np.max(refstack, axis=0)
        tarplane = np.max(tarstack, axis=0)
        if shiftx != FAILED_SHIFT:
            if 'cycle_88_ch_2' in tarname:
                tarplane = _vim_clip(tarplane)
            refp, tarp, _, _ = reg.clip(refplane, tarplane, refstack, tarstack, shiftx, shifty)
            aligned = utils._update_data(aligned, {
                'imagename': tarname, 'refplane': refp, 'tarplane': tarp, 'flag': 'OK',
            })
        else:
            aligned = utils._update_data(aligned, {
                'imagename': tarname, 'refplane': refplane, 'tarplane': tarplane, 'flag': 'DND',
            })

    plot_array_aligned(imagedict=aligned, dest=dest, fov=fov, keyword=keyword)


##############
##############

def _extract_dapi_fc(qctouse):
    """Find the prehyb DAPI fluor_curve in qctouse. Returns np.ndarray."""
    for element in qctouse['imagename']:
        if 'cycle_99_ch_2' in element:
            ind = qctouse[qctouse['imagename'].str.contains('cycle_99_ch_2')].index[0]
            return _parse_fc_array(qctouse['fluor_curve'][ind])
    return None


def _clamp_z(z, length):
    """Clamp z-shift to [0, length-1] window the original uses for plane indexing."""
    if z >= length:
        return 0
    return z


def _run_post_alignment_z(fov, qctouse, allchs, df, zcolname1, zcolname2, zcolname3,
                          dest, keyword):
    aligned = {'imagename': [], 'refplane': [], 'tarplane0': [], 'tarplane1': [],
               'tarplane2': [], 'tarplane3': [], 'planes': [], 'flag': []}
    misaligned = {'imagename': [], 'refplane': [], 'tarplane0': [], 'tarplane1': [],
                  'tarplane2': [], 'tarplane3': [], 'planes': [], 'flag': []}

    refid = [i for i, val in enumerate(allchs.get('imagename')) if 'cycle_99_ch_2' in val][0]
    refstack = allchs['image'][refid]
    dapifcf = _extract_dapi_fc(qctouse)

    for tid, tarname in enumerate(df.get('tar')):
        tarid = [i for i, val in enumerate(allchs.get('imagename')) if tarname in val][0]
        tarstack = allchs['image'][tarid]
        shiftx = int(df.get('x')[tid])
        shifty = int(df.get('y')[tid])
        shiftz = _clamp_z(int(df.get('z')[tid]), dapifcf.shape[0])
        zfinal = _clamp_z(int(df.get(zcolname1)[tid]), dapifcf.shape[0])
        diffmaxz = _clamp_z(int(df.get(zcolname2)[tid]), dapifcf.shape[0])
        pccz = _clamp_z(int(df.get(zcolname3)[tid]), dapifcf.shape[0])

        # Pick refind: the dapi plane brightest within the safe window.
        minz = min(zfinal, pccz, diffmaxz)
        maxz = max(zfinal, pccz, diffmaxz)
        if maxz > 0:
            fc = dapifcf[maxz:dapifcf.shape[0]]
            refind = maxz + np.argmax(fc)
        else:
            refind = np.argmax(dapifcf[:dapifcf.shape[0] + minz])
        if refind >= dapifcf.shape[0]:
            refind = 0

        tarindvals = [refind - shiftz, refind - zfinal, refind - diffmaxz, refind - pccz]
        # Clip target indices to valid range.
        for k, item in enumerate(tarindvals):
            if item < 0:
                tarindvals[k] = 0
            if item > dapifcf.shape[0]:
                tarindvals[k] = dapifcf.shape[0] - 1

        refmip = np.max(refstack, axis=0)
        tarmip = np.max(tarstack, axis=0)

        if shiftx != FAILED_SHIFT:
            _, _, refs, tars = reg.clip(refmip, tarmip, refstack, tarstack, shiftx, shifty)
            refplane = refs[refind, :, :]
            tarplanes = [tars[idx, :, :] for idx in tarindvals]
            if 'cycle_88_ch_2' in tarname:
                tarplanes = [_vim_clip(p) for p in tarplanes]
            row = {
                'imagename': tarname, 'refplane': refplane,
                'tarplane0': tarplanes[0], 'tarplane1': tarplanes[1],
                'tarplane2': tarplanes[2], 'tarplane3': tarplanes[3],
                'planes': [refind, refind - shiftz, refind - zfinal,
                           refind - diffmaxz, refind - pccz],
                'flag': 'OK',
            }
        else:
            row = {
                'imagename': tarname, 'refplane': refmip,
                'tarplane0': tarmip, 'tarplane1': tarmip,
                'tarplane2': tarmip, 'tarplane3': tarmip,
                'planes': ['mip', 'mip', 'mip', 'mip', 'mip'],
                'flag': 'DND',
            }

        aligned = utils._update_data(aligned, row)
        if maxz - minz > 7:
            misaligned = utils._update_data(misaligned, row)

    plot_array_aligned_z_mC(zcolname1, zcolname2, zcolname3,
                            imagedict=aligned, dest=dest, fov=fov, keyword=keyword)
    plot_array_aligned_z_mC(zcolname1, zcolname2, zcolname3,
                            imagedict=misaligned, dest=dest, fov=fov, keyword='_Neg_' + keyword)


##############
##############

def convert_to_int(df, row, zcolnames):
    """Coerce three z-columns from a row to integers, tolerating str/list inputs."""
    output = [0, 0, 0]
    for idx, zcolname in enumerate(zcolnames):
        cell = df[zcolname][row]
        if isinstance(cell, str):
            output[idx] = int(cell[1:-1])
        elif isinstance(cell, list):
            output[idx] = int(cell[0])
        elif isinstance(cell, pd.Series):
            print(cell)
        else:
            output[idx] = int(cell)
    return output


def seqadd(df, currvals, tar, ref, xcolname, ycolname, zcolname1, zcolname2, zcolname3):
    """Walk one step up the reference chain, adding the tar->ref shifts to
    `currvals`. Returns (currvals, ref-of-this-row) or (currvals, '999') if
    the target isn't in df. `ref` is in the signature for backward compat
    with the original; the function does not read it.
    """
    row = df[df['tarname'] == tar].index
    if len(row) == 0:
        return currvals, '999'

    row = row[0]
    rowref = df['refname'][row]
    output = convert_to_int(df, row, [zcolname1, zcolname2, zcolname3])
    newvals = np.array((df[xcolname][row], df[ycolname][row],
                        output[0], output[1], output[2]))
    if currvals is None:
        return newvals, rowref
    return np.add(currvals, newvals), rowref


##############
##############

def update_success(kv, df, reffcf, tarfcf, zcolname1, zcolname2, zcolname3):
    """Append a successful registration row to df (alongside derived z, refind, tarind).

    Computes z as the mean of the three z-estimates (with z3 capped at 0 when
    it's out of range), then locates the brightest aligned plane in the
    fluor-curve product to set refind/tarind.
    """
    df = utils._update_data(df, kv)
    z1, z2, z3 = kv[zcolname1], kv[zcolname2], kv[zcolname3]
    if z3 > reffcf.shape[0]:
        df['z'].append(int(np.mean(np.array([z1, z2, 0]))))
    else:
        df['z'].append(int(np.mean(np.array([z1, z2, z3]))))

    zrow = df['tar'].index(kv['tar'])
    zshift = int(df['z'][zrow])

    newtarfc = np.zeros(tarfcf.shape[0])
    if zshift < 0:
        newtarfc[:zshift] = tarfcf[0:(tarfcf.shape[0] + zshift)]
    elif zshift > 0:
        newtarfc[zshift:] = tarfcf[zshift:]
    refind = np.multiply(reffcf, newtarfc).argmax()
    df['refind'].append(refind)
    df['tarind'].append(refind - zshift)
    return df


##############
##############

def _placeholder_row(element, qcflag, comment, zcolname1, zcolname2, zcolname3,
                    include_qc_flags=True):
    """Build a FAILED_SHIFT-filled row for missing/non-terminating targets.

    `include_qc_flags=False` is used for the non-terminating-sequence branch
    where qc_flags has already been appended earlier in the loop body.
    """
    row = {
        'x': FAILED_SHIFT, 'y': FAILED_SHIFT, 'z': FAILED_SHIFT,
        'ref': 'No reference', 'tar': element,
        'refind': FAILED_SHIFT, 'tarind': FAILED_SHIFT,
        zcolname1: FAILED_SHIFT, zcolname2: FAILED_SHIFT, zcolname3: FAILED_SHIFT,
        'comments': comment,
    }
    if include_qc_flags:
        row['qc_flags'] = qcflag
    return row


def _process_fov_for_finalreg(val, source, dest, QC, xcolname, ycolname,
                              zcolname1, zcolname2, zcolname3, saveword, output_keyword,
                              masterdf=None, default_comment=''):
    """Build a FINAL_Reg_fov_<fov>_<saveword>.csv from one Reg_report CSV.

    If `masterdf` is given (RegF_failed case), prepend a synthetic
    prehyb_dapi row using cross-FOV averages for x/y.
    """
    fov = _fov_from_filename(val)
    print(f'Currently reading {val}')

    qc_csv = [v for v in QC if '_' + fov + '.csv' in v][0]
    fovdata = pd.read_csv(os.path.join(source, val))
    qcdata = pd.read_csv(os.path.join(source, qc_csv))

    fovtouse = fovdata[[xcolname, ycolname, 'refname', 'tarname',
                        zcolname1, zcolname2, zcolname3]]
    qctouse = qcdata[['imagename', 'fluor_curve', 'flags']]

    if masterdf is not None:
        # Insert a dummy prehyb_dapi row using cross-FOV averages.
        firstref = fovtouse['refname'][0]
        chunks = firstref.split('_')
        firstref_short = '_'.join(chunks[3:])

        xpopavg = masterdf.filter(regex='x').loc[firstref_short].to_numpy()
        xavg = np.mean(xpopavg[xpopavg != FAILED_SHIFT])
        ypopavg = masterdf.filter(regex='y').loc[firstref_short].to_numpy()
        yavg = np.mean(ypopavg[ypopavg != FAILED_SHIFT])

        new_row = pd.DataFrame.from_dict({
            xcolname: [int(xavg)], ycolname: [int(yavg)],
            'refname': ['pre' + '_'.join(chunks[:3]) + '_cycle_99_ch_2'],
            'tarname': [fovtouse['refname'][0]],
            zcolname1: [0], zcolname2: [0], zcolname3: [0],
        })
        fovtouse = pd.concat([new_row, fovtouse], ignore_index=True)

    df = {'x': [], 'y': [], 'z': [], 'ref': [], 'tar': [], 'refind': [], 'tarind': [],
          zcolname1: [], zcolname2: [], zcolname3: [], 'qc_flags': [], 'comments': []}

    for element in qctouse['imagename']:
        if 'cycle_99_ch_2' in element:
            continue
        tarexists = fovtouse[fovtouse['tarname'].str.contains(element)].index
        qcrow = qctouse[qctouse['imagename'] == element].index[0]

        if len(tarexists) == 0:
            df = utils._update_data(df, _placeholder_row(
                element, qctouse['flags'][qcrow], default_comment,
                zcolname1, zcolname2, zcolname3))
            continue

        tarrow = qctouse[qctouse['imagename'] == element].index[0]
        refrow = qctouse[qctouse['imagename'].str.contains('cycle_99_ch_2')].index[0]
        tarfcf = _parse_fc_array(qctouse['fluor_curve'][tarrow])
        reffcf = _parse_fc_array(qctouse['fluor_curve'][refrow])
        df['qc_flags'].append(qctouse['flags'][tarrow])

        row = fovtouse[fovtouse['tarname'] == element].index[0]
        rowref = fovtouse['refname'][row]
        if masterdf is not None:
            print(f'Now examining: element:{element}, row:{row}, rowref:{rowref}')

        currvals, rowref = seqadd(fovtouse, None, element, rowref,
                                  xcolname, ycolname, zcolname1, zcolname2, zcolname3)

        if 'cycle_99_ch_2' in rowref:
            kv = {'x': currvals[0], 'y': currvals[1], 'ref': rowref, 'tar': element,
                  zcolname1: currvals[2], zcolname2: currvals[3], zcolname3: currvals[4],
                  'comments': default_comment}
            df = update_success(kv, df, reffcf, tarfcf, zcolname1, zcolname2, zcolname3)
        else:
            while 'cycle_99_ch_2' not in rowref and rowref != '999':
                currvals, rowref = seqadd(fovtouse, currvals, rowref, 'cycle_99_ch_2',
                                          xcolname, ycolname, zcolname1, zcolname2, zcolname3)
            if rowref != '999':
                kv = {'x': currvals[0], 'y': currvals[1], 'ref': rowref, 'tar': element,
                      zcolname1: currvals[2], zcolname2: currvals[3], zcolname3: currvals[4],
                      'comments': default_comment}
                df = update_success(kv, df, reffcf, tarfcf, zcolname1, zcolname2, zcolname3)
            else:
                print(f'Non-terminating sequence detected for {rowref}. Filling with 999s instead.')
                row_dict = _placeholder_row(element, None, default_comment,
                                            zcolname1, zcolname2, zcolname3,
                                            include_qc_flags=False)
                df = utils._update_data(df, row_dict)

    dfpd = pd.DataFrame(df)
    dfpd.to_csv(os.path.join(dest, f'FINAL_Reg_fov_{fov}_{saveword}{output_keyword}.csv'),
                float_format='%.4f')
    if masterdf is None:
        print(f'{source}, {fov}')
    return fov


##############
##############

def run_qc_registration_post_processing(args):
    if not (os.path.exists(args.source) and os.path.exists(args.source_im)):
        return
    print('Source and source-image directories found.')
    if len(args.dest) == 0:
        args.dest = args.source

    os.chdir(args.source)

    xcolname = args.xcoltouse
    ycolname = args.ycoltouse
    zcolname1 = args.zcoltouse1
    zcolname2 = args.zcoltouse2
    zcolname3 = args.zcoltouse3
    saveword = args.saveword

    Failed = glob.glob('FAILED' + '*.csv')  # noqa: F841 (read for parity; not used downstream)
    Attempts = glob.glob('Reg_attempts' + '*.csv')  # noqa: F841 (read for parity; not used downstream)
    QC = glob.glob('QC_Rpt' + '*.csv')
    Reg = glob.glob('Reg_report' + '*.csv')

    print(f'Looking for reports with the word {saveword}, length {len(saveword)} in {args.source}.')
    difftries = []
    for val in Reg:
        splits = val.split('_')
        if len(saveword) < 2:
            saveword = splits[-1][:-4]
            if saveword not in difftries:
                difftries.append(saveword)
        else:
            if saveword not in difftries:
                difftries.append(saveword)
    print(f'Processing all entries with the words {difftries} in {args.source}.')

    for saveword in difftries:
        RegF = sorted(v for v in Reg if saveword in v)

        # Step 1: classify each Reg_report.
        # - 'NO TISSUE': skip entirely.
        # - reference list missing prehyb_dapi: RegF_failed (needs synthetic row later).
        # - has prehyb_dapi but fewer rows than max: RegF_passed (gaps filled with 999).
        # - has prehyb_dapi and full row count: RegF_passed.
        num_rows = [pd.read_csv(os.path.join(args.source, v)).shape[0] for v in RegF]
        numrows = int(max(num_rows))

        RegF_passed = []
        RegF_failed = []
        for val in RegF:
            fovdata = pd.read_csv(os.path.join(args.source, val))
            strfind = fovdata['refname'].str.find('cycle_99_ch_2')
            if 'NO TISSUE' in fovdata['refname'].iloc[0]:
                print(f'Omitting {val} from the list of fovs due to NO TISSUE flag.')
            elif max(strfind) < 0:
                RegF_failed.append(val)
                print(f'Adding {val} to list of fovs with prehyb_DAPI not in reference list.')
            elif fovdata.shape[0] < numrows and max(strfind) > 0:
                RegF_passed.append(val)
                print(f'Note: {val} has not enough rows, will correct later.')
            else:
                RegF_passed.append(val)
        print(f'RegF_failed:{RegF_failed}.')

        # Step 2: process the "passed" pile (no synthetic row).
        for val in RegF_passed:
            _process_fov_for_finalreg(
                val, args.source, args.dest, QC, xcolname, ycolname,
                zcolname1, zcolname2, zcolname3, saveword, args.keyword,
                masterdf=None, default_comment='')

        # Step 3: process the "failed" pile (insert synthetic prehyb_dapi row).
        masterdf, _ = generate_masterdf(args.dest, args.keyword)
        print(masterdf)
        for val in RegF_failed:
            _process_fov_for_finalreg(
                val, args.source, args.dest, QC, xcolname, ycolname,
                zcolname1, zcolname2, zcolname3, saveword, args.keyword,
                masterdf=masterdf, default_comment='Z is unverified')

        # Step 4: re-generate masterdf and apply outlier correction in-place.
        masterdf, finals = generate_masterdf(args.dest, args.keyword)
        print(masterdf)
        for filename in finals:
            df = pd.read_csv(os.path.join(args.dest, filename), index_col=0)
            xdata = df['x'].copy()
            ydata = df['y'].copy()
            zdata = df['z'].copy()
            commentsdata = df['comments'].copy()

            rownames = masterdf.index.values.tolist()
            cycles = list(set('cycle_' + r.split('_')[1] + '_' for r in rownames))

            # Set z=999 entries to z=0 and add a comment marker.
            zdata, commentsdata = correctz(zdata, commentsdata, rownames, 'z')

            print(f'Now correcting possible x-/y- outliers in {filename}.')
            for ind, rowname in enumerate(rownames):
                xpopavg = masterdf.filter(regex='x').iloc[ind]
                xdata, commentsdata = correctdata(xdata, ind, rowname, xpopavg, commentsdata, 'x')
                ypopavg = masterdf.filter(regex='y').iloc[ind]
                ydata, commentsdata = correctdata(ydata, ind, rowname, ypopavg, commentsdata, 'y')

            for cycle in cycles:
                indices = masterdf.index.str.contains(cycle)
                xpopavg = masterdf.filter(regex='x')[indices]
                xdata, commentsdata = correctdatabych(xdata, indices, xpopavg, commentsdata)
                ypopavg = masterdf.filter(regex='y')[indices]
                ydata, commentsdata = correctdatabych(ydata, indices, ypopavg, commentsdata)

            df['x'] = xdata.tolist()
            df['y'] = ydata.tolist()
            df['z'] = zdata.tolist()
            df['comments'] = commentsdata.tolist()
            df.to_csv(os.path.join(args.dest, filename), float_format='%.4f')
            print(f'Corrections in {filename} complete.')

        # Step 5: optional alignment plots.
        if args.plot_alignment == 'True':
            for filename in finals:
                if saveword not in filename:
                    continue
                print(f'Now doing {filename} that contains {saveword}.')
                fov = _fov_from_filename(os.path.basename(filename))
                df = pd.read_csv(os.path.join(args.dest, filename))

                st = time.time()
                allchs = utils.parse_directory(args.source_im, fov)
                allchs = utils._read_ims(allchs)
                _run_post_alignment(fov, allchs, df, args.dest, saveword)

                if args.plot_alignment_z == 'True':
                    qc_csv = [v for v in QC if '_' + fov + '.csv' in v][0]
                    qcdata = pd.read_csv(os.path.join(args.source, qc_csv))
                    qctouse = qcdata[['imagename', 'fluor_curve', 'flags']]
                    _run_post_alignment_z(fov, qctouse, allchs, df,
                                          zcolname1, zcolname2, zcolname3,
                                          args.dest, saveword)
                print(f'time taken: {time.time() - st} seconds')


##############
##############

def generate_masterdf(folder, keyword):
    finals = sorted(glob.glob(os.path.join(folder, 'FINAL_Reg_fov_' + '*' + keyword + '.csv')))

    masterdict = {}
    df = None
    for filename in finals:
        fov = os.path.basename(filename).split('_')[3]
        df = pd.read_csv(filename, index_col=0)
        masterdict[f'x_FOV_{fov}'] = df['x'].tolist()
        masterdict[f'y_FOV_{fov}'] = df['y'].tolist()
        masterdict[f'z_FOV_{fov}'] = df['z'].tolist()

    masterdf = pd.DataFrame.from_dict(masterdict).reindex(sorted(masterdict), axis=1)
    newnames = {i: '_'.join(name.split('_')[3:]) for i, name in enumerate(df['tar'])}
    masterdf = masterdf.rename(index=newnames)

    return masterdf, finals


##############
##############

def correctdata(df, ind, rowname, popavg, comments, axis=''):
    popavg = popavg.to_numpy()
    avg = np.mean(popavg[popavg != FAILED_SHIFT])
    std = np.std(popavg[popavg != FAILED_SHIFT])
    if df.iloc[ind] >= avg + 2 * std + 5 or df.iloc[ind] <= avg - 2 * std - 5:
        if pd.isna(comments.iloc[ind]):
            comments.iloc[ind] = 'AutoCorr: orig_' + axis + '_' + str(round(df.iloc[ind]))
        else:
            comments.iloc[ind] = (str(comments[ind]) + ' / orig_' + axis + '_'
                                  + str(round(df.iloc[ind])))
        df.iloc[ind] = round(avg)
    return df, comments


def correctz(df, comments, rownames, axis=''):
    for ind, rowname in enumerate(rownames):
        if df.iloc[ind] == FAILED_SHIFT:
            if pd.isna(comments.iloc[ind]):
                comments.iloc[ind] = 'Z-unverified; AutoCorr: orig_' + axis + '_' + str(FAILED_SHIFT)
            else:
                comments.iloc[ind] = (str(comments[ind]) + ' / Z-unverified, orig_'
                                      + axis + '_' + str(FAILED_SHIFT))
            df.iloc[ind] = 0
    return df, comments


##############
##############

def correctdatabych(df, indices, popavg, comments):
    subset = df.loc[indices]
    if (max(subset) - min(subset)) < 4:
        return df, comments

    if len(subset.tolist()) == 4:
        # Antibody channels: pick the value with smallest sum-of-squared-pairwise-differences,
        # then drag the outliers towards the kept subset.
        diffs = np.zeros(len(subset.tolist()))
        for i, val in enumerate(subset.index.values.tolist()):
            diffs[i] = np.sum((df.loc[val] - subset) ** 2)
        minarg = np.argmin(diffs)
        subsetfailed = (abs(subset.to_numpy() - subset.to_numpy()[minarg]) >= 2)
        subsetpassed = subset.to_numpy()[np.invert(subsetfailed)]
        for i, val in enumerate(subset.index.values.tolist()):
            if subsetfailed[i]:
                if df.loc[val] < max(subsetpassed):
                    df.loc[val] = max(subsetpassed) - 1
                else:
                    df.loc[val] = min(subsetpassed) + 1
    else:
        # Two values: drag the outlier towards the one closer to popavg.
        avg = np.round(popavg.mean())
        diffs = np.zeros(len(subset.tolist()))
        for i, val in enumerate(subset.index.values.tolist()):
            diffs[i] = np.sum((df.loc[val] - avg) ** 2)
        minarg = np.argmin(diffs)
        maxarg = np.argmax(diffs)
        names = subset.index.values.tolist()
        if df.loc[names[maxarg]] > df.loc[names[minarg]]:
            df.loc[names[maxarg]] = df.loc[names[minarg]] + 1
        else:
            df.loc[names[maxarg]] = df.loc[names[minarg]] - 1

    return df, comments


##############
##############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='')
    parser.add_argument('--source_im', default='')
    parser.add_argument('--dest', default='')
    parser.add_argument('--xcoltouse', default='shiftx_final')
    parser.add_argument('--ycoltouse', default='shifty_final')
    parser.add_argument('--zcoltouse1', default='shiftz_final')
    parser.add_argument('--zcoltouse2', default='diff_max_z')
    parser.add_argument('--zcoltouse3', default='pcc_z')
    parser.add_argument('--plot_alignment', default='True')
    parser.add_argument('--plot_alignment_z', default='True')
    parser.add_argument('--saveword', default='')
    parser.add_argument('--keyword', default='')
    args = parser.parse_args()

    run_qc_registration_post_processing(args)
