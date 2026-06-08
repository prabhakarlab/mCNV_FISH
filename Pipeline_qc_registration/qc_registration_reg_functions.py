"""
Registration primitives: XY (StackReg + phase cross correlation) and Z
(custom line-projection PCC, with an optional mask variant for VIM/antibody
staining).
"""
import statistics
import time
from collections import Counter

import cv2
import numpy as np
from pystackreg import StackReg
from scipy import stats
from skimage.registration import phase_cross_correlation

import qc_registration_utils as utils


# Sentinel for "registration could not be determined". Used pervasively for
# downstream comparisons (`if shiftx == FAILED_SHIFT: ...`).
FAILED_SHIFT = 999


##############
##############

def init_reg_results(pcc_ds, pcc):
    regdata = {
        'sr_x': [], 'sr_y': [], 'sr_x_y_time': [],
        'sr_mf_x': [], 'sr_mf_y': [], 'sr_mf_x_y_time': [],
        'pcc_mip_x': [], 'pcc_mip_y': [], 'pcc_mip_time': [],
        'pcc_mf_mip_x': [], 'pcc_mf_mip_y': [], 'pcc_mf_mip_time': [],
        'conf_x': [], 'conf_y': [],
    }
    if pcc_ds == 'True':
        regdata.update({'pcc_ds_fz_x': [], 'pcc_ds_fz_y': [], 'pcc_ds_fz_z': [], 'pcc_ds_fz_time': []})
    if pcc == 'True':
        regdata.update({'pcc_x': [], 'pcc_y': [], 'pcc_z': [], 'pcc_time': []})

    regdata.update({
        'diff_max_z': [], 'z_est': [], 'z_est_vals': [], 'z_est_time': [], 'z_final_time': [],
        'shiftx_final': [], 'shifty_final': [], 'shiftz_final': [],
        'refname': [], 'tarname': [], 'params': [],
    })
    return regdata

##############
##############

def downsize(allchs, ds_xy, ds_z, ds_add):
    """Build downsized image stacks (image_ds, image_ds_mips, image_ds_fz)."""
    images = allchs.get('image')
    allchs['image_ds'] = []
    allchs['image_ds_mips'] = []
    allchs['image_ds_fz'] = []
    ds_xy = int(ds_xy)
    ds_z = int(ds_z)
    ds_add = int(ds_add)

    for image in images:
        zshape = int(np.floor(image.shape[0] / ds_z))
        h = int(np.floor(image.shape[1] / ds_xy))
        w = int(np.floor(image.shape[2] / ds_xy))

        imageds = np.zeros((zshape, h, w))
        for i in range(zshape):
            if ds_add > 0:
                # Must be uint16 for cv2.resize to accept it.
                planes = np.sum(image[(ds_z * i):(ds_z * i + ds_add + 1), :, :], axis=0, dtype='uint16')
            else:
                planes = image[ds_z * i, :, :]
            imageds[i, :, :] = cv2.resize(planes, (w, h), interpolation=cv2.INTER_CUBIC)

        imagedsfz = np.zeros((image.shape[0], h, w))
        for i in range(imagedsfz.shape[0]):
            imagedsfz[i, :, :] = cv2.resize(image[i, :, :], (w, h), interpolation=cv2.INTER_CUBIC)

        utils._update_data(allchs, {'image_ds': imageds,
                                    'image_ds_mips': np.max(imageds, axis=0),
                                    'image_ds_fz': imagedsfz})

    return allchs

##############
##############

def extract_bundle(imagename, allchs, qcparams):
    imageid = [i for i, val in enumerate(allchs.get('imagename')) if imagename in val]

    if isinstance(imageid, str):
        imageid = int(imageid)
    if isinstance(imageid, list):
        imageid = np.array(imageid)[0]

    return {
        'imagename': imagename, 'imageid': imageid,
        'mip':        allchs.get('image_ds_mips')[imageid],
        'stackds':    allchs.get('image_ds')[imageid],
        'stackdsfz':  allchs.get('image_ds_fz')[imageid],
        'stackfs':    allchs.get('image')[imageid],
        'gm':         qcparams.get('global_mode')[imageid],
        'sd':         qcparams.get('global_stdev')[imageid],
        'fc':         qcparams.get('fluor_curve')[imageid],
        'flags':      qcparams.get('flags')[imageid],
    }

##############
##############

def normalize_image(image, low, high):
    tempimage = (image - low) / (high - low)
    tempimage[tempimage < 0] = 0
    tempimage[tempimage > 1] = 1
    return tempimage

##############
##############

def get_fluor_curve(stack):
    fc = np.zeros(min(stack.shape))
    for plane in range(len(fc)):
        fc[plane] = np.sum(stack[plane, :, :])
    fc = (fc - np.min(fc)) / (np.max(fc) - np.min(fc))
    return fc

##############
##############

def find_roi(refmcn, tarmcn, numtiles=9, divisor=2, printoutput=False):
    """Find a subregion of the MIPs with the greatest pixel-intensity variance."""
    rsubsetvar = np.zeros(numtiles)
    tsubsetvar = np.zeros(numtiles)

    length = int(np.sqrt(numtiles))
    lengthr = int(np.round(refmcn.shape[0] / divisor))
    lengthc = int(np.round(refmcn.shape[1] / divisor))

    jumplengthr = int(np.floor((refmcn.shape[0] - lengthr) / (length - 1)))
    jumplengthc = int(np.floor((refmcn.shape[1] - lengthc) / (length - 1)))

    for i in range(length):
        for j in range(length):
            rsubset = refmcn[(i * jumplengthr):(i * jumplengthr + lengthr),
                             (j * jumplengthc):(j * jumplengthc + lengthc)]
            tsubset = tarmcn[(i * jumplengthr):(i * jumplengthr + lengthr),
                             (j * jumplengthc):(j * jumplengthc + lengthc)]
            rsubsetvar[i * length + j] = np.var(rsubset)
            tsubsetvar[i * length + j] = np.var(tsubset)

    totalvar = np.add(rsubsetvar / np.sum(rsubsetvar), tsubsetvar / np.sum(tsubsetvar))
    maxvar = totalvar.argmax()
    subseti = int(np.floor(maxvar / length))
    subsetj = maxvar % length

    startr = subseti * jumplengthr
    endr = subseti * jumplengthr + lengthr
    startc = subsetj * jumplengthc
    endc = subsetj * jumplengthc + lengthc

    if printoutput:
        print(f'params: rsubsetvar:{rsubsetvar}, tsubsetvar:{tsubsetvar}, length:{length}, '
              f'lengthr:{lengthr}, lengthc:{lengthc}, jumplengthr:{jumplengthr}, jumplengthc:{jumplengthc}, '
              f'totalvar:{totalvar}, maxvar:{maxvar}, subseti:{subseti}, subsetj:{subsetj}')
    # print(f'Looking at region: {startr}:{endr}, {startc}:{endc}.')

    return startr, endr, startc, endc

##############
##############

def register_xy_pccmip(refM, tarM):
    st = time.time()
    pccmipshift, _, _ = phase_cross_correlation(refM, tarM, normalization=None)
    et = time.time()
    return pccmipshift[1], pccmipshift[0], et - st

##############
##############

def register_xy_sr(refname, tarname, refM, tarM):
    """StackReg translation registration with raw + percentile-thresholded inputs.

    We threshold at six different percentiles to build six binarised copies,
    register each, take the values within 1 stdev of the median, and average.
    """
    st = time.time()

    refM = np.clip(refM, a_min=np.min(refM.flatten()), a_max=np.percentile(refM, 99.9))
    tarM = np.clip(tarM, a_min=np.min(tarM.flatten()), a_max=np.percentile(tarM, 99.8))

    names = [refname, tarname]
    images = [refM.astype(int), tarM.astype(int)]
    compname = f'{refname}_vs_{tarname}'

    results = {'compname': [], 'prop_proc_im1': [], 'prop_proc_im2': [],
               'test': [], 'res_sr_x': [], 'res_sr_y': []}

    im1, im2 = images[0], images[1]
    sr = StackReg(StackReg.TRANSLATION)
    sr.register_transform(im1, im2)
    utils._update_data(results, {
        'compname': compname,
        'prop_proc_im1': 0,
        'prop_proc_im2': 0,
        'test': 'raw',
        'res_sr_x': np.round(-sr.get_matrix()[0, 2]).astype('int'),
        'res_sr_y': np.round(-sr.get_matrix()[1, 2]).astype('int'),
    })

    for percentile in np.arange(60, 90, 6):
        pim1 = (im1 > np.percentile(im1, percentile)).astype(int)
        pim2 = (im2 > np.percentile(im2, percentile)).astype(int)

        srp = StackReg(StackReg.TRANSLATION)
        srp.register_transform(pim1, pim2)
        utils._update_data(results, {
            'compname': compname,
            'prop_proc_im1': np.sum(pim1) / (pim1.shape[0] * pim1.shape[0]),
            'prop_proc_im2': np.sum(pim1) / (pim1.shape[0] * pim1.shape[0]),
            'test': f'{percentile}th_percentile',
            'res_sr_x': np.round(-srp.get_matrix()[0, 2]).astype('int'),
            'res_sr_y': np.round(-srp.get_matrix()[1, 2]).astype('int'),
        })

    stdevx = stats.tstd(np.array(results.get('res_sr_x')))
    stdevy = stats.tstd(np.array(results.get('res_sr_y')))
    medx = statistics.median(results.get('res_sr_x'))
    medy = statistics.median(results.get('res_sr_y'))

    shiftx = [v for v in results.get('res_sr_x') if (medx - stdevx) <= v <= (medx + stdevx)]
    shifty = [v for v in results.get('res_sr_y') if (medy - stdevy) <= v <= (medy + stdevy)]

    shiftx = int(np.round(np.mean(np.array(shiftx))))
    shifty = int(np.round(np.mean(np.array(shifty))))

    return shiftx, shifty, time.time() - st

##############
##############

def clip(refmip, tarmip, ref, tar, shiftx, shifty):
    """Slice both the MIP and stack arrays by `shiftx`/`shifty` to bring overlap regions into alignment."""
    if shiftx < 0:
        tarmip = tarmip[:, -shiftx:]
        refmip = refmip[:, :shiftx]
        tar = tar[:, :, -shiftx:]
        ref = ref[:, :, :shiftx]
    elif shiftx > 0:
        tarmip = tarmip[:, :-shiftx]
        refmip = refmip[:, shiftx:]
        tar = tar[:, :, :-shiftx]
        ref = ref[:, :, shiftx:]

    if shifty < 0:
        tarmip = tarmip[-shifty:, :]
        refmip = refmip[:shifty, :]
        tar = tar[:, -shifty:, :]
        ref = ref[:, :shifty, :]
    elif shifty > 0:
        tarmip = tarmip[:-shifty, :]
        refmip = refmip[shifty:, :]
        tar = tar[:, :-shifty, :]
        ref = ref[:, shifty:, :]

    return refmip, tarmip, ref, tar

##############
##############

def generate_mask(refbundle, tarbundle, qcparams, reference, maskpercentile=75):
    """Build masked stacks (pixels below the mask threshold are floored to the
    global min), pick the high-variance ROI on the unmasked normalised MIPs,
    and crop. Returns (refm2, tarm2, ref2, tar2, startr, endr, startc, endc).
    """
    refm = refbundle.get('mip')
    tarm = tarbundle.get('mip')
    ref = refbundle.get('stackds')
    tar = tarbundle.get('stackds')

    mask = np.array(refm > np.percentile(refm, maskpercentile)).astype(int)
    print(f'Masking: {mask.shape}')
    globalrefmin = np.min(ref)
    globaltarmin = np.min(tar)

    # Vectorised replacement for the per-pixel for-loop. Broadcast the 2D mask
    # over Z; pixels outside the mask are floored to the global min.
    mask_bool = mask.astype(bool)[np.newaxis, :, :]
    refmasked = np.where(mask_bool, ref, globalrefmin)
    tarmasked = np.where(mask_bool, tar, globaltarmin)

    refmaskedm = np.max(refmasked, axis=0)
    tarmaskedm = np.max(tarmasked, axis=0)

    propdapiim = qcparams.get('prop_im_covered')[reference]
    print(f'prop_dapi_im is: {propdapiim}')

    if propdapiim > 0.1:
        refmc = np.clip(refm, a_min=np.min(refm.flatten()), a_max=np.percentile(refm, 99.8))
        tarmc = np.clip(tarm, a_min=np.min(tarm.flatten()), a_max=np.percentile(tarm, 99.8))
        refmcn = normalize_image(refmc, np.min(refmc.flatten()), np.max(refmc.flatten()))
        tarmcn = normalize_image(tarmc, np.min(tarmc.flatten()), np.max(tarmc.flatten()))
        startr, endr, startc, endc = find_roi(refmcn, tarmcn, numtiles=9, divisor=2)
    else:
        print('too little tissue to determine variance of subregions. Now using entire image.')
        startr, endr = 0, refm.shape[0]
        startc, endc = 0, refm.shape[1]

    refm2 = refmaskedm[startr:endr, startc:endc]
    tarm2 = tarmaskedm[startr:endr, startc:endc]
    ref2 = refmasked[:, startr:endr, startc:endc]
    tar2 = tarmasked[:, startr:endr, startc:endc]

    return refm2, tarm2, ref2, tar2, startr, endr, startc, endc

##############
##############

def _select_z_indices(fc, flags, fccutoff, every_other=False):
    """Indices in `fc` above the chosen threshold (0.5 for clean ZFCs, else
    fccutoff for linear ones), optionally only even indices.
    """
    threshold = (0.5 if 'ZFC_is_linear' not in flags else fccutoff) * max(fc)
    if every_other:
        return [i for i, val in enumerate(fc) if val > threshold and i % 2 == 0]
    return [i for i, val in enumerate(fc) if val > threshold]


def _resolve_shiftz(masterz_dict, shiftzpcc, diff_max_z, dsz, refbundle=None, tarbundle=None):
    """Pull the consensus shiftz from a masterz dict. Falls back to PCC and
    diff_max_z when the masterz search produced nothing.

    Second return is `uniq` from getmode (or None when fallback was used).
    """
    if len(masterz_dict.get('shiftzest')) == 0:
        if shiftzpcc is not None:
            return int(np.mean([diff_max_z, shiftzpcc])), None, []
        return int(diff_max_z), None, []

    mostcommon = Counter(masterz_dict.get('shiftzest')).most_common()
    # print(f'Most common estimated values from Counter: {mostcommon}')
    shiftz, uniq = getmode(mostcommon, dsz)
    return shiftz, uniq, masterz_dict.get('shiftzest')


def _run_z_registration(allchs, qcparams, reference, target, fccutoff, linespread,
                        lowerthresh, upperthresh, zsearchwindow, dsz,
                        shiftx_final, shifty_final, zclipnorm,
                        shiftzpcc=None, use_mask=False):
    """Z-registration. With `use_mask=True`, the ref-mask is applied to both
    stacks (used for antibody/VIM channels where the signal is sparse).
    """
    refbundle = extract_bundle(allchs.get('imagename')[reference], allchs, qcparams)
    tarbundle = extract_bundle(allchs.get('imagename')[target], allchs, qcparams)

    reffcraw = get_fluor_curve(refbundle.get('stackfs'))
    tarfcraw = get_fluor_curve(tarbundle.get('stackfs'))
    diff_max_z = int(np.where(reffcraw == np.max(reffcraw))[0] - np.where(tarfcraw == np.max(tarfcraw))[0])

    if use_mask:
        print('STARTING Z_REGISTRATION_W_MASKING:', allchs.get('imagename')[reference],
              allchs.get('imagename')[target], ' diff_max_z', str(diff_max_z),
              ' FLAGS: ', refbundle.get('flags'), tarbundle.get('flags'))
        refm2, tarm2, ref2, tar2, startr, endr, startc, endc = generate_mask(
            refbundle, tarbundle, qcparams, reference, maskpercentile=75)
    else:
        propdapiim = qcparams.get('prop_im_covered')[reference]
        print('STARTING Z_REGISTRATION: ref:', allchs.get('imagename')[reference],
              ' tar:', allchs.get('imagename')[target], ' diff_max_z:', str(diff_max_z),
              ' prop_dapi_im:', str(propdapiim), ' FLAGS: ',
              refbundle.get('flags'), tarbundle.get('flags'))
        refm = refbundle.get('mip')
        tarm = tarbundle.get('mip')

        if propdapiim > 0.05:
            refmc = np.clip(refm, a_min=np.min(refm.flatten()), a_max=np.percentile(refm, 99.8))
            tarmc = np.clip(tarm, a_min=np.min(tarm.flatten()), a_max=np.percentile(tarm, 99.8))
            refmcn = normalize_image(refmc, np.min(refmc.flatten()), np.max(refmc.flatten()))
            tarmcn = normalize_image(tarmc, np.min(tarmc.flatten()), np.max(tarmc.flatten()))
            startr, endr, startc, endc = find_roi(refmcn, tarmcn, numtiles=9, divisor=2)
        else:
            print('too little tissue to determine variance of subregions. Now using entire image.')
            startr, endr = 0, refm.shape[0]
            startc, endc = 0, refm.shape[1]

        refm2 = refm[startr:endr, startc:endc]
        tarm2 = tarm[startr:endr, startc:endc]
        ref2 = refbundle.get('stackds')[:, startr:endr, startc:endc]
        tar2 = tarbundle.get('stackds')[:, startr:endr, startc:endc]

    _, _, ref3, tar3 = clip(refm2, tarm2, ref2, tar2, shiftx_final, shifty_final)

    # First pass: z-estimate on the (possibly masked) downsized stack.
    st = time.time()
    reffc = get_fluor_curve(ref3)
    tarfc = get_fluor_curve(tar3)

    np.set_printoptions(precision=4)
    refind = _select_z_indices(reffc, refbundle.get('flags'), fccutoff)
    tarind = _select_z_indices(tarfc, tarbundle.get('flags'), fccutoff)

    windowlength = np.min([
        (np.sum(reffc > lowerthresh * np.max(reffc)) - np.sum(reffc > upperthresh * np.max(reffc))),
        (np.sum(tarfc > lowerthresh * np.max(tarfc)) - np.sum(tarfc > upperthresh * np.max(tarfc))),
    ])
    windowlength = max(3, min(8, windowlength))

    if not use_mask and (len(refind) <= 5 or len(tarind) <= 5):
        print(f'Printing reffc and tarfc to check because refind and/or tarind is small: {reffc}, {tarfc}.')

    label = 'masked reference' if use_mask else 'ref'
    label2 = 'masked target' if use_mask else 'tar'
    print(f'Searching {label} z in:{refind}, searching {label2} z in:{tarind}, '
          f'windowlength_range: {np.arange(windowlength - 1, windowlength + 2, 1)}')

    mindim = min(ref3.shape[1], ref3.shape[2])
    colvec = np.arange(10, mindim, linespread)
    rowvec = np.arange(10, mindim, linespread)

    masterzest = {'i': [], 'j': [], 'shiftzest': []}
    masterzest = get_z_est(colvec, rowvec, ref3, tar3, refind, tarind, masterzest,
                           windowlength, zsearchwindow, shiftzest=0, cutoff=99,
                           zclipnorm=zclipnorm)
    z_est_time = time.time() - st

    shiftz_est, uniq, shiftz_est_vals = _resolve_shiftz(masterzest, shiftzpcc, diff_max_z, dsz)

    # only retest the algo's stability when masterzest produced
    # values (otherwise `uniq` is None and we keep the fallback).
    if uniq is not None:
        if ('ZFC_is_linear' not in refbundle.get('flags')
                and 'ZFC_is_linear' not in tarbundle.get('flags')
                and max(uniq) < 3):
            # The algo disagreed with itself on clean-ZFC inputs - don't trust it.
            print('Detecting unstable results from z-registration algo when fluor_curves are good. '
                  'Defaulting to diff_max_z and shiftzpcc values.')
            if shiftzpcc is not None:
                shiftz_est = int(np.mean([diff_max_z, shiftzpcc]))
            else:
                shiftz_est = int(diff_max_z)

    print('Z_REG EST:', str(shiftz_est), 'planes from:', masterzest.get('shiftzest'),
          '_time:', str(int(z_est_time)), ' seconds')

    # Second pass (full-z): refine the estimate on the unmasked full-resolution stack.
    if dsz <= 1:
        return diff_max_z, shiftz_est, shiftz_est_vals, z_est_time, shiftz_est, z_est_time

    reffz = refbundle.get('stackdsfz')
    tarfz = tarbundle.get('stackdsfz')
    reffz2 = reffz[:, startr:endr, startc:endc]
    tarfz2 = tarfz[:, startr:endr, startc:endc]
    _, _, reffz3, tarfz3 = clip(refm2, tarm2, reffz2, tarfz2, shiftx_final, shifty_final)

    st = time.time()
    reffc = get_fluor_curve(reffz3)
    tarfc = get_fluor_curve(tarfz3)

    refind = _select_z_indices(reffc, refbundle.get('flags'), fccutoff, every_other=True)
    tarind = _select_z_indices(tarfc, tarbundle.get('flags'), fccutoff, every_other=True)
    print(f'Searching ref z in:{refind}, searching tar z in:{tarind}')

    mindim = min(reffz3.shape[1], reffz3.shape[2])
    colvec = np.arange(10, mindim, linespread)
    rowvec = np.arange(10, mindim, linespread)

    masterz = {'i': [], 'j': [], 'shiftzest': []}
    masterz = get_z_est(colvec, rowvec, reffz3, tarfz3, refind, tarind, masterz,
                        windowlength=windowlength * dsz, zsearchwindow=zsearchwindow,
                        shiftzest=shiftz_est, cutoff=5, zclipnorm=zclipnorm)
    zfinaltime = time.time() - st

    if len(masterz.get('shiftzest')) > 0:
        mostcommon = Counter(masterz.get('shiftzest')).most_common()
        # print(f'Most common final values from Counter: {mostcommon}')
        shiftz, _ = getmode(mostcommon, 1)
    else:
        if shiftzpcc is not None:
            shiftz = int(np.mean([diff_max_z, shiftzpcc]))
        else:
            shiftz = int(diff_max_z)

    print('ENDING Z_REGISTRATION: final:', str(shiftz), 'planes',
          '_ref:', allchs.get('imagename')[reference],
          '_tar:', allchs.get('imagename')[target],
          '_time:', str(int(zfinaltime)), ' seconds')

    return diff_max_z, shiftz_est, shiftz_est_vals, z_est_time, shiftz, zfinaltime

##############
##############

def _run_z_registration_mask(allchs, qcparams, reference, target, fccutoff, linespread,
                             lowerthresh, upperthresh, zsearchwindow, dsz,
                             shiftx_final, shifty_final, zclipnorm, shiftzpcc=None):
    """Backwards-compatible thin wrapper: forwards to _run_z_registration with use_mask=True."""
    return _run_z_registration(allchs, qcparams, reference, target, fccutoff, linespread,
                               lowerthresh, upperthresh, zsearchwindow, dsz,
                               shiftx_final, shifty_final, zclipnorm,
                               shiftzpcc=shiftzpcc, use_mask=True)

##############
##############

def get_z_est(colvec, rowvec, ref3, tar3, refind, tarind, masterzest,
              windowlength=8, zsearchwindow=0, shiftzest=0, cutoff=99, zclipnorm='False'):
    mindim = min(ref3.shape[1], ref3.shape[2])
    st = time.time()

    for window in np.arange(windowlength - zsearchwindow, windowlength + zsearchwindow + 1, 1):
        for i in refind:
            _mins = {'i': [], 'j': [], 'shift': [], 'error': []}
            for j in tarind:
                if not (i + window < ref3.shape[0] and j + window < tar3.shape[0]):
                    continue
                if shiftzest != 0 and ((i - j) < (shiftzest - cutoff) or (i - j) > (shiftzest + cutoff)):
                    continue

                refmc = ref3[i:(window + i), 0:mindim, colvec]
                tarmc = tar3[j:(window + j), 0:mindim, colvec]
                refmr = ref3[i:(window + i), rowvec, 0:mindim]
                tarmr = tar3[j:(window + j), rowvec, 0:mindim]

                combref = np.zeros((2 * refmc.shape[0] * refmc.shape[2], refmc.shape[1]))
                combtar = np.zeros((2 * refmc.shape[0] * refmc.shape[2], refmc.shape[1]))
                for k in range(refmc.shape[2]):
                    combref[(2 * k * refmc.shape[0]):((2 * k + 1) * refmc.shape[0]), :] = refmc[:, :, k]
                    combref[((2 * k + 1) * refmc.shape[0]):((2 * k + 2) * refmc.shape[0]), :] = refmr[:, k, :]
                    combtar[(2 * k * refmc.shape[0]):((2 * k + 1) * refmc.shape[0]), :] = tarmc[:, :, k]
                    combtar[((2 * k + 1) * refmc.shape[0]):((2 * k + 2) * refmc.shape[0]), :] = tarmr[:, k, :]

                if zclipnorm == 'True':
                    combrefc = np.clip(combref, np.percentile(combref, 1), np.percentile(combref, 98))
                    combrefcn = normalize_image(combrefc, np.min(combrefc), np.max(combrefc))
                    combtarc = np.clip(combtar, np.percentile(combtar, 1), np.percentile(combtar, 98))
                    combtarcn = normalize_image(combtarc, np.min(combtarc), np.max(combtarc))
                    shift, error, _ = phase_cross_correlation(combrefcn, combtarcn, normalization=None)
                else:
                    shift, error, _ = phase_cross_correlation(combref, combtar, normalization=None)

                _mins['i'].append(i)
                _mins['j'].append(j)
                _mins['shift'].append(shift)
                _mins['error'].append(error)

            if len(_mins.get('error')) > 0:
                minind = np.array(_mins.get('error')).argmin()
                masterzest['i'].append(_mins.get('i')[minind])
                masterzest['j'].append(_mins.get('j')[minind])
                masterzest['shiftzest'].append(_mins.get('i')[minind] - _mins.get('j')[minind])

    print(f'time: {time.time() - st:.3f} seconds')
    return masterzest

##############
##############

def getmode(mostcommon, dsz):
    occurrences = [val[1] for val in mostcommon]
    uniq = np.unique(occurrences)[::-1]

    modelist = np.zeros(len(uniq))
    for i in range(len(uniq)):
        listtoavg = [val[0] for val in mostcommon if val[1] == uniq[i]]
        modelist[i] = np.mean(listtoavg)

    # print(f'uniq: {uniq}, modelist: {modelist}')
    if len(modelist) == 1:
        return modelist[0] * dsz, uniq
    return int(np.round((modelist[0] * uniq[0] + modelist[1] * uniq[1]) / (uniq[0] + uniq[1]) * dsz)), uniq
