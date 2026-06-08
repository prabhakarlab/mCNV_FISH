"""
Main pipeline orchestrator. Performs QC then per-FOV registration:
1. parse_directory + _read_ims → catalog of all stacks
2. MIP plot
3. _run_qc → fluor curves, background mode/stdev, image-coverage proportion,
   flags (out-of-focus, linear ZFC, no tissue)
4. _run_registration: pick a reference (default prehyb DAPI), register
   each target in xy then z. On failure, pick a new reference and retry
   up to 10 attempts.
5. Force-register any remaining unregistered bits using same-cycle peers.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from skimage.registration import phase_cross_correlation

import qc_registration_qc_functions as qc
import qc_registration_reg_functions as reg
import qc_registration_postprocess_functions as post  # noqa: F401 (used by commented-out postproc call below)
import qc_registration_utils as utils
from qc_registration_reg_functions import FAILED_SHIFT


#################
#################

def _run_qc_mip(fov, allchs, dest):
    print(f'Generating MIP plot.')
    imagenames = allchs.get('imagename')
    mips = {'imagename': [], 'imagemip': []}

    for imageid, imagename in enumerate(imagenames):
        image = allchs['image'][imageid]
        imagemip = np.max(image, axis=0)
        mips['imagename'].append(imagename)
        mips['imagemip'].append(imagemip)
    qc.plot_array_mip(imagedict=mips, dest=dest, fov=fov)

    return mips

#################
#################

def _run_qc(fov, allchs, dest, dapicutoff, fccutoff, swidth):
    qcparams = qc.init_qc_params()
    imagenames = allchs.get('imagename')

    for imageid, imagename in enumerate(imagenames):
        st1 = time.time()
        print(f'{imagename} / {imageid}')
        image = allchs['image'][imageid]
        imageflat = image.flatten()
        imagebincts = np.bincount(imageflat.astype(int))

        # Smoothed global bincounts across the whole stack.
        smoothing = np.zeros((imagebincts.shape[0], 9))
        searchlist = [1, 3, 5, 10, 15, 20, 30, 40, 50]
        for i in searchlist:
            smoothing[:, searchlist.index(i)] = qc.smooth(imagebincts, i)
        bincttouse = smoothing[:, 5]  # USE A SMOOTH=20 by default
        globalmode = qc.find_local_max(bincttouse, windowsize=10, minthreshold=1000)
        globalmode = int(globalmode[0])  # First local max defines the background.
        globalstdev = qc.stdev_half_gaussian(bincttouse, globalmode)

        # Per-plane fluorescence.
        collatedarraysums = reg.get_fluor_curve(image)
        imagemip = np.max(image, axis=0)
        decayL, decayR = qc.fluorescence_decay(collatedarraysums, collatedarraysums.argmax())
        prop = np.sum(imagemip > globalmode + 4 * globalstdev) / imagemip.shape[0] / imagemip.shape[1]
        signalwidth = np.sum(collatedarraysums > fccutoff * np.max(collatedarraysums))

        qcparams = utils._update_data(qcparams, {
            'imagename':         imagename,
            'global_mode':       globalmode,
            'global_stdev':      globalstdev,
            'fluor_curve':       collatedarraysums,
            'max_plane':         collatedarraysums.argmax(),
            'decay_from_max_L':  decayL,
            'decay_from_max_R':  decayR,
            'prop_im_covered':   prop,
        })

        flag = []
        if min(decayL, decayR) <= 0.05:
            flag.append('ZFC_is_linear')
        if signalwidth < swidth * len(collatedarraysums):
            flag.append('Out_of_focus')
        qcparams['flags'].append(flag if len(flag) > 0 else [])

        print(f'QC: {time.time() - st1:.3f} seconds')

    fluorcurves = qcparams.get('fluor_curve')
    fluorcurvearr = np.zeros((len(fluorcurves[0]), len(fluorcurves)))
    for i in range(len(fluorcurves)):
        fluorcurvearr[:, i] = np.array(fluorcurves[i])

    metaname = '_Acr_Bits_' + '_fov_' + str(fov) + '.png'
    qc.plot_array_hist(fluorcurvearr, filename=metaname, dest=dest,
                       yscaletype='linear', titlewords=imagenames)

    propims = qcparams.get('prop_im_covered')
    propdapiind = [i for i, val in enumerate(qcparams.get('imagename')) if '99_ch_2' in val][0]
    propdapi = propims[propdapiind]

    if propdapi < dapicutoff:
        print('No tissue found')
        for i in range(len(imagenames)):
            qcparams['flags'][i].append('No_tissue')

    df = pd.DataFrame(qcparams)
    df.to_csv(os.path.join(dest, 'QC_Rpt_fov_' + str(fov) + '.csv'), float_format='%.4f')

    return qcparams

#################
#################

def get_next_ref(reflist, targetlistids, targetlistnames, mips, allchs, issuccess):
    """Pick the next reference image by median MIP fluorescence."""
    medianfluors = [np.percentile(mips.get('imagemip')[i], 50) for i in targetlistids]
    medianindex = np.argsort(medianfluors)[len(medianfluors) // 2]

    if issuccess == 'True':
        medianindexname = allchs.get('imagename')[targetlistids[medianindex]]
    else:
        # When `targetlistnames` is hybnames rather than allchs imagenames.
        medianindexname = targetlistnames[medianindex]

    print(f'new reference image for next round: {medianindex}; {medianindexname}')
    newbittoreg = int(medianindexname.split('_')[4])
    newchtoreg = int(medianindexname.split('_')[6])
    newreference = [i for i, val in enumerate(allchs.get('imagename'))
                    if ('_cycle_' + str(newbittoreg) + '_ch_' + str(newchtoreg) in val)][0]
    print(f'newbittoreg: {newbittoreg}, newchtoreg: {newchtoreg}, newref: {newreference}')

    return newbittoreg, newchtoreg, newreference

#################
#################

# speed-up — do sr_x and pcc_mip_x first; only run the median-
# filtered (_mf) versions if they're within 20% of each other.
def _run_registration_sr_pcc(refname, tarname, refm, tarm, refm2, tarm2,
                             regdata, failedregdata, regattempts):
    sr_x, sr_y, sr_x_y_time = reg.register_xy_sr(refname, tarname, refm, tarm)
    pcc_mip_x, pcc_mip_y, pcc_mip_time = reg.register_xy_pccmip(refm, tarm)
    print(f'Initial xy regs: sr_x:{sr_x}, sr_y:{sr_y}; pcc_mip_x:{pcc_mip_x}, pcc_mip_y:{pcc_mip_y}.')

    initial_disagrees = (
        (abs(sr_x - pcc_mip_x) > 10 and abs(sr_x - pcc_mip_x) / max(abs(sr_x), abs(pcc_mip_x)) > 0.2)
        or
        (abs(sr_y - pcc_mip_y) > 10 and abs(sr_y - pcc_mip_y) / max(abs(sr_y), abs(pcc_mip_y)) > 0.2)
    )

    if initial_disagrees:
        print(f'Initial xy failed to reach consensus, skipping downstream steps.')
        shiftx_final = FAILED_SHIFT
        shifty_final = FAILED_SHIFT

        failedregdata = utils._update_data(failedregdata, {
            'sr_x': sr_x, 'sr_y': sr_y, 'sr_x_y_time': sr_x_y_time,
            'sr_mf_x': FAILED_SHIFT, 'sr_mf_y': FAILED_SHIFT, 'sr_mf_x_y_time': FAILED_SHIFT,
            'pcc_mip_x': pcc_mip_x, 'pcc_mip_y': pcc_mip_y, 'pcc_mip_time': pcc_mip_time,
            'pcc_mf_mip_x': FAILED_SHIFT, 'pcc_mf_mip_y': FAILED_SHIFT, 'pcc_mf_mip_time': FAILED_SHIFT,
            'refname': refname, 'tarname': tarname,
            'conf_x': 0, 'conf_y': 0,
            'shiftx_final': 'INCONCLUSIVE', 'shifty_final': 'INCONCLUSIVE',
        })
        regattempts = utils._update_data(regattempts,
            {'ref': refname, 'tar': tarname, 'status': 'Failed'})
        return shiftx_final, shifty_final, regdata, failedregdata, regattempts

    sr_mf_x, sr_mf_y, sr_mf_x_y_time = reg.register_xy_sr(refname, tarname, refm2, tarm2)
    pcc_mf_mip_x, pcc_mf_mip_y, pcc_mf_mip_time = reg.register_xy_pccmip(refm2, tarm2)

    xvec = np.array([sr_x, sr_mf_x, pcc_mip_x, pcc_mf_mip_x])
    yvec = np.array([sr_y, sr_mf_y, pcc_mip_y, pcc_mf_mip_y])

    mean_x_est = np.median(xvec)
    sd_x_est = max(3, 0.1 * np.std(xvec))
    close_x = np.array([abs(xvec - mean_x_est) <= sd_x_est])
    # print(f'sr_x, sr_mf_x, pcc_mip_x, pcc_mf_mip_x: {xvec}, mean:{mean_x_est}, std:{sd_x_est}, conf:{close_x}')
    conf_x = np.sum(np.array(close_x).astype(int))

    mean_y_est = np.median(yvec)
    sd_y_est = max(3, 0.1 * np.std(yvec))
    close_y = np.array([abs(yvec - mean_y_est) <= sd_y_est])
    # print(f'sr_y, sr_mf_y, pcc_mip_y, pcc_mf_mip_y: {yvec}, mean:{mean_y_est}, std:{sd_y_est}, conf:{close_y}')
    conf_y = np.sum(np.array(close_y).astype(int))

    if conf_x >= 3 and conf_y >= 3:
        shiftx_final = int(np.round(np.sum(xvec * close_x) / np.sum(close_x)))
        shifty_final = int(np.round(np.sum(yvec * close_y) / np.sum(close_y)))

        regdata = utils._update_data(regdata, {
            'sr_x': sr_x, 'sr_y': sr_y, 'sr_x_y_time': sr_x_y_time,
            'sr_mf_x': sr_mf_x, 'sr_mf_y': sr_mf_y, 'sr_mf_x_y_time': sr_mf_x_y_time,
            'pcc_mip_x': pcc_mip_x, 'pcc_mip_y': pcc_mip_y, 'pcc_mip_time': pcc_mip_time,
            'pcc_mf_mip_x': pcc_mf_mip_x, 'pcc_mf_mip_y': pcc_mf_mip_y, 'pcc_mf_mip_time': pcc_mf_mip_time,
            'refname': refname, 'tarname': tarname,
            'conf_x': conf_x, 'conf_y': conf_y,
            'shiftx_final': shiftx_final, 'shifty_final': shifty_final,
        })
        regattempts = utils._update_data(regattempts,
            {'ref': refname, 'tar': tarname, 'status': 'Successful'})
    else:
        shiftx_final = FAILED_SHIFT
        shifty_final = FAILED_SHIFT

        failedregdata = utils._update_data(failedregdata, {
            'sr_x': sr_x, 'sr_y': sr_y, 'sr_x_y_time': sr_x_y_time,
            'sr_mf_x': sr_mf_x, 'sr_mf_y': sr_mf_y, 'sr_mf_x_y_time': sr_mf_x_y_time,
            'pcc_mip_x': pcc_mip_x, 'pcc_mip_y': pcc_mip_y, 'pcc_mip_time': pcc_mip_time,
            'pcc_mf_mip_x': pcc_mf_mip_x, 'pcc_mf_mip_y': pcc_mf_mip_y, 'pcc_mf_mip_time': pcc_mf_mip_time,
            'refname': refname, 'tarname': tarname,
            'conf_x': conf_x, 'conf_y': conf_y,
            'shiftx_final': 'INCONCLUSIVE', 'shifty_final': 'INCONCLUSIVE',
        })
        regattempts = utils._update_data(regattempts,
            {'ref': refname, 'tar': tarname, 'status': 'Failed'})

    return shiftx_final, shifty_final, regdata, failedregdata, regattempts

#################
#################

def _run_registration(fov, allchs, qcparams, mips, dest, regdata, failedregdata, regattempts,
                     reflist, success, targets, bittoreg, chtoreg, saveword,
                     dsxy, dsz, dsadd, fccutoff, swidth, linespread, lowerthresh, upperthresh,
                     pccds, pcc, zsearchwindow, zclipnorm):
    # Define the reference (always the latest entry in reflist).
    if isinstance(reflist, list):
        reference = int(reflist[len(reflist) - 1])
        if len(reflist) > 1:
            print(f'Multiple references detected. Using the latest reference id: {reference}.')
        else:
            print(f'Reference id: {reference}.')

    # Initialise data structures on first call.
    if regdata is None:
        regdata = reg.init_reg_results(pccds, pcc)
    if failedregdata is None:
        failedregdata = reg.init_reg_results(pccds, pcc)
    if regattempts is None:
        regattempts = {'ref': [], 'tar': [], 'status': []}

    print(f'Executing downsizing of all image stacks for subsequent steps.')
    allchs = reg.downsize(allchs, ds_xy=dsxy, ds_z=dsz, ds_add=dsadd)
    params = ('DS_xy_z_add_' + str(dsxy) + '_' + str(dsz) + '_' + str(dsadd)
              + '__FC_SW_LS_LT_UT_' + str(fccutoff) + '_' + str(swidth) + '_'
              + str(linespread) + '_' + str(lowerthresh) + '_' + str(upperthresh)
              + '_zclipnorm_' + str(zclipnorm)[0])

    for target in targets:
        print('STARTING REGISTRATION. Saving to:', dest, '\nRegistering: ',
              allchs.get('imagename')[target], 'to', allchs.get('imagename')[reference])
        starttime = time.time()

        refbundle = reg.extract_bundle(allchs.get('imagename')[reference], allchs, qcparams)
        tarbundle = reg.extract_bundle(allchs.get('imagename')[target], allchs, qcparams)

        reffcraw = reg.get_fluor_curve(refbundle.get('stackfs'))
        tarfcraw = reg.get_fluor_curve(tarbundle.get('stackfs'))
        diff_max_z = (np.where(reffcraw == np.max(reffcraw))[0]
                      - np.where(tarfcraw == np.max(tarfcraw))[0])
        print(f'Diff in max z-planes: {diff_max_z} from '
              f'{np.where(reffcraw == np.max(reffcraw))[0]} and '
              f'{np.where(tarfcraw == np.max(tarfcraw))[0]}')

        reffz = refbundle.get('stackdsfz')
        tarfz = tarbundle.get('stackdsfz')

        # XY registration - run on the full 2048x2048 MIPs (and median-filtered copies).
        refmfs = np.max(refbundle.get('stackfs'), axis=0)
        tarmfs = np.max(tarbundle.get('stackfs'), axis=0)
        refmfs_mf = median_filter(refmfs, size=[3, 3])
        tarmfs_mf = median_filter(tarmfs, size=[3, 3])

        st = time.time()
        shiftx_final, shifty_final, regdata, failedregdata, regattempts = _run_registration_sr_pcc(
            allchs.get('imagename')[reference], allchs.get('imagename')[target],
            refmfs, tarmfs, refmfs_mf, tarmfs_mf, regdata, failedregdata, regattempts)
        print(f'XY-registration: {time.time() - st:.3f} seconds')

        # Optional 3D-PCC passes (downsized and full-res).
        shiftzpcctosave = None
        xy_failed = (shiftx_final == FAILED_SHIFT and shifty_final == FAILED_SHIFT)

        if pccds == 'True':
            print(f'Now doing 3D PCC on the downsized stack.')
            st = time.time()
            shift_ds_fz, _, _ = phase_cross_correlation(reffz, tarfz, normalization=None)
            elapsed = time.time() - st
            print(f'PCC_DS: {elapsed:.3f} seconds')
            kv = {'pcc_ds_fz_x': 2 * shift_ds_fz[2], 'pcc_ds_fz_y': 2 * shift_ds_fz[1],
                  'pcc_ds_fz_z': shift_ds_fz[0], 'pcc_ds_fz_time': elapsed}
            target_dict = failedregdata if xy_failed else regdata
            utils._update_data(target_dict, kv)
            shiftzpcctosave = shift_ds_fz[0]

        if pcc == 'True':
            print(f'Now doing 3D PCC on the full-resolution stack. Selecting the region with greatest variance.')
            st = time.time()
            # run PCC only on the most-interesting 3/4 * 3/4 region of the MIP.
            startr, endr, startc, endc = reg.find_roi(refmfs, tarmfs, numtiles=9, divisor=4 / 3)
            reftemp = refbundle.get('stackfs')[:, startr:endr, startc:endc]
            tartemp = tarbundle.get('stackfs')[:, startr:endr, startc:endc]
            shift_fz, _, _ = phase_cross_correlation(reftemp, tartemp, normalization=None)
            elapsed = time.time() - st
            print(f'PCC: {elapsed:.3f} seconds')
            kv = {'pcc_x': shift_fz[2], 'pcc_y': shift_fz[1], 'pcc_z': shift_fz[0], 'pcc_time': elapsed}
            target_dict = failedregdata if xy_failed else regdata
            utils._update_data(target_dict, kv)
            shiftzpcctosave = shift_fz[0]

        # Z-registration only runs if xy succeeded.
        if xy_failed:
            failedregdata = utils._update_data(failedregdata, {
                'params': params, 'diff_max_z': diff_max_z,
                'z_est': 'NOT DONE', 'z_est_vals': 'NOT DONE',
                'z_est_time': 'NOT DONE', 'z_final_time': 'NOT DONE',
                'shiftz_final': 'INCONCLUSIVE',
            })
        else:
            diff_max_z, shiftz_est, shiftz_est_vals, z_est_time, shiftz, zfinaltime = reg._run_z_registration(
                allchs, qcparams, reference, target, fccutoff, linespread, lowerthresh, upperthresh,
                zsearchwindow, dsz, shiftx_final, shifty_final, zclipnorm, shiftzpcc=shiftzpcctosave)
            regdata = utils._update_data(regdata, {
                'params': params, 'diff_max_z': diff_max_z,
                'z_est': shiftz_est, 'z_est_vals': shiftz_est_vals,
                'z_est_time': z_est_time, 'z_final_time': zfinaltime,
                'shiftz_final': shiftz,
            })
            success.append(target)

        print(f'TOTAL Registration time: {time.time() - starttime:.3f} seconds')
        # print(f'reference_id: {reference}, success_ids: {success}, targets_ids: {targets}\n')

        # Save partial results after each target so the run is resumable.
        pd.DataFrame(regdata).to_csv(
            os.path.join(dest, 'Reg_report_fov_' + fov + '_' + saveword + '.csv'),
            float_format='%.4f')
        pd.DataFrame(failedregdata).to_csv(
            os.path.join(dest, 'FAILED_Reg_report_fov_' + fov + '_' + saveword + '.csv'),
            float_format='%.4f')
        pd.DataFrame(regattempts).to_csv(
            os.path.join(dest, 'Reg_attempts_fov_' + fov + '_' + saveword + '.csv'),
            float_format='%.4f')

    targets = [val for val in targets if val not in success]
    # print(f'New targets after round of registration: {targets}')
    return regdata, failedregdata, regattempts, reflist, success, targets

#################
#################

def _force_register_remaining(allchs, qcparams, targets, reflist, success, regdata,
                              fov, dest, args):
    """For any target that never got registered, find a previously-registered
    channel from the same cycle and use its xy shift (assumed 0) plus a
    fresh z-registration.
    """
    if len(targets) == 0:
        return

    for target in targets:
        targetbit = int(allchs.get('imagename')[target].split('_')[4])
        targetch = int(allchs.get('imagename')[target].split('_')[6])

        print(f'*****\nClean-up: Forcing registration for {target}; bit:{targetbit}; ch:{targetch}.\n *****\n')
        regsuccess = reflist + success
        regsuccessnames = [val for i, val in enumerate(allchs.get('imagename')) if i in regsuccess]
        print(f'Searching all successfully registered and prehyb_dapi: {regsuccess}.')

        same_cycle_prefix = '_cycle_' + str(targetbit)
        same_cycle_self = '_cycle_' + str(targetbit) + '_ch_' + str(targetch)
        samecycles = [val for val in regsuccessnames
                      if (same_cycle_prefix in val
                          and '_cycle_99_ch_2' not in val
                          and same_cycle_self not in val)]
        print(f'Found chs from the same bit that have been registered: {samecycles}.')
        if len(samecycles) == 0:
            continue

        # For antibody channels, prefer the second match (closest in MIP).
        samecycle = samecycles[0] if len(samecycles) == 1 else samecycles[1]
        print(f'From the same bit that has been registered, choosing: {samecycle}.')

        newref = samecycle
        newrefid = [i for i, val in enumerate(allchs.get('imagename')) if newref in val][0]
        print(f'From the bit that was registered, this is the reference: {newref}, {newrefid}.')

        shiftx = 0
        shifty = 0
        print(f'From the bit that was registered, these are the shifts: {shiftx}, {shifty}.')

        # Mask z-registration when targetbit==88 OR targetch==2 (preserves original
        # `if (targetbit!=88 and targetch!=2): regular else: mask` — by De Morgan, the
        # mask branch fires whenever either condition holds, not only for VIM bit88/ch2).
        use_mask = (targetbit == 88 and targetch == 2) # not (targetbit != 88 and targetch != 2)
        z_fn = reg._run_z_registration_mask if use_mask else reg._run_z_registration
        zclip = 'True' if use_mask else args.z_clip_norm

        diff_max_z, shiftz_est, shiftz_est_vals, z_est_time, shiftz_final, shiftz_time = z_fn(
            allchs, qcparams, newrefid, target,
            fccutoff=float(args.fc_cutoff), linespread=int(args.line_spread),
            lowerthresh=float(args.lower_thresh), upperthresh=float(args.upper_thresh),
            zsearchwindow=int(args.z_search_window), dsz=int(args.downsize_z),
            shiftx_final=int(shiftx), shifty_final=int(shifty),
            zclipnorm=zclip, shiftzpcc=None)

        print(f'Shiftz values are: diffmaxz:{diff_max_z}, est:{shiftz_est}, '
              f'est_vals:{shiftz_est_vals}, final:{shiftz_final}. '
              f'Time taken:{z_est_time + shiftz_time:.3f} seconds.')

        regdata = utils._update_data(regdata, {
            'sr_x': FAILED_SHIFT, 'sr_y': FAILED_SHIFT, 'sr_x_y_time': FAILED_SHIFT,
            'sr_mf_x': FAILED_SHIFT, 'sr_mf_y': FAILED_SHIFT, 'sr_mf_x_y_time': FAILED_SHIFT,
            'pcc_mip_x': FAILED_SHIFT, 'pcc_mip_y': FAILED_SHIFT, 'pcc_mip_time': FAILED_SHIFT,
            'pcc_mf_mip_x': FAILED_SHIFT, 'pcc_mf_mip_y': FAILED_SHIFT, 'pcc_mf_mip_time': FAILED_SHIFT,
            'conf_x': FAILED_SHIFT, 'conf_y': FAILED_SHIFT,
            'diff_max_z': diff_max_z, 'z_est': shiftz_est, 'z_est_vals': shiftz_est_vals,
            'z_est_time': z_est_time, 'z_final_time': shiftz_time,
            'shiftx_final': shiftx, 'shifty_final': shifty, 'shiftz_final': shiftz_final,
            'refname': newref, 'tarname': allchs.get('imagename')[target],
            'params': regdata.get('params')[0],
        })

        if args.pcc == 'True':
            regdata = utils._update_data(regdata, {
                'pcc_x': FAILED_SHIFT, 'pcc_y': FAILED_SHIFT,
                'pcc_z': FAILED_SHIFT, 'pcc_time': FAILED_SHIFT,
            })
        if args.pcc_ds == 'True':
            regdata = utils._update_data(regdata, {
                'pcc_ds_fz_x': FAILED_SHIFT, 'pcc_ds_fz_y': FAILED_SHIFT,
                'pcc_ds_fz_z': FAILED_SHIFT, 'pcc_ds_fz_time': FAILED_SHIFT,
            })

        pd.DataFrame(regdata).to_csv(
            os.path.join(dest, 'Reg_report_fov_' + fov + '_' + args.saveword + '.csv'),
            float_format='%.4f')

#################
#################

def _registration_kwargs(args):
    """Common kwargs threaded through _run_registration."""
    return dict(
        dsxy=int(args.downsize_xy), dsz=int(args.downsize_z), dsadd=int(args.downsize_add),
        fccutoff=float(args.fc_cutoff), swidth=float(args.s_width),
        linespread=int(args.line_spread),
        lowerthresh=float(args.lower_thresh), upperthresh=float(args.upper_thresh),
        pccds=args.pcc_ds, pcc=args.pcc,
        zsearchwindow=int(args.z_search_window), zclipnorm=args.z_clip_norm,
    )


def run_qc_registration_pipeline(args):
    if not (os.path.exists(args.source) and os.path.exists(args.dest)):
        return
    print(f'Source and destination directories found.')

    fovs = args.fovs.split(',')
    print(fovs)
    if len(fovs) == 0:
        return

    reg_kwargs = _registration_kwargs(args)

    for fov in fovs:
        if args.create_separate == 'True':
            sub = os.path.join(args.dest, 'FOV' + fov)
            if not os.path.exists(sub):
                os.makedirs(sub)
            dest = sub
        else:
            dest = args.dest

        # Get channel info, generate MIPs.
        allchs = utils.parse_directory(args.source, fov)
        allchs = utils._read_ims(allchs)
        mips = _run_qc_mip(fov, allchs, dest)

        # QC: either skip and load existing CSV, or run fresh.
        if args.skipqc == 'True':
            paramsfile = os.path.join(dest, 'QC_Rpt_fov_' + str(fov) + '.csv')
            print(paramsfile)
            if os.path.isfile(paramsfile):
                qcparams = pd.read_csv(paramsfile)
            else:
                print(f'Skipqc is True, but qcparams not found. Will now generate qcparams.')
                qcparams = _run_qc(fov, allchs, dest, float(args.dapicutoff),
                                   float(args.fc_cutoff), float(args.s_width))
        else:
            qcparams = _run_qc(fov, allchs, dest, float(args.dapicutoff),
                               float(args.fc_cutoff), float(args.s_width))

        # Skip FOVs with no tissue.
        no_tissue = [i for i, val in enumerate(qcparams.get('flags')) if 'No_tissue' in str(val)]
        if len(no_tissue) > 0:
            print(f'Not enough tissue detected in {fov}. Skipping registration.')
            regdata = reg.init_reg_results(args.pcc_ds, args.pcc)
            for key in regdata:
                regdata[key].append('NO TISSUE')
            pd.DataFrame(regdata).to_csv(
                os.path.join(dest, 'Reg_report_fov_' + fov + '_' + args.saveword + '.csv'))
            continue

        if args.skipreg != 'False':
            continue

        # regdata holds only successes; regattempts logs every attempt.
        regdata = None
        failedregdata = None
        regattempts = None

        # Identify reference and targets. First pass uses prehyb DAPI (bit 99 ch 2).
        bit_ch = '_cycle_' + str(args.bittoreg) + '_ch_' + str(args.chtoreg)
        reflist = [i for i, val in enumerate(allchs.get('imagename')) if bit_ch in val]
        success = []
        alltargets = [i for i, val in enumerate(allchs.get('imagename')) if bit_ch not in val]
        allhybtargets = [i for i, val in enumerate(allchs.get('imagename'))
                         if ('_cycle_99_' not in val and '_cycle_88_' not in val)]
        allhybnames = [val for i, val in enumerate(allchs.get('imagename')) if i in allhybtargets]  # noqa: F841

        # Filter out anything out-of-focus.
        non_oof_targets = [i for i, val in enumerate(qcparams.get('flags'))
                           if 'Out_of_focus' not in str(val)]
        targets = [v for v in alltargets if v in non_oof_targets]
        hybtargets = [v for v in allhybtargets if v in non_oof_targets]
        hybnames = [val for i, val in enumerate(allchs.get('imagename')) if i in hybtargets]
        print(f'ref:{reflist}, suc:{success}, tar:{targets}, hybtar:{hybtargets}, hybnames:{hybnames}')

        if len(hybtargets) <= 20:
            regdata = reg.init_reg_results(args.pcc_ds, args.pcc)
            for key in regdata:
                regdata[key].append('NOT ENOUGH USABLE HYBS')
            pd.DataFrame(regdata).to_csv(
                os.path.join(dest, 'Reg_report_fov_' + fov + '_' + args.saveword + '.csv'))
            continue

        # First pass with prehyb_dapi as reference.
        if len(success) == 0 and failedregdata is None:
            regdata, failedregdata, regattempts, reflist, success, targets = _run_registration(
                fov, allchs, qcparams, mips, dest, regdata, failedregdata, regattempts,
                reflist, success, targets,
                bittoreg=int(args.bittoreg), chtoreg=int(args.chtoreg),
                saveword=args.saveword, **reg_kwargs)
            print(f'post1strd: ref:{reflist}, success:{success}, targets:{targets}')

        # Retry rounds: pick a new reference, register the remaining targets. Up to 10 attempts.
        counter = 0
        while counter < 10 and len(targets) > 0:
            if len(success) == 0 and failedregdata is not None:
                # Nothing registered to prehyb_dapi - promote it to a target and pick a new ref.
                targets.append(reflist[0])
                reflist.remove(reflist[0])
                newbittoreg, newchtoreg, newreference = get_next_ref(
                    reflist, hybtargets, hybnames, mips, allchs, issuccess='False')
                reflist.append(newreference)
                targets.remove(newreference)
                print(f'ref:{reflist}, suc:{success}, tar:{targets}')
                regdata, failedregdata, regattempts, reflist, success, targets = _run_registration(
                    fov, allchs, qcparams, mips, dest, regdata, failedregdata, regattempts,
                    reflist, success, targets,
                    bittoreg=newbittoreg, chtoreg=newchtoreg,
                    saveword=args.saveword, **reg_kwargs)

            elif len(success) > 0 and failedregdata is not None:
                print(f'*****\nTrying new registration with alternate hyb: attempt number {counter}\n*****\n')
                newbittoreg, newchtoreg, newreference = get_next_ref(
                    reflist, success, allchs.get('imagename'), mips, allchs, issuccess='True')
                reflist.append(newreference)
                success.remove(newreference)
                print(f'ref:{reflist}, suc:{success}, tar:{targets}')
                regdata, failedregdata, regattempts, reflist, success, targets = _run_registration(
                    fov, allchs, qcparams, mips, dest, regdata, failedregdata, regattempts,
                    reflist, success, targets,
                    bittoreg=newbittoreg, chtoreg=newchtoreg,
                    saveword=args.saveword, **reg_kwargs)
                counter += 1

        # Clean-up: force-register anything left over.
        _force_register_remaining(allchs, qcparams, targets, reflist, success, regdata,
                                  fov, dest, args)


#################
#################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='')
    parser.add_argument('--dest', default='')
    parser.add_argument('--create_separate', default='False')

    parser.add_argument('--fovs', default={})
    parser.add_argument('--runqc', default='True')
    parser.add_argument('--skipqc', default='False')
    parser.add_argument('--skipreg', default='False')
    parser.add_argument('--savereg', default='True')

    parser.add_argument('--bittoreg', default=99)
    parser.add_argument('--chtoreg', default=2)
    parser.add_argument('--dapicutoff', default=0.05)
    parser.add_argument('--downsize_xy', default=2)
    parser.add_argument('--downsize_z', default=3)
    parser.add_argument('--downsize_add', default=1)
    parser.add_argument('--fc_cutoff', default=0.33)
    # proportion of z-planes that must exceed fc_cutoff for the bit to not be flagged out-of-focus
    parser.add_argument('--s_width', default=0.2)
    parser.add_argument('--line_spread', default=10)
    parser.add_argument('--lower_thresh', default=0.4)
    parser.add_argument('--upper_thresh', default=0.9)

    parser.add_argument('--pcc_ds', default='False')
    parser.add_argument('--pcc', default='True')
    parser.add_argument('--z_search_window', default=0)
    parser.add_argument('--z_clip_norm', default='True')

    parser.add_argument('--postproc_z_colname1', default='shiftz_final')
    parser.add_argument('--postproc_z_colname2', default='diff_max_z')
    parser.add_argument('--postproc_z_colname3', default='pcc_z')
    parser.add_argument('--saveword', default='')
    args = parser.parse_args()

    run_qc_registration_pipeline(args)

#################
#################
