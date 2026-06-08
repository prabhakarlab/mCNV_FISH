"""
QC helpers: array-of-MIP plotting, fluorescence histograms, local-max finding,
half-gaussian background fits, and decay statistics.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture


##############
##############

def plot_array_mip(imagedict, dest, fov, subplotwidth=4, subplotheight=4, hspace=0.17, wspace=0.17):
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
            mip = imagedict.get('imagemip')[idx]
            im = axis[i, j].imshow(mip, cmap=plt.cm.gray,
                                   vmin=np.percentile(mip, 5),
                                   vmax=np.percentile(mip, 99))
            axis[i, j].set_title(f'{imagedict.get("imagename")[idx]}', fontsize=12)
            axis[i, j].axis('off')
            figure.colorbar(mappable=im, ax=axis[i, j], pad=0.02, fraction=0.05, aspect=18)

    figure.subplots_adjust(hspace=hspace, wspace=wspace)
    figure.savefig(os.path.join(dest, 'MIPs_FOV_' + fov + '.png'),
                   dpi=300, format='png', bbox_inches='tight')
    plt.close(figure)

##############
##############

def plot_array_hist(array, filename, dest, imagetype='', subplotwidth=3, subplotheight=3,
                    hspace=0.4, wspace=0.4, yscaletype='log', titlewords=()):
    if len(array.shape) > 2:
        print('This array appears to have more than 2 dimensions. Please ensure the array passed only has 1 dimension.')
        return

    numcol = int(np.ceil(np.sqrt(array.shape[1])))
    numrow = int(np.ceil(array.shape[1] / numcol))
    figure, axis = plt.subplots(nrows=numrow, ncols=numcol,
                                figsize=(subplotheight * numcol, subplotwidth * numrow))

    if numcol == 1 and numrow == 1:
        axis.plot(range(array.shape[0]), array[:, :])
        axis.set_title(f'{imagetype}_Fluor_across_planes')
        axis.grid(which='both')
        axis.minorticks_on()
    else:
        for i in range(numrow):
            for j in range(numcol):
                idx = i * numcol + j
                if idx >= array.shape[1]:
                    axis[i, j].axis('off')
                    continue
                axis[i, j].plot(range(array.shape[0]), array[:, idx])
                axis[i, j].set_title(f'{titlewords[idx]}')
                axis[i, j].set_yscale(yscaletype)
                axis[i, j].grid(which='both')
                axis[i, j].minorticks_on()

    figure.subplots_adjust(hspace=0.4, wspace=0.4)
    figure.savefig(os.path.join(dest, filename), dpi=300, format='png', bbox_inches='tight')
    plt.close(figure)

##############
##############

def plot_array_hist_smoothing(array, smoothingvector, filename, dest,
                              subplotwidth=5, subplotheight=5, hspace=0.4, wspace=0.4,
                              yscaletype='log', titleword='Plane'):
    if len(array.shape) > 2:
        print('This array appears to have more than 2 dimensions. Please ensure the array passed only has 1 dimension.')
        return

    numcol = int(np.ceil(np.sqrt(array.shape[1])))
    numrow = int(np.ceil(array.shape[1] / numcol))
    figure, axis = plt.subplots(nrows=numrow, ncols=numcol,
                                figsize=(subplotheight * numcol, subplotwidth * numrow))

    if numcol == 1 and numrow == 1:
        axis.plot(range(array.shape[0]), array[:, :])
        axis.set_title('Fluor_across_planes')
        axis.grid(which='both')
        axis.minorticks_on()
    else:
        for i in range(numrow):
            for j in range(numcol):
                idx = i * numcol + j
                if idx >= array.shape[1]:
                    axis[i, j].axis('off')
                    continue
                axis[i, j].plot(range(array.shape[0]), array[:, idx])
                axis[i, j].set_title(f'{titleword}_{smoothingvector[idx]}')
                axis[i, j].set_yscale(yscaletype)
                axis[i, j].grid(which='both')
                axis[i, j].minorticks_on()

    figure.subplots_adjust(hspace=0.4, wspace=0.4)
    figure.savefig(os.path.join(dest, filename), dpi=450, format='png', bbox_inches='tight')
    plt.close(figure)

##############
##############

def find_local_max(array, windowsize=5, minthreshold=100):
    """Indices i such that array[i] is the strict max in a window of size
    `windowsize` around it and array[i] > minthreshold.
    """
    if len(array.shape) > 1:
        print('This appears to have more than 1 dimension. Please re-try after flattening the array.')
        return

    localmaxindexes = []
    n = array.shape[0]
    for index in range(n):
        vec = np.arange(index - windowsize, index + windowsize, 1)
        # Drop the centre element so a run of zeros doesn't classify every
        # element as a local max.
        vec = np.delete(vec, windowsize)
        vec = vec[(vec >= 0) & (vec < n)]
        if array[index] > np.max(array[vec]) and array[index] > minthreshold:
            localmaxindexes.append(index)
    return localmaxindexes

##############
##############

def stdev_half_gaussian(bincounts, localmaxindex, threshold=0.0001):
    """Fit a single-component GMM to a symmetric reflection of the left tail
    around `localmaxindex` and return its stdev. Used as a background-noise
    estimate.
    """
    if len(bincounts.shape) > 1:
        print('This appears to have more than 1 dimension. Please re-try after flattening the array.')
        return

    vec = bincounts[:localmaxindex]
    vec = vec[vec > threshold * bincounts[localmaxindex]]
    # Mirror the left half around the peak to build a synthetic symmetric gaussian.
    vec2 = np.concatenate([np.delete(vec, vec.shape[0] - 1), vec[::-1]])
    gmm = mixture.GaussianMixture(n_components=1)
    xaxis = range(vec2.shape[0])
    total = np.c_[vec2, xaxis]
    gmm.fit(total)
    variance = gmm.covariances_[0][1][1]
    return np.sqrt(variance)

##############
##############

def fluorescence_decay(array, globalmaxindex):
    """Return (decay_left, decay_right): fractional drop from peak to the min
    on each side of `globalmaxindex`.
    """
    if len(array.shape) > 1:
        print('This appears to have more than 1 dimension. Please re-try after flattening the array.')
        return

    vecL = array[:globalmaxindex + 1]
    vecR = array[globalmaxindex:]
    decayL = (np.max(vecL) - np.min(vecL)) / np.max(vecL)
    decayR = (np.max(vecR) - np.min(vecR)) / np.max(vecR)
    return decayL, decayR

##############
##############

def init_qc_params(imagename=None, globalmode=None, globalstdev=None, fluorcurve=None,
                   maxplane=None, decayfrommaxL=None, decayfrommaxR=None,
                   propimcovered=None, flags=None):
    return {'imagename': [], 'global_mode': [], 'global_stdev': [], 'fluor_curve': [],
            'max_plane': [], 'decay_from_max_L': [], 'decay_from_max_R': [],
            'prop_im_covered': [], 'flags': []}

##############
##############

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

##############
##############
