"""
File discovery, directory parsing, and image reading helpers.

This module holds the "image catalog" data structure used throughout the
pipeline: a dict-of-lists keyed by filename, imagename, ch, cycle_ch, etc.
"""
import gc
import glob
import os
import time

import numpy as np
import tifffile
from skimage.io import imread


##############
##############

def find_links(directory, fov, filetype='.tif', fovs_to_exclude=None):
    """Return tif filenames for a given FOV (or all FOVs if fov==999).

    `fov` is a string (e.g. '000', '014', '101'), unless it is 999 meaning
    "use all FOVs". If the FOV string is shorter than the max FOV-string
    found in the directory, leading zeros are prepended so 'F002' matches
    when the user passed '2'.
    """
    alllinks = glob.glob('*' + filetype)
    maxlength = max((len(link.split('_')[-1][1:-4]) for link in alllinks), default=0)

    if fov != 999:
        # Specific FOV: pad with leading zeros to match the directory's width.
        numzerostoappend = maxlength - len(str(fov))
        links = glob.glob('*' + 'F' + '0' * numzerostoappend + fov + filetype)
        return links

    # fov == 999: return everything, minus anything in fovs_to_exclude.
    print('returning all links')
    if fovs_to_exclude is None:
        print(alllinks)
        return list(alllinks)

    fovs_excluded = fovs_to_exclude.split(',')
    for i, fov_to_exclude in enumerate(fovs_excluded):
        numzerostoappend = maxlength - len(str(fov_to_exclude))
        fovs_excluded[i] = 'F' + '0' * numzerostoappend + fov_to_exclude
    print(fovs_excluded)

    tokeep = [f for f in alllinks if not any(x in f for x in fovs_excluded)]
    print(tokeep)
    return tokeep

##############
##############

def process_links(links, allchs, source, directory, keyword, hybcode='', numchs=2, multiplier=1000):
    for link in links:
        if '.tif' not in link:
            continue
        imagename = link.split('_')
        fovid = imagename[-1][1:-4]
        hybid = int(imagename[-2]) if hybcode == '' else hybcode

        for i in range(int(numchs)):
            allchs['filename'].append(os.path.join(source, directory, link))
            allchs['imagetype'].append(keyword)
            allchs['imagename'].append(directory + '_fov_' + fovid + '_cycle_' + str(hybid) + '_ch_' + str(i))
            allchs['image'].append(0)
            allchs['ch'].append(i)
            # Sort first by channel, then by hyb index: channel 0/hyb 0 first, etc.
            allchs['cycle_ch'].append((i + 1) * multiplier + hybid)

    return allchs

##############
##############

# (keyword, hybcode, numchs, multiplier) per directory type recognised below.
_DIR_RULES = (
    ('hyb',    '', 2, 1000),   # 'hyb*' but not 'prehyb*'
    ('prehyb', 99, 3, 100),
    ('ab',     88, 4, 10000),  # 'ab*'/'Ab*' but not 'ab_*'/'Ab_*'
)


def _matches_dir(directory, keyword):
    if keyword == 'hyb':
        return 'hyb' in directory and 'prehyb' not in directory
    if keyword == 'prehyb':
        return 'prehyb' in directory
    if keyword == 'ab':
        return (('ab' in directory or 'Ab' in directory)
                and 'ab_' not in directory and 'Ab_' not in directory)
    return False


def parse_directory(source, fov, fovs_exclude=None, allchs=None, printoutput='True'):
    dirlist = sorted(os.listdir(source))
    if printoutput == 'True':
        print(f'Dirlist in utils.parse_directory is {dirlist}.')

    if allchs is None:
        allchs = {'filename': [], 'imagetype': [], 'imagename': [],
                  'image': [], 'cycle_ch': [], 'ch': []}

    for directory in dirlist:
        for keyword, hybcode, numchs, multiplier in _DIR_RULES:
            if not _matches_dir(directory, keyword):
                continue
            os.chdir(os.path.join(source, directory))
            if printoutput == 'True':
                print(f'Choosing {keyword} FOV{fov}: {os.path.join(source, directory)}')
            links = sorted(find_links(os.getcwd(), fov, filetype='.tif', fovs_to_exclude=fovs_exclude))
            allchs = process_links(links, allchs, source, directory, keyword=keyword,
                                   hybcode=hybcode, numchs=numchs, multiplier=multiplier)

    sort_index = sorted(range(len(allchs['cycle_ch'])), key=lambda i: allchs['cycle_ch'][i])
    for key in list(allchs.keys()):
        allchs[key] = [allchs[key][i] for i in sort_index]

    return allchs

##############
##############

def _select_channel(image, channel_index, axis):
    """Extract one channel from a 4D (Z, X, Y, Channel)-ish stack.

    The channel axis position is detected from the smallest dim. This wraps
    that branching into a single numpy call.
    """
    return np.copy(np.moveaxis(image, axis, 0)[channel_index])


def _read_ims(allchs):
    ims = allchs.get('filename')
    imsunique = list(dict.fromkeys(ims))  # ordered unique

    for im in imsunique:
        indicestoupdate = [i for i, val in enumerate(ims) if im == val]
        st = time.time()

        # Open via tifffile so we can read the actual axis order from the
        # TIFF metadata. `series[0].axes` is a string like 'ZCYX', 'CZYX',
        # 'ZYX', 'TZCYX', etc. — derived from OME-XML, ImageJ metadata,
        # or inferred from the page layout. skimage.io.imread uses tifffile
        # internally but discards this axis-order information.
        with tifffile.TiffFile(im) as tif:
            image = tif.asarray()
            axes = tif.series[0].axes

        if 'C' in axes:
            channel_axis = axes.index('C')
            numchannels = image.shape[channel_axis]
            print(f'Num channels: {numchannels} --- Channel axis from TIFF metadata: '
                  f'{channel_axis} (axes={axes!r})')
            for i in range(numchannels):
                stack = _select_channel(image, i, channel_axis)
                indextoupdate = indicestoupdate[i]
                allchs['image'][indextoupdate] = stack
        elif image.ndim == 4:
            # Fallback for 4D TIFFs that have no 'C' axis in their metadata
            # (e.g. raw multi-page TIFFs written without OME/ImageJ headers).
            # Uses the original "smallest axis = channel" heuristic.
            numchannels = min(image.shape)
            channel_axis = image.shape.index(numchannels)
            print(f'No C axis in TIFF metadata (axes={axes!r}); '
                  f'falling back to smallest-axis heuristic. '
                  f'Num channels: {numchannels} --- Index: {channel_axis}')
            for i in range(numchannels):
                stack = _select_channel(image, i, channel_axis)
                indextoupdate = indicestoupdate[i]
                allchs['image'][indextoupdate] = stack

        et = time.time()
        del image
        gc.collect()
        print(f'imread for index {indextoupdate} = {et - st} seconds.')

    return allchs

##############
##############

def process_links_stitching(links, allfovs, source, directory, keyword, hybcode='', chtouse=2, numchs=2, multiplier=1000):
    for link in links:
        if '.tif' not in link:
            continue
        imagename = link.split('_')
        fovid = imagename[-1][1:-4]
        hybid = int(imagename[-2]) if hybcode == '' else hybcode

        allfovs['filename'].append(os.path.join(source, directory, link))
        allfovs['imagetype'].append(keyword)
        allfovs['imagename'].append(directory + '_fov_' + fovid + '_cycle_' + str(hybid) + '_ch_' + str(chtouse))
        allfovs['image'].append(0)
        allfovs['mip'].append(0)
        allfovs['ch'].append(chtouse)
        allfovs['fov'].append(int(fovid))
        allfovs['cycle_ch'].append((chtouse + 1) * multiplier + hybid)

    return allfovs

##############
##############

def parse_directory_stitching(source, fov, allfovs):
    dirlist = sorted(os.listdir(source))
    print(dirlist)
    if allfovs is None:
        allfovs = {'filename': [], 'imagetype': [], 'imagename': [], 'image': [],
                   'mip': [], 'cycle_ch': [], 'ch': [], 'fov': []}

    for directory in dirlist:
        if 'prehyb' not in directory:
            continue
        os.chdir(os.path.join(source, directory))
        print(f'Choosing prehyb FOV{fov}: {os.path.join(source, directory)}')
        links = sorted(find_links(os.getcwd(), fov, filetype='.tif'))
        allfovs = process_links_stitching(links, allfovs, source, directory,
                                          keyword='prehyb', hybcode=99, numchs=3, multiplier=100)

    sort_index = sorted(range(len(allfovs['cycle_ch'])), key=lambda i: allfovs['cycle_ch'][i])
    for key in list(allfovs.keys()):
        allfovs[key] = [allfovs[key][i] for i in sort_index]

    return allfovs

##############
##############

def _read_ims_stitching(allfovs, fovstoread=(), chtouse=2):
    if len(fovstoread) == 0:
        print('Reading all files in allfovs.')
        ims = allfovs.get('filename')
    else:
        print(f'Reading from fovs:{fovstoread} within allfovs.')
        ims = [val for i, val in enumerate(allfovs.get('filename'))
               if allfovs['fov'][i] in fovstoread]

    imsunique = list(dict.fromkeys(ims))

    for im in imsunique:
        indicestoupdate = [i for i, val in enumerate(allfovs.get('filename')) if im == val]
        st = time.time()
        image = imread(im)

        if len(image.shape) == 4:
            numchannels = min(image.shape)
            index = image.shape.index(numchannels)
            print(f'Num channels: {numchannels} --- Index of channel info: {index}')

            stack = _select_channel(image, chtouse, index)
            indextoupdate = indicestoupdate[0]
            allfovs['image'][indextoupdate] = stack
            allfovs['mip'][indextoupdate] = np.max(stack, axis=0)

        elif len(image.shape) == 3:
            indextoupdate = indicestoupdate[0]
            allfovs['image'][indextoupdate] = image
            allfovs['mip'][indextoupdate] = np.max(image, axis=0)

        del image
        gc.collect()
        et = time.time()
        print(f'imread for index {indextoupdate} = {et - st} seconds.')

    return allfovs, allfovs['mip'][0]

##############
##############

def _update_data(data, kv):
    """Append values to `data` (a dict of lists).

    `kv` may be a dict {key: value, ...} or a sequence of (key, value) tuples
    (the old API). Keys missing from `data` are skipped with a warning.
    """
    items = kv.items() if isinstance(kv, dict) else kv
    for k, v in items:
        if k not in data:
            print(f'Key: {k} not found in data; omitting update.')
        else:
            data[k].append(v)
    return data
