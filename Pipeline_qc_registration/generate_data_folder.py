"""
Generate the data folder layout the pipeline expects, by symlinking
raw acquisition files into a flattened directory structure.

Source layout (one subdir per acquisition batch):
    SOURCE/
      ab/                 -- antibody tifs + metadata
      hyb00-04/           -- raw hybs; bit numbering is LOCAL to this directory
      hyb05-10/
      hyb11-17/
      prehyb/             -- DAPI / prehyb tifs + metadata

Destination layout (what the pipeline consumes):
    DEST/
      ab/                 -- symlinks to SOURCE/ab/*
      hyb/                -- symlinks to SOURCE/hyb*/*, with bit numbers
                             RENUMBERED to be GLOBAL across all hyb subdirs
      prehyb/             -- symlinks to SOURCE/prehyb/*
      dapi -> prehyb/     -- pipeline reads DAPI from the prehyb files
      fpkm_data.txt       -- optionally placed via --codebook (lives at root)
      segmentation/       -- left empty; filled in by Cellpose downstream
      registration/       -- left empty; filled in by registration pipeline
      stitching/          -- left empty; filled in by stitching pipeline

Usage:
    python generate_data_folder.py --source SRC --dest DST
                                   [--codebook PATH]
                                   [--skip_subdirs hyb17,prehyb2]
"""

import argparse
import glob
import os
import shutil


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _renumber_bit(filename, bitoffset, tempoffset):
    """
    Compute the global-bit filename for a hyb file.

    The per-directory bit number is encoded as the second-to-last `_`-
    separated field of the filename. Files lacking a numeric bit (signaled
    by 'Sona' appearing in that field -- the microscope's channel string)
    are treated as bit 0 of this directory.

    Returns the renamed filename, or None if the file's bit number exceeds
    `tempoffset` (the highest bit number actually observed in the *metadata*
    files for this directory). That mismatch indicates an incomplete
    acquisition, and the file should be skipped.
    """
    splits = filename.split('_')
    if 'Sona' in splits[-2]:
        new_bit = 0
    else:
        bit_in_dir = int(splits[-2])
        if bit_in_dir > tempoffset:
            return None
        new_bit = bit_in_dir
    return '_'.join(splits[:-2] + [str(new_bit + bitoffset)] + [splits[-1]])


def _symlink_pattern(patterns, target_dir):
    """Symlink each file matching any of `patterns` (CWD-relative) into
    `target_dir`, keeping its original name. Returns the count linked."""
    count = 0
    for p in patterns:
        for name in sorted(glob.glob(p)):
            os.symlink(os.path.join(os.getcwd(), name),
                       os.path.join(target_dir, name))
            count += 1
    return count


# ---------------------------------------------------------------------------
# Per-subdir handling
# ---------------------------------------------------------------------------

def parse_dirtype(subdir, bitoffset, dest):
    """
    Process one source subdirectory (CWD is assumed to be that subdir).
    Returns the (possibly updated) global bitoffset.

    Three cases by subdir name:
        - antibody dirs ('Ab*' / 'ab*' but not 'Ab_*' / 'ab_*')
        - prehyb dirs   ('prehyb*')
        - hyb dirs      ('hyb*' but not 'prehyb*' or '*_hyb*')
    """
    # ---- antibody dir ----
    if ('Ab' in subdir or 'ab' in subdir) and ('Ab_' not in subdir and 'ab_' not in subdir):
        n = _symlink_pattern(
            ['Ab*.tif', 'ab*.tif', 'Ab*metadata.txt', 'ab*metadata.txt'],
            os.path.join(dest, 'ab'),
        )
        print(f'  ab: linked {n} files')
        return bitoffset

    # ---- prehyb dir ----
    if 'prehyb' in subdir:
        n = _symlink_pattern(
            ['prehyb*.tif', 'prehyb*metadata.txt'],
            os.path.join(dest, 'prehyb'),
        )
        print(f'  prehyb: linked {n} files')
        return bitoffset

    # ---- hyb dir ----
    # Hyb directories are the complex case: bit numbers in source filenames
    # are LOCAL to the directory, and we have to renumber them to GLOBAL bit
    # indices using a running offset.
    if 'hyb' in subdir and 'prehyb' not in subdir and '_hyb' not in subdir:
        # Parse the directory's stated bit range from its name, e.g. 'hyb00-04'
        # -> dirbitoffset=0, or 'hyb05' -> dirbitoffset=5.
        numinfo = subdir[3:]
        dashindex = numinfo.find('-')
        dirbitoffset = int(numinfo[:dashindex]) if dashindex != -1 else int(numinfo)
        print(f'  hyb subdir {subdir}: stated offset = {dirbitoffset}, '
              f'running offset = {bitoffset}')

        tifffiles = sorted(glob.glob('hyb*.tif'))
        metafiles = sorted(glob.glob('hyb*metadata.txt'))

        # tempoffset = highest bit number actually present in this directory
        # (read from the metadata filenames, +1 for the 'Sona' / bit-0 file).
        # This may be LESS than the directory name suggests if the
        # acquisition was incomplete.
        tempoffset = 0
        for meta in metafiles:
            splits = meta.split('_')
            if 'Sona' not in splits[-2] and int(splits[-2]) > tempoffset:
                tempoffset = int(splits[-2])
        tempoffset += 1  # +1 for the 'Sona' / bit-0 file

        # Symlink tifs and metadata, both renumbered with the running offset.
        # The same helper handles both (the rename logic is identical) -- this
        # also ensures the incomplete-acquisition skip is consistent for tifs
        # and metas (previously: duplicated logic with a latent NameError /
        # silent-duplicate-symlink bug when a file's bit exceeded tempoffset).
        for fname in tifffiles + metafiles:
            new_name = _renumber_bit(fname, bitoffset, tempoffset)
            if new_name is None:
                print(f'    skip {fname} '
                      f'(bit > metadata range; incomplete acquisition?)')
                continue
            os.symlink(os.path.join(os.getcwd(), fname),
                       os.path.join(dest, 'hyb', new_name))

        # Advance the running global bitoffset.
        # On the first hyb dir we just take tempoffset; subsequent ones add to
        # it. We trust tempoffset (read from metadata) over dirbitoffset (read
        # from the directory name) so that incomplete acquisitions don't leave
        # gaps in the global bit numbering.
        if dirbitoffset == 0:
            bitoffset = tempoffset
        else:
            bitoffset += tempoffset
        print(f'  hyb subdir {subdir}: new running offset = {bitoffset}')

    return bitoffset


def parse_source(source, dest, skip_subdirs=()):
    """Walk each subdir of `source` (alphabetical order so hyb dirs are
    visited in numerical order) and symlink its contents into `dest`."""
    subdirs = sorted(os.listdir(source))
    print(f'Found subdirs: {subdirs}')

    bitoffset = 0
    for subdir in subdirs:
        if any(pat in subdir for pat in skip_subdirs):
            print(f'  skipping {subdir} (matches --skip_subdirs)')
            continue
        full = os.path.join(source, subdir)
        if not os.path.isdir(full):
            continue
        os.chdir(full)
        bitoffset = parse_dirtype(subdir, bitoffset, dest)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def run_generate(args):
    print(f'source: {args.source}')
    print(f'dest:   {args.dest}')

    if not os.path.exists(args.source):
        print(f'Source directory not found: {args.source}')
        print('  (please pass a full path, e.g. '
              '/mnt/.../20231101_minimal_example)')
        return
    if os.path.exists(args.dest):
        print(f'Destination already exists: {args.dest}')
        print('  Refusing to overwrite. Choose a new --dest path.')
        return

    # Create the destination skeleton.
    os.makedirs(args.dest)
    for sub in ('ab', 'hyb', 'prehyb', 'segmentation', 'registration', 'stitching'):
        os.makedirs(os.path.join(args.dest, sub))
        # Create the segmentation / cellpose_modelD folder:
        if sub=='segmentation':
            os.makedirs(os.path.join(args.dest, sub, 'cellpose_modelD'))
    # The pipeline reads DAPI from the prehyb files; expose them as a symlink.
    os.symlink(os.path.join(args.dest, 'prehyb'),
               os.path.join(args.dest, 'dapi'))

    # Optionally copy a codebook into dest/fpkm_data.txt.
    # The orchestrator reads it directly from ws.data_path (no codebook/ subdir).
    # (Previously: hardcoded path triggered when 'LOSBDA' appeared in --source,
    # and the copy went to dest/codebook/fpkm_data.txt.)
    if args.codebook:
        if os.path.isfile(args.codebook):
            print(f'Copying codebook from {args.codebook}')
            shutil.copyfile(args.codebook,
                            os.path.join(args.dest, 'fpkm_data.txt'))
        else:
            print(f'Warning: --codebook path not found: {args.codebook} '
                  f'(skipping codebook copy)')

    skip = tuple(s.strip() for s in args.skip_subdirs.split(',') if s.strip())
    if skip:
        print(f'Skipping subdirs matching: {skip}')
    parse_source(args.source, args.dest, skip_subdirs=skip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate the pipeline data folder layout by symlinking '
                    'raw acquisition files.'
    )
    parser.add_argument('--source', required=True,
                        help='Raw acquisition directory '
                             '(contains ab/, hyb*/, prehyb/ subdirs).')
    parser.add_argument('--dest', required=True,
                        help='Destination directory to create. Must not '
                             'already exist.')
    parser.add_argument('--codebook', default=None,
                        help='Optional path to a codebook (fpkm_data.txt) '
                             'to copy into dest/ (read by the orchestrator '
                             'directly from the data root).')
    parser.add_argument('--skip_subdirs', default='',
                        help='Comma-separated substrings; source subdirs '
                             'containing any of these are skipped. '
                             'Previously hardcoded as "hyb17,prehyb2" for '
                             'specific datasets -- now pass them explicitly '
                             'if needed.')
    args = parser.parse_args()
    run_generate(args)
