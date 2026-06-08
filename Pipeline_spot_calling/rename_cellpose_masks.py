import os
import sys
from utils import rename_cellpose_mask

def rename_all(seg_dir):
    """Rename every cp_masks file in `seg_dir` to F<fov>_cp_masks.tif."""
    renamed, skipped = 0, 0
    for f in sorted(os.listdir(seg_dir)):
        if not f.endswith('_cp_masks.tif'):
            continue
        # Already in the target form? Skip.
        if f[1:].replace('_cp_masks.tif', '').isdigit() and f.startswith('F'):
            skipped += 1
            continue
        try:
            new = rename_cellpose_mask(os.path.join(seg_dir, f))
            print(f'  {f}\n    -> {new}')
            renamed += 1
        except ValueError as e:
            print(f'  SKIP {f}: {e}')
            skipped += 1
    print(f'\nDone. Renamed {renamed}; skipped {skipped}.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python rename_cellpose_masks.py <segmentation_dir>')
    rename_all(sys.argv[1])