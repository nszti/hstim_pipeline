from typing import List, Tuple, Dict, Any
from itertools import count
import numpy as np
from scipy.ndimage import percentile_filter

from suite2p.detection.sparsedetect import extendROI
from suite2p import default_ops
from suite2p.extraction.masks import create_neuropil_masks


def create_neuropil_masks(ypixs, xpixs, cell_pix, inner_neuropil_radius,
                          min_neuropil_pixels, circular=False):
    """ creates surround neuropil masks for ROIs in stat by EXTENDING ROI (slower if circular)

    Parameters
    ----------

    cellpix : 2D array
        1 if ROI exists in pixel, 0 if not;
        pixels ignored for neuropil computation

    Returns
    -------

    neuropil_masks : list
        each element is array of pixels in mask in (Ly*Lx) coordinates

    """
    valid_pixels = lambda cell_pix, ypix, xpix: cell_pix[ypix, xpix] < .5
    extend_by = 5

    Ly, Lx = cell_pix.shape
    assert len(xpixs) == len(ypixs)
    neuropil_ipix = []
    idx = 0
    for ypix, xpix in zip(ypixs, xpixs):
        neuropil_mask = np.zeros((Ly, Lx), bool)
        # extend to get ring of dis-allowed pixels
        ypix, xpix = extendROI(ypix, xpix, Ly, Lx, niter=inner_neuropil_radius)
        nring = np.sum(valid_pixels(cell_pix, ypix,
                                    xpix))  # count how many pixels are valid

        nreps = count()
        ypix1, xpix1 = ypix.copy(), xpix.copy()
        while next(nreps) < 100 and np.sum(valid_pixels(
                cell_pix, ypix1, xpix1)) - nring <= min_neuropil_pixels:
            if circular:
                ypix1, xpix1 = extendROI(ypix1, xpix1, Ly, Lx,
                                         extend_by)  # keep extending
            else:
                ypix1, xpix1 = np.meshgrid(
                    np.arange(max(0,
                                  ypix1.min() - extend_by),
                              min(Ly,
                                  ypix1.max() + extend_by + 1), 1, int),
                    np.arange(max(0,
                                  xpix1.min() - extend_by),
                              min(Lx,
                                  xpix1.max() + extend_by + 1), 1, int), indexing="ij")

        ix = valid_pixels(cell_pix, ypix1, xpix1)
        neuropil_mask[ypix1[ix], xpix1[ix]] = True
        neuropil_mask[ypix, xpix] = False
        neuropil_ipix.append(np.ravel_multi_index(np.nonzero(neuropil_mask), (Ly, Lx)))
        idx += 1

    return neuropil_ipix