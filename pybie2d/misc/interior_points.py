import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib as mpl
import matplotlib.path

from ..kernels.cauchy import Cauchy_Layer_Apply
from ..point_set import PointSet

def find_interior_points(source, target, boundary_acceptable=False):
    """
    quick finding of which points in target are outside vs. inside
    """
    # first exclude things outside of bounding box
    xmin = source.x.min()
    xmax = source.x.max()
    ymin = source.y.min()
    ymax = source.y.max()
    in_bounding_box = np.logical_and.reduce([ target.x > xmin, target.x < xmax,
                                              target.y > ymin, target.y < ymax])
    out_bounding_box = np.logical_not(in_bounding_box)
    small_targ = PointSet(c=target.c[in_bounding_box], compute_tree=False)
    wn = np.zeros(target.N, dtype=complex)
    wn[out_bounding_box] = 0.0
    # compute winding number via cauchy sums
    wn[in_bounding_box] = Cauchy_Layer_Apply(source, small_targ, \
                                                    dipstr=source.ones_vec).real
    wn = np.abs(wn)
    bad = np.logical_or(np.isnan(wn), np.isinf(wn))
    good = np.logical_not(bad)
    big = np.zeros_like(wn)
    big[good] = wn[good] > 1e5
    bad = np.logical_or(big, bad)
    wn[bad] = 1.0
    # get region where that sum was not accurate enough
    dist = source.tolerance_to_distance(1e-2)
    q = target.find_near_points(source, dist).ravel()
    # phys array, good except in near boundary region
    wn[q] = 0.0
    phys = wn > 0.5
    # brute force search
    poly = mpl.path.Path(source.stacked_boundary)
    xq = target.x[q]
    yq = target.y[q]
    tq = np.column_stack([xq, yq])
    interior = poly.contains_points(tq)
    phys[q] = interior
    phys[bad] = boundary_acceptable
    ext = np.logical_not(phys)
    return phys, ext
