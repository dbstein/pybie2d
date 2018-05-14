import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib as mpl
import matplotlib.path

from ..kernels.cauchy import Cauchy_Layer_Apply
from .near_points import find_near_points

def find_interior_points(source, target):
    """
    quick finding of which points in target are outside vs. inside
    """
    # compute winding number via cauchy sums
    wn = Cauchy_Layer_Apply(source, target, dipstr=source.ones_vec)
    np.abs(wn, wn)
    # get region where that sum was not accurate enough
    dist = source.tolerance_to_distance(1e-2)
    q = find_near_points(source, target, r=dist)
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
    ext = np.logical_not(phys)
    return phys, ext
