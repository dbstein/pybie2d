import numpy as np
import scipy as sp
import scipy.spatial
from .misc.near_points import find_near_points
from .point_set import PointSet
from .close import Close_Corrector

class Pairing(object):
    """
    This class impelements a "pairing" between a source and a target
    for use in Boundary Integral methods

    It assumes that all points are on one side, which must be provided by the
    user

    Instantiation: see documentation to self.__init__()
    """
    def __init__(self, source, target, side='i', error_tol=1e-12):
        """
        This function initializes the source --> target pairing

        source,       required, type(boundary_element)
        target,       required, type(target_element)
        side,         optional, str, 'i' or 'e'
        error_tol,    optional, float, error tolerance for close evaluations
        """
        self.source = source
        self.target = target
        self.side = side
        self.error_tol = error_tol
        self.close_distance = source.tolerance_to_distance(error_tol)
        self.close_points = self.target.find_near_points(self.source, 
                                                        self.close_distance)
        self.close_targ = PointSet(c=self.target.c[self.close_points],
                                                            compute_tree=False)
        self.close_correctors = {}
    # end __init__ function definition

    def Setup_Close_Corrector(self, do_DLP=False, DLP_weight=None,
            do_SLP=False, SLP_weight=None, kernel='laplace', backend='fly'):
        code = (do_DLP, DLP_weight, do_SLP, SLP_weight, kernel, backend)
        self.close_correctors[code] = \
        Close_Corrector(self.source, self.close_targ, self.side, do_DLP,
                                DLP_weight, do_SLP, SLP_weight, kernel, backend)
        return code

    def Close_Correction(self, u, tau, code):
        u[self.close_points] += self.close_correctors[code](tau)

