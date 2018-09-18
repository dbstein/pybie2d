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
        self.close_targ = PointSet(c=self.target.c[self.close_points])
        self.close_correctors = {}
    # end __init__ function definition

    def Setup_Close_Corrector(self, do_DLP=False, DLP_weight=None,
            do_SLP=False, SLP_weight=None, kernel='laplace', backend='fly'):
        code = (do_DLP, DLP_weight, do_SLP, SLP_weight, kernel, backend)
        if self.close_targ.N > 0:
            self.close_correctors[code] = \
                Close_Corrector(self.source, self.close_targ, self.side, do_DLP,
                                DLP_weight, do_SLP, SLP_weight, kernel, backend)
        else:
            self.close_correctors[code] = Null_Corrector
        return code

    def Close_Correction(self, u, tau, code):
        u[self.close_points] += self.close_correctors[code](tau)

def Null_Corrector(tau):
    return 0.0

class CollectionPairing(object):
    """
    This class impelements a "pairing" between a source collection and a target
    for use in Boundary Integral methods

    Instantiation: see documentation to self.__init__()
    """
    def __init__(self, collection, target, error_tol=1e-12):
        """
        This function initializes the collection --> target pairing

        collection,   required, BoundaryCollection
        target,       required, PointSet
        error_tol,    optional, float, error tolerance for close evaluations
        """
        self.collection = collection
        self.target = target
        self.error_tol = error_tol
        self.close_correctors = []
        self.codes = {}
        for bdy, side in zip(self.collection.boundaries, self.collection.sides):
            self.close_correctors.append(
                Pairing(bdy, target, side=side, error_tol=self.error_tol)
            )
    def Setup_Close_Corrector(self, e_args={}, i_args={}):
        code = (e_args, i_args)
        icode = code
        ecode = code
        for i in range(self.collection.n_boundaries):
            args = e_args if self.collection.sides[i] == 'e' else i_args
            code = self.close_correctors[i].Setup_Close_Corrector(**args)
            if self.collection.sides[i] == 'i':
                icode = code
            else:
                ecode = code
        self.codes[code] = (ecode, icode)
        return code
    def Close_Correction(self, u, tau, code):
        ecode, icode = self.codes[code]
        for i in range(self.collection.n_boundaries):
            i1, i2 = self.collection.get_inds(i)
            code = ecode if self.collection.sides[i] == 'e' else icode
            self.close_correctors[i].Close_Correction(u, tau[i1:i2], code)

