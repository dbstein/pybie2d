import numpy as np
import scipy as sp
import scipy.spatial

class Close_Corrector(object):
    """
    This class impelements a "close evaluator" for use in
    Boundary Integral methods

    For now, this class assumes that all points in 'target' lie either in
    the interior of the boundary, or the exterior of the boundary
    The side should be provided by the user

    Instantiation: see documentation to self.__init__()
    Methods:
        __call__():
            computes u = u + Ct - Nt, where Ct is the corrected quadrature
            applied to the density t, and Nt is the naive quadrature applied
            to the density t
    """
    def __init__(self, source, target, side, do_DLP=False, DLP_weight=None,
            do_SLP=False, SLP_weight=None, kernel='laplace', backend='fly'):
        """
        Constructs an object which corrects naive evaluation of the provided
        kernel for the given source/target pairing

        Parameters:
            source,    required, type(Boundary_Element)
            target,    required, type(Target_Element)
            side,      required, str, ('i' or 'e'),       interior or exterior
            do_DLP,     optional, bool,  include DLP correction
            DLP_weight, optional, float, weight to apply to DLP
            do_SLP,     optional, bool,  include SLP correction
            SLP_weight, optional, float, weight to apply to SLP
            kernel,     optional, str,   kernel to use
            backend,    optional, str,   list of acceptable backends
                (this is determined by the kind of source in pairing)
        """
        self.source = source
        self.target = target
        self.side = side
        self.do_DLP = do_DLP
        self.DLP_weight = DLP_weight
        self.do_SLP = do_SLP
        self.SLP_weight = SLP_weight
        self.kernel = kernel
        self.backend = backend
        self.preparations, self.correction_function = \
            self.source.Get_Close_Correction_Function(self.target, self.side,
                self.do_DLP, self.DLP_weight, self.do_SLP, self.SLP_weight,
                self.kernel, self.backend)
    # end __init__ function definition
    def __call__(self, tau):
        return self.correction_function(tau, self.preparations)
