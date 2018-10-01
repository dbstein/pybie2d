import numpy as np
import scipy as sp
import scipy.special
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os

from ...kernels.high_level.laplace import Laplace_Layer_Form
from .laplace_close_quad import Compensated_Laplace_Form
from ...misc.numba_special_functions import numba_k0, numba_k1, numba_i0, numba_i1, _numba_i0

class Modified_Helmholtz_Close_Quad(object):
    """
    Module providing Modified Helmholtz Close Eval
    """
    def __init__(self, GSB):
        """
        Initializes the Close Quad Module

        GSB (required): boundary of type Global_Smooth_Boundary
        Currently only handles Laplace
        """
        self.boundary = GSB
        GSB.add_module('Laplace_Close_Quad')

    def Form(self, target, side, k=1.0, do_DLP=False, do_SLP=False):
        return Compensated_Laplace_Form(self.boundary, target, side, do_DLP,
            do_SLP, gradient, main_type, gradient_type, forstokes)

    def Get_Close_Corrector(self, target, side, k=1.0, do_DLP=False, do_SLP=False):
        return Modified_Helmholtz_Close_Corrector(self.boundary, target, side, k, do_DLP, do_SLP)

class Modified_Helmholtz_Close_Corrector(object):
    def __init__(self, source, target, side, k, do_DLP, do_SLP, backend='preformed'):
        self.source = source
        self.target = target
        self.side = side
        self.k = k
        self.do_DLP = do_DLP
        self.do_SLP = do_SLP
        self.backend = 'preformed'
        self.preformed = True
        # determine which internal function to call
        self.simple = _numba_i0(20*self.source.max_h*k) < 1e5
        if not self.simple:
            warnings.warn('The modified helmholtz solver may suffer serious errors without refinement.')
        self.prepare = self._prepare_formed if self.preformed else self._prepare_apply
        self.call_func = self._call_formed if self.preformed else self._call_apply
        self.prepare()
    def __call__(self, *args, **kwargs):
        self.call_func(*args, **kwargs)
    def _prepare_formed(self):
        if self.simple:
            self.correction_mat = Modified_Helmholtz_Correction_Form(self.source, self.target, self.side, self.k, self.do_DLP, self.do_SLP)
        else:
            self.correction_mat = Modified_Helmholtz_Correction_Form(self.source, self.target, self.side, self.k, self.do_DLP, self.do_SLP, self.simple)
    def _prepare_apply(self):
        pass
    def _call_formed(self, u, tau, close_pts):
        u[close_pts] += self.correction_mat.dot(tau)
    def _call_apply(self, u, tau, close_pts):
        pass

def Modified_Helmholtz_Correction_Form(source, target, side, k, do_DLP, do_SLP, simple=True):
    tx = target.x[:,None]
    ty = target.y[:,None]
    sx = source.x
    sy = source.y
    dx = ne.evaluate('tx-sx')
    dy = ne.evaluate('ty-sy')
    nx = source.normal_x
    ny = source.normal_y
    r = ne.evaluate('sqrt(dx**2 + dy**2)')
    kr = k*r
    # construct windowing function
    flat_width = 5
    if _numba_i0(30*source.max_h*k) < 1e2:
        trans_width = 25
        eps = 1e-8
    else:
        trans_width = 15
        eps = 1e-5
    aw = trans_width + flat_width
    dw = aw - flat_width
    erf = sp.special.erf
    erfinv = sp.special.erfinv
    alpha = 2*erfinv(2*eps-1)/(dw*source.max_h)
    beta = erfinv(2*eps-1) - aw*alpha*source.max_h
    window = lambda x: 0.5*(erf(alpha*x + beta)+1)
    max_eval = aw*source.max_h*k
    if do_DLP:
        CL = Compensated_Laplace_Form(source, target, side, do_DLP=True)
        L = Laplace_Layer_Form(source, target, ifdipole=True)
        A1 = ne.evaluate('(CL - L)')
        CL = Compensated_Laplace_Form(source, target, side, do_SLP=True)
        L = Laplace_Layer_Form(source, target, ifcharge=True)
        A2_0 = ne.evaluate('(CL - L)')
        if _numba_i1(max_eval) < 1e5:
            adj1 = numba_i1(kr)
        elif max_eval**5/384.0 < 1e5:
            adj1 = kr/2.0+kr**3/16.0+kr**5/384.0
        elif max_eval**3/16.0 < 1e5:
            adj1 = kr/2.0+kr**3/16.0
        else:
            adj1 = kr/2.0
        adj1 *= window(r)
        adj = ne.evaluate('-adj1*k*(nx*dx+ny*dy)/r')
        A2 = ne.evaluate('A2_0*adj')
        CC = ne.evaluate('(A1 + A2)')
    if do_SLP:
        if not do_DLP:
            CL = Compensated_Laplace_Form(source, target, side, do_SLP=True)
            L = Laplace_Layer_Form(source, target, ifcharge=True)
            A2_0 = ne.evaluate('(CL - L)')
        if _numba_i0(max_eval) < 1e5: 
            adj0 = numba_i0(kr)
        elif max_eval**4/64.0 < 1e5:
            adj0 = (1+kr**2/4.0+kr**4/64.0)
        elif max_eval**2/4.0 < 1e5:
            adj0 = (1+kr**2/4.0)
        else:
            adj0 = 1.0
        adj0 *= window(r)
        CC = ne.evaluate('A2_0*adj0')
    return CC
