import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os

from ...misc.basic_functions import rowsum, differentiation_matrix, differentiate
from ...kernels.high_level.laplace import Laplace_Layer_Form, Laplace_Layer_Apply
from ...kernels.high_level.cauchy import Cauchy_Layer_Form, Cauchy_Layer_Apply

class Laplace_Close_Quad(object):
    """
    Module providing Laplace Close Eval based on Globaly Compensated Cauchy Quad
    """
    def __init__(self, GSB):
        """
        Initializes the Close Quad Module

        GSB (required): boundary of type Global_Smooth_Boundary
        """
        self.boundary = GSB
        GSB.add_module('Laplace_CSLP_Self_Kress')
        inside_point = self.boundary.get_inside_point()
        complex_weights = self.boundary.complex_weights
        c = self.boundary.c
        t = self.boundary.t
        N = self.boundary.N

        self.sawlog = -1j*t + np.log(inside_point - c)
        self.sawlog.imag = np.unwrap(self.sawlog.imag)
        self.inf_scale = complex_weights/(c-inside_point) / (2.0j*np.pi)

    def get_differentiation_matrix(self):
        if not hasattr(self, 'differentiation_matrix'):
            self.differentiation_matrix = differentiation_matrix(self.boundary.N)
        return self.differentiation_matrix

    def Form(self, target, side, do_DLP=False,
        do_SLP=False, gradient=False, main_type='real', gradient_type='real', forstokes=False):
        return Compensated_Laplace_Form(self.boundary, target, side, do_DLP,
            do_SLP, gradient, main_type, gradient_type, forstokes)

    def Apply(self, target, side, tau, do_DLP=False,
        do_SLP=False, gradient=False, main_type='real', gradient_type='real', backend='fly', forstokes=False):
        return Compensated_Laplace_Apply(self.boundary, target, side, tau, do_DLP,
            do_SLP, gradient, main_type, gradient_type, backend, forstokes)

    def Get_Close_Corrector(self, target, side, do_DLP=False, do_SLP=False, backend='fly'):
        return Laplace_Close_Corrector(self.boundary, target, side, do_DLP, do_SLP, backend)

class Laplace_Close_Corrector(object):
    def __init__(self, source, target, side, do_DLP, do_SLP, backend):
        self.source = source
        self.target = target
        self.side = side
        self.do_DLP = do_DLP
        self.do_SLP = do_SLP
        self.backend = backend
        self.preformed = self.backend == 'preformed'
        self.prepare = self._prepare_formed if self.preformed else self._prepare_apply
        self.call_func = self._call_formed if self.preformed else self._call_apply
        self.prepare()
    def __call__(self, *args, **kwargs):
        self.call_func(*args, **kwargs)
    def _prepare_formed(self):
        close_mat = Compensated_Laplace_Form(self.source, self.target, self.side, self.do_DLP, self.do_SLP)
        naive_mat = Laplace_Layer_Form(self.source, self.target, ifcharge=self.do_SLP, ifdipole=self.do_DLP)
        self.correction_mat = close_mat.real - naive_mat
    def _prepare_apply(self):
        pass
    def _call_formed(self, u, tau, close_pts):
        u[close_pts] += self.correction_mat.dot(tau)
    def _call_apply(self, u, tau, close_pts):
        v1 = Compensated_Laplace_Apply(self.source, self.target, self.side, tau, do_DLP=self.do_DLP, do_SLP=self.do_SLP, backend=self.backend)
        ch = tau if self.do_SLP else None
        ds = tau if self.do_DLP else None
        v2 = Laplace_Layer_Apply(self.source, self.target, charge=ch, dipstr=ds, backend=self.backend)
        u[close_pts] += (v1.real - v2)

def Compensated_Laplace_Form(source, target, side, do_DLP=False,
        do_SLP=False, gradient=False, main_type='real', gradient_type='real', forstokes=False):
    """
    Full Formation of Close-Eval Matrix for Laplace Problem

    Parameters:
    source     (required): Boundary, source
    target     (required): PointSet, target
    side       (required): 'i' or 'e' for interior/exterior evaluation
    do_DLP     (optional): whether to include DLP evaluation
    do_SLP     (optional): whether to include SLP evaluation
    gradient   (optional): compute gradient matrices or not
    main_type  (optional): if 'real', return only real part of main matrix,
                            otherwise return both real and complex parts
    grad_type  (optional): if 'real', return two real matrices for u_x and u_y,
                            otherwise return one complex matrix with the real
                            part giving evaluation of u_x and the imaginary
                            part giving evaluation of u_y
    Returns:
    if not gradient and main_type =='real':
        MAT; real matrix such that u = MAT.dot(tau)
    if not gradient and main_type == 'complex':
        MAT; complex matrix such that u = MAT.real.dot(tau)
        the complex part of MAT is used in Stokes evaluations
    if gradient and grad_type == 'real':
        (MAT, DX_MAT, DY_MAT), tuple of matrices
            MAT as described above
            DX_MAT real matrix such that u_x = DX_MAT.dot(tau)
            DY_MAT real matrix such that u_y = DY_MAT.dot(tau)
    if gradient and grad_type == 'complex':
        (MAT, DMAT), tuple of matrices
            MAT as described above
            DMAT complex matrix such that:
                u_x = DMAT.real.dot(tau)
                u_y = -DMAT.imag.dot(tau)
    """
    N = source.N
    M = target.N
    PM = np.zeros([N, N], dtype=complex)
    if do_DLP:
        PM += compensated_laplace_dlp_preform(source, side)
    if do_SLP:
        SPM, AFTER_MATS = compensated_laplace_slp_preform(source, target, side,
                                                            gradient=gradient)
        if gradient:
            AFTER_MAT = AFTER_MATS[0]
            AFTER_DER_MAT = AFTER_MATS[1]
        else:
            AFTER_MAT = AFTER_MATS
        if forstokes:
            SPM *= 0.5
            AFTER_MAT *= 0.5
            if gradient:
                AFTER_DER_MAT *= 0.5
        PM += SPM
    cauchy_mats = compensated_cauchy_form(source, target, side,
                                                        derivative=gradient)
    if gradient:
        cauchy_mat = cauchy_mats[0]
        der_cauchy_mat = cauchy_mats[1]
    else:
        cauchy_mat = cauchy_mats
    MAT1 = cauchy_mat.dot(PM)
    if gradient:
        der_cauchy_mat = cauchy_mats[1]
        MATD = der_cauchy_mat.dot(PM)
    if do_SLP:
        MAT1 += AFTER_MAT
        if gradient:
            MATD += AFTER_DER_MAT
    MAT = MAT1.real if main_type == 'real' else MAT1
    if gradient:
        if gradient_type == 'real':
            ret = (MAT, MATD.real, -MATD.imag)
        else:
            ret = (MAT, MATD)
    else:
        ret = MAT
    return ret

def compensated_cauchy_form(source, target, side, derivative=False):
    sc = source.c
    scT = sc[:,None]
    tcT = target.c[:,None]
    cw = source.complex_weights
    cwT = cw[:,None]

    comp = Cauchy_Layer_Form(source, target)
    J0 = rowsum(comp)
    if side == 'e':
        J0 += 1.0
    prefac = 1.0/J0
    prefacT = prefac[:,None]

    MAT = ne.evaluate('prefacT*comp')

    if derivative:
        # get Schneider-Werner derivative matrix
        DMAT = 2.0j*np.pi*Cauchy_Layer_Form(source, source)
        np.fill_diagonal(DMAT, 0.0)
        np.fill_diagonal(DMAT, -rowsum(DMAT))
        if side == 'e':
            np.fill_diagonal(DMAT, DMAT.diagonal() - 2.0j*np.pi)
        ne.evaluate('DMAT/cwT', out=DMAT)
        ret = MAT, MAT.dot(DMAT)
    else:
        ret = MAT
    return ret

def compensated_laplace_dlp_preform(source, side):
    method = source.Laplace_Close_Quad
    A1 = Cauchy_Layer_Form(source, source)
    np.fill_diagonal(A1, -rowsum(A1))
    scale = 1.0j/source.N
    MAT = A1 + scale*method.get_differentiation_matrix()
    if side == 'i':
        np.fill_diagonal(MAT, MAT.diagonal()-1)
    return MAT

def compensated_laplace_slp_preform(source, target, side, gradient=False):
    cslp_method = source.Laplace_CSLP_Self_Kress
    method = source.Laplace_Close_Quad
    target_difference = source.get_inside_point() - target.c
    # check if the CSLP Matrix was already generated
    CSLP = cslp_method.Form(side)
    if side == 'e':
        # what gets done before cauchy
        MAT1 = CSLP + method.sawlog[:,None]*(source.weights/(2.0*np.pi))
        MAT2 = method.inf_scale.dot(MAT1)[:,None]
        MAT = MAT1 - MAT2.T
        # what gets done after cauchy
        LA = np.log(np.abs(target_difference))
        AFTER_MAT = -LA[:,None]*(source.weights/(2.0*np.pi))-MAT2.T
    else:
        MAT = CSLP
        AFTER_MAT = np.zeros([target.N, source.N])
    if gradient:
        if side == 'e':
            LD = 1.0/target_difference
            AFTER_DER_MAT = source.weights/(2.0*np.pi)*LD[:,None]
        else:
            AFTER_DER_MAT = np.zeros([target.N, source.N])
        ret = MAT, (AFTER_MAT, AFTER_DER_MAT)
    else:
        ret = MAT, AFTER_MAT
    return ret

def Compensated_Laplace_Apply(source, target, side, tau, do_DLP=False,
        do_SLP=False, gradient=False, main_type='real', gradient_type='real', backend='fly', forstokes=False):
    N = source.N
    M = target.N
    vb = np.zeros(source.N, dtype=complex)
    if do_DLP:
        vb += compensated_laplace_dlp_preapply(source, side, tau, backend=backend)
    if do_SLP:
        Svb, after_adjs = compensated_laplace_slp_preapply(source, target, side,
                                                        tau, gradient=gradient)
        if gradient:
            after_adj = after_adjs[0]
            after_der_adj = after_adjs[1]
        else:
            after_adj = after_adjs
        if forstokes:
            Svb *= 0.5
            after_adj *= 0.5
            if gradient:
                after_der_adj *= 0.5
        vb += Svb
    us = compensated_cauchy_apply(source, target, side, vb, derivative=gradient,
                                                                backend=backend)
    if gradient:
        u = us[0]
        du = us[1]
    else:
        u = us
    if do_SLP:
        u += after_adj
        if gradient:
            du += after_der_adj
    u = u.real if main_type == 'real' else u
    if gradient:
        if gradient_type == 'real':
            ret = (u, du.real, -du.imag)
        else:
            ret = (u, du)
    else:
        ret = u
    return ret

def compensated_cauchy_apply(source, target, side, tau, derivative=False,
                                                                backend='fly'):
    I0 = Cauchy_Layer_Apply(source, target, tau, backend=backend)
    J0 = Cauchy_Layer_Apply(source, target, np.ones(source.N), backend=backend)
    if side == 'e':
        J0 += 1.0
    IJ0 = 1.0/J0
    if derivative:
        D0 = Cauchy_Layer_Apply(source, source, tau, backend=backend)
        D1 = Cauchy_Layer_Apply(source, source, np.ones(source.N), backend=backend)
        taup = 2.0j*np.pi*(D0 - D1*tau)
        if side == 'e':
            taup -= 2.0j*np.pi*tau
        taup /= source.complex_weights
        I0p = Cauchy_Layer_Apply(source, target, taup, backend=backend)
        ret = (I0*IJ0, I0p*IJ0)
    else:
        ret = I0*IJ0
    return ret

def compensated_laplace_dlp_preapply(source, side, tau, backend='fly'):
    # this separation of the barycentric form is okay
    # because it is source-->source ignoring diagonal point
    # no subtractions of points that are arbitrarily close
    # unless you have a nearly self-intersecting curve!
    u1 = Cauchy_Layer_Apply(source, source, tau, backend=backend)
    u2 = Cauchy_Layer_Apply(source, source, np.ones(source.N), backend=backend)
    scale = 1j/source.N
    u = u1 - u2*tau + scale*differentiate(tau)
    if side == 'i':
        u -= tau
    return u

def compensated_laplace_slp_preapply(source, target, side, tau, gradient=False):
    # how to reliably control that this only gets done once if there
    # are multiple apply calls? I.e. to multiple targets?
    # can't just store it and depend on that as in the form
    # because it may be called with different taus
    # check to see if source has this computed, use it if it is!
    method = source.Laplace_Close_Quad
    vb = source.Laplace_CSLP_Self_Kress.Apply(side, tau)
    target_difference = source.get_inside_point() - target.c
    if side == 'e':
        # what gets done before cauchy
        totchgp = np.sum(source.weights*tau)/(2*np.pi)

        vb += totchgp*method.sawlog
        vinf = np.sum(method.inf_scale*vb)
        vb -= vinf

        after_adj = -totchgp*np.log(np.abs(target_difference)) - vinf
    else:
        after_adj = np.zeros(target.N, dtype=complex)
    if gradient:
        if side == 'e':
            after_der_adj = totchgp/target_difference
        else:
            after_der_adj = np.zeros(target.N, dtype=complex)
        ret = vb, (after_adj, after_der_adj)
    else:
        ret = vb, after_adj
    return ret

