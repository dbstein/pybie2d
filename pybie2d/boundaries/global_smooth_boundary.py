import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os
import numba

from ..misc.basic_functions import interpolate_to_p, differentiate, \
                differentiation_matrix, apply_circulant_matrix
from .boundary import Boundary
from ..kernels.laplace import Laplace_Layer_Form, Laplace_Layer_Apply, \
                            Laplace_Layer_Self_Form, Laplace_Layer_Self_Apply
from ..kernels.cauchy import Cauchy_Layer_Form, Cauchy_Layer_Apply
from .. import have_fmm
if have_fmm:
    from .. import FMM

class Global_Smooth_Boundary(Boundary):
    """
    This class impelements a "global smooth boundary" for use in
    Boundary Integral methods

    Instantiation: see documentation to self.__init__()
        note that both compute_quadrature and compute_tree methods
        are called by default in the instantiation
        if you have fairly simple needs, you may be able to set one
        or both of these computations off to save time
    Public Methods:
        all parent methods (see documentation for Boundary class)
        Laplace_Correct_Close
        Stokes_Correct_Close
    """
    def __init__(   self, x=None, y=None, c=None, x_func=None, y_func=None,
                    c_func=None, xp_func=None, yp_func=None, cp_func=None,
                    xpp_func=None, ypp_func=None, cpp_func=None, N=None,
                    inside_point=None, compute_quadrature=True,
                    compute_tree=True, compute_differentiation_matrix=True,
                    self_type='kress'):
        """
        This function initializes the boundary element.
        It also computes, by default:
            quadrature
            kd-tree
            kress matrices for quickly applying kress quadratures
            a differentiation matrix
        These computations can be turned off if they aren't needed

        x (optional): real vector of x-coordinates
        y (optional): real vector of y-coordinates
        c (optional): complex vector with c.real giving x-coordinates
            and c.imag giving y-coordinates
        # the following functions must take a real parameter vector defined
          on [0, 2pi) and return an array of the appropriate type
          * denotes x, y, or c; if x the function returns x values,
          y returns y values, and c returns complex values with x = c.real and
          y = c.imag
        *_func (optional): coordinates
        *p_func (optional): first parameter derivative of coordinates
        *pp_func (optional): second parameter derivative of coordinates
        inside_point (optional): either tuple/list/array (real, real)
            or complex scalar giving coordinates for a point inside s,
            preferably far from the boundary if this is not provided, will be
            computed as (x.mean(), y.mean()), which may not be inside!
        self_type (optional): 'kress'; (only kress implemented for now)
        The user must provide at least one of the following sets of inputs:
            (1) x and y            
                (x and y positions, as real vectors)
            (2) c
                (x and y positions as a complex vector, x=c.real, y=cimag)
            (3) x_func, y_func, N
                (functions returning x, y positions as real vectors given a
                parameter vector on [0,2*pi])
                the parameterization will be constructed to have N elements
            (4) c_func, N
                (function returning x, y positions as a complex vector given a
                paremeter vector on [0,2*pi])
                the parameterization will be constructed to have N elements

        If real and complex functions or positions are provided real values
            will be used
        If values and functions are provided, functions will be used
        Functions for derivatives are only used if functions for coordinates are
            provided and if the functions for derivatives use the same
            (real/complex) type as the functions for coordinates
        If derivative functions are not provided spectral differentiation is
            used to compute the derivatives
        If inside_point is not provided, it will be computed as the mean
            this may not actually be inside!

        As of now, its not clear to me that the Alpert quadrature scheme will
        work if n is odd. For now, I am throwing an error if N is odd
        """
        super(Global_Smooth_Boundary, self).__init__()

        real_functions_available = x_func is not None and y_func is not None
        complex_function_available = c_func is not None
        functions_available = real_functions_available or \
                                complex_function_available
        N_available = N is not None
        use_functions = N_available and functions_available

        self.extras =   {   'compute_quadrature' : compute_quadrature,
                            'compute_tree'       : compute_tree,
                            'compute_differentiation_matrix' : \
                                        compute_differentiation_matrix,
                        }

        if use_functions:
            self.N = N
            self.t, self.h = np.linspace(0.0, 2.0*np.pi, self.N,
                                    endpoint=False, retstep=True)
            if real_functions_available:
                self.x_func = x_func
                self.y_func = y_func
                self.x = self.x_func(self.t)
                self.y = self.y_func(self.t)
                self.c = self.x + 1j*self.y
                self.position_functions = 'Real'
                if xp_func is not None and yp_func is not None:
                    self.xp_func = xp_func
                    self.yp_func = yp_func
                    self.first_derivative_functions = 'Real'
                else:
                    self.first_derivative_functions = 'None'
                if xpp_func is not None and ypp_func is not None:
                    self.xpp_func = xpp_func
                    self.ypp_func = ypp_func
                    self.second_derivative_functions = 'Real'
                else:
                    self.second_derivative_functions = 'None'
            else:
                self.c_func = c_func
                self.c = self.c_func(self.t)
                self.x = self.c.real
                self.y = self.c.imag
                self.position_functions = 'Complex'
                if cp_func is not None:
                    self.cp_func = cp_func
                    self.first_derivative_functions = 'Complex'
                else:
                    self.first_derivative_functions = 'None'
                if cpp_func is not None:
                    self.cpp_func = cpp_func
                    self.second_derivative_functions = 'Complex'
                else:
                    self.second_derivative_functions = 'None'
        elif x is not None and y is not None:
            self.x = x
            self.y = y
            self.c = self.x + 1j*self.y
            self.N = self.x.shape[0]
            self.t, self.h = np.linspace(0.0, 2*np.pi, self.N,
                                endpoint=False, retstep=True)
            self.position_functions = 'None'
            self.first_derivative_functions = 'None'
            self.second_derivative_functions = 'None'
        elif c is not None:
            self.c = c
            self.x = c.real
            self.y = c.imag
            self.N = self.c.shape[0]
            self.t, self.h = np.linspace(0.0, 2*np.pi, self.N,
                                endpoint=False, retstep=True)
            self.first_derivative_functions = 'None'
            self.second_derivative_functions = 'None'
        else:
            raise StandardError('Unacceptable set of parameters provided.')
        if inside_point is not None:
            if type(inside_point) == complex:
                self.inside_point_c = inside_point
                self.inside_point_x = self.inside_point_c.real
                self.inside_point_y = self.inside_point_c.imag
            else:
                self.inside_point_x = inside_point[0]
                self.inside_point_y = inside_point[1]
                self.inside_point_c = self.inside_point_x + \
                                        1j*self.inside_point_y
        else:
            self.inside_point_c = np.mean(self.c)
            self.inside_point_x = self.inside_point_c.real
            self.inside_point_y = self.inside_point_c.imag

        if self.N % 2 != 0:
            raise StandardError('The closed_boundary_element class currently \
                only accepts N being even.')

        self.self_type = self_type

        self.__init2__(compute_quadrature, compute_tree)
    # end __init__ function definition

    #########################
    #### Public Methods ####
    #########################

    def compute_quadrature(self):
        """
        Compute various parameters related to the boundary parameterization
        That will be needed throughout these computations
        This code is based off of quadr.m ((c) Alex Barnett 10/8/14)
        Names have been made to be more readable. Old names are also included.

        Note that 
        """
        self.stacked_boundary = np.column_stack([self.x, self.y])
        self.stacked_boundary_T = self.stacked_boundary.T
        self.k = np.fft.rfftfreq(self.N, self.h/(2.0*np.pi)) # fourier modes
        self.ik = 1j*self.k
        self.ik2 = self.ik**2
        self.xhat = np.fft.rfft(self.x)
        self.yhat = np.fft.rfft(self.y)
        if self.first_derivative_functions == 'Real':
            self.xp = self.xp_func(self.t)
            self.yp = self.yp_func(self.t)
            self.cp = self.xp + 1j*self.yp
        elif self.first_derivative_functions == 'Complex':
            self.cp = self.cp_func(self.t)
            self.xp = self.cp.real
            self.yp = self.cp.imag
        else:
            self.xp = np.fft.irfft(self.xhat*self.ik)
            self.yp = np.fft.irfft(self.yhat*self.ik)
            self.cp = self.xp + 1j*self.yp
        if self.second_derivative_functions == 'Real':
            self.xpp = self.xpp_func(self.t)
            self.ypp = self.ypp_func(self.t)
            self.cpp = self.xpp + 1j*self.ypp
        elif self.second_derivative_functions == 'Complex':
            self.cpp = self.cpp_func(self.t)
            self.xpp = self.cpp.real
            self.ypp = self.cpp.imag
        else:
            self.xpp = np.fft.irfft(self.xhat*self.ik2)
            self.ypp = np.fft.irfft(self.yhat*self.ik2)
            self.cpp = self.xpp + 1j*self.ypp
        self.speed = np.abs(self.cp)
        self.tangent_c = self.cp/self.speed
        self.tangent_x = self.tangent_c.real
        self.tangent_y = self.tangent_c.imag
        self.normal_c = -1.0j*self.tangent_c
        self.normal_x = self.normal_c.real
        self.normal_y = self.normal_c.imag
        self.stacked_normal = np.column_stack([self.normal_x, self.normal_y])
        self.stacked_normal_T = self.stacked_normal.T
        self.tangent_c = 1.0j*self.normal_c
        self.tangent_x = self.tangent_c.real
        self.tangent_y = self.tangent_c.imag
        self.curvature = -(np.conj(self.cpp)*self.normal_c).real/self.speed**2
        self.weights = self.h*self.speed
        self.complex_weights = self.h*self.cp
        self.scaled_cp = self.cp/self.N
        self.scaled_speed = self.speed/self.N
        self.max_h = 2.0*np.pi*np.max(self.scaled_speed)
        self.quadrature_computed = True
        self.area = 2.0*np.pi*np.sum(self.x*self.yp)/self.N
        self.perimeter = 2.0*np.pi*np.sum(self.speed)/self.N
        # for taking dot products
        self.ones_vec = np.ones(self.N, dtype=float)
        # Alex Barnett's old names
        self.sp = self.speed
        self.tang = self.tangent_c
        self.cur = self.curvature
        self.w = self.weights
        self.cw = self.complex_weights
        self.a = self.inside_point_c
        # my old names (to be removed once the code is changed)
        self.ctang = self.tang
        self.xtang = self.tangent_x
        self.ytang = self.tangent_y
        self.cnx = self.normal_c
        self.nx = self.normal_x
        self.ny = self.normal_y
        self.cpn = self.scaled_cp
        self.spn = self.scaled_speed
        # vectors to be used to speed up Kress/CSLP Routines
        v1 = 4.0*np.sin(np.pi*np.arange(self.N)/self.N)**2
        v1[0] = 1.0
        self.Kress_V1 = 0.5*np.log(v1)*self.N/(2*np.pi)
        self.Kress_V1_hat = np.fft.fft(self.Kress_V1)
        v2 = np.abs(np.fft.fftfreq(self.N, 1.0/self.N))
        v2[0] = np.Inf
        IV = 1.0/v2
        self.Kress_V2 = 0.5*np.fft.ifft(IV).real*self.N/(2*np.pi)
        self.Kress_V2_hat = 0.5*IV*self.N/(2*np.pi)
        self.Kress_V_hat = self.Kress_V1_hat/self.N + self.Kress_V2_hat
        self.Kress_V = self.Kress_V1/self.N + self.Kress_V2
        RVe = IV.copy()
        RVe[int(self.N/2)+1:] = 0.0
        RVe[int(self.N/2)] /= 2
        RVi = IV.copy()
        RVi[:int(self.N/2)] = 0.0
        # notice the switch here of i/e things
        # this prevents the transposition/conj in my old code                                   older code)
        self.Kress_VRe = np.fft.ifft(RVi)
        self.Kress_VRi = np.fft.ifft(RVe)
        self.Kress_VRe_hat = RVi
        self.Kress_VRi_hat = RVe
        self.Kress_x = np.cos(self.t) + 1j*np.sin(self.t)
        self.CSLP_limit =  np.log(1j*self.Kress_x/self.cp) * \
                                            self.weights/(2*np.pi)
        self.sawlog = -1j*self.t + np.log(self.inside_point_c - self.c)
        self.sawlog.imag = np.unwrap(self.sawlog.imag)
        self.inf_scale = self.complex_weights/(self.c-self.inside_point_c) \
                                                                / (2.0j*np.pi)
        if self.extras['compute_differentiation_matrix']:
            # some extra matrices used to speed up setting up close evals
            self.differentiation_matrix = differentiation_matrix(self.N)
    # end compute_quadrature function definition

    # self quadrature (apply) for Laplace SLP
    def Laplace_SLP_Self_Apply(self, tau):
        """
        Apply Laplace SLP self-interaction kernel

        Inputs:
            tau,    required, dtype(ns): density (dtype can be float or complex)
        """
        return Laplace_SLP_Self_Kress_Apply(self.x, self.y, self.speed, tau)
    # end Laplace_SLP_Self_Apply function definition

    # self quadrature (form) for Laplace SLP
    def Laplace_SLP_Self_Form(self):
        """
        Form Laplace SLP self-interaction matrix
            (and normal derivative matrix, if requested)
        """
        return Laplace_SLP_Self_Kress_Form(self)
    # end Laplace_SLP_Self_Apply function definition

    ###### Method for generating close corrections
    def tolerance_to_distance(self, tol):
        """
        Given error tolerance, finds distance where close evaluation is needed
        """
        return -np.log(tol)*self.max_h/4.5

    def Get_Close_Correction_Function(self, target, side, do_DLP, DLP_weight,
                                        do_SLP, SLP_weight, kernel, backend):
        """
        Given target, kernel, and type ('preformed', 'fly', 'numba', or 'fmm'),

        Returns a dictionary 'preparation'
        And a function that takes as parameters tau, preparation that
        is repsonsible for the close correction

        Note that this does not check if target points need close evaluation!
        """
        if kernel is 'laplace':
            return Get_Laplace_Close_Correction_Function(self, target, side,
                            do_DLP, DLP_weight, do_SLP, SLP_weight, backend)
        else:
            raise Exception('Close evaluation for the selected kernel has not \
                                                        yet been implemented.')

    #########################
    #### Private Methods ####
    #########################

    def _test_inside_point(self, eps=1e-10):
        """
        Test whether the provided or generated inside point is acceptable
        returns True if the point is okay, False if its not
        """
        if not self.quadrature_computed:
            self.compute_quadrature()
        test_value = np.sum(self.complex_weights/(self.c-self.inside_point_c))
        return np.abs(test_value - 2.0j*np.pi) < eps
    # end _test_inside_point function
##### End of Global_Smooth_Boundary class definition ###########################

################################################################################
#### Kress Self-Evaluation routines for Laplace SLP ############################
################################################################################

def Laplace_SLP_Self_Kress_Form(source):
    # requires the function forming the matrices Kress_C to have been called
    weights = source.weights
    C = sp.linalg.circulant(source.Kress_V)
    A = Laplace_Layer_Self_Form(source, ifcharge=True, self_type='naive')
    np.fill_diagonal(A, -np.log(source.speed)/(2*np.pi)*weights)
    return ne.evaluate('A + C*weights')

def Laplace_SLP_Self_Kress_Apply(source, tau, backend='fly'):
    """
    source, required, global_smooth_boundary, source
    tau,   required, dtype(ns), density

    for now, just form the matrix and apply it
    this should be recoded using numba and FFTs for applying the circulant mats
    """
    weighted_tau = tau*source.weights
    u1 = Laplace_Layer_Self_Apply(source, charge=tau, backend=backend)
    u1 -= np.log(source.speed)/(2*np.pi)*weighted_tau
    u2 = apply_circulant_matrix(weighted_tau, c_hat=source.Kress_V_hat,
                                                                real_it=True)
    return u1 + u2

################################################################################
#### Cauchy Compensated Evaluation                       #######################
#### Functions compatible with Close_Corrector interface #######################
################################################################################

# note that this function is somewhat more complex than it might seem to be
# this is to avoid forming the differentiation matrix, since
# the differentiate function gives better error than applying the
# differentiation matrix itself
def _Laplace_Close_Correction_Function_Preformed(tau, preparation):
    w = np.zeros(tau.shape, dtype=complex)
    if preparation['do_DLP']:
        D = preparation['dlp_pre_mat'].dot(tau)
        D += preparation['dlp_c1']*differentiate(tau)
        D += preparation['dlp_c2']*tau
        w += D
    if preparation['do_SLP']:
        S = preparation['slp_pre_mat'].dot(tau)
        w += S
    v = preparation['cauchy_mat'].dot(w)
    if preparation['do_SLP']:
        v += preparation['slp_post_mat'].dot(tau)
    return v.real - preparation['naive_mat'].dot(tau)

def _Laplace_Close_Correction_Function_Full_Preformed(tau, preparation):
    return preparation['correction_mat'].dot(tau)

def _Laplace_Close_Correction_Function_Fly(tau, preparation):
    v1 = Compensated_Laplace_Apply(
            source     = preparation['source'],
            target     = preparation['target'],
            side       = preparation['side'],
            tau        = tau,
            do_DLP     = preparation['do_DLP'], 
            DLP_weight = preparation['DLP_weight'],
            do_SLP     = preparation['do_SLP'], 
            SLP_weight = preparation['SLP_weight'],
            backend    = preparation['backend']
        )
    ch_adj = 1.0 if preparation['SLP_weight'] is None \
                                    else preparation['SLP_weight']
    ds_adj = 1.0 if preparation['DLP_weight'] is None \
                                    else preparation['DLP_weight']
    ch = tau*ch_adj*int(preparation['do_SLP'])
    ds = tau*ds_adj*int(preparation['do_DLP'])
    v2 = Laplace_Layer_Apply(
            source     = preparation['source'],
            target     = preparation['target'],
            charge     = ch,
            dipstr     = ds,
            backend    = preparation['backend']
        )
    return v1.real - v2

def Get_Laplace_Close_Correction_Function(source, target, side, do_DLP,
                                    DLP_weight, do_SLP, SLP_weight, backend):
    if backend == 'preformed':
        mats = Compensated_Laplace_Form(source, target, side, do_DLP, 
                                                DLP_weight, do_SLP, SLP_weight)
        naive_mat = Laplace_Layer_Form(source, target, ifcharge=do_SLP,
                    chweight=SLP_weight, ifdipole=do_DLP, dpweight=DLP_weight)
        preparation = {
            'do_DLP'       : do_DLP,
            'do_SLP'       : do_SLP,
            'dlp_pre_mat'  : mats[0],
            'dlp_c1'       : mats[1],
            'dlp_c2'       : mats[2],
            'cauchy_mat'   : mats[3],
            'slp_pre_mat'  : mats[4],
            'slp_post_mat' : mats[5],
            'naive_mat'    : naive_mat,
        }
        return preparation, _Laplace_Close_Correction_Function_Preformed
    elif backend == 'full preformed':
        close_mat = Compensated_Laplace_Full_Form(source, target, side, do_DLP, 
                                                DLP_weight, do_SLP, SLP_weight)
        naive_mat = Laplace_Layer_Form(source, target, ifcharge=do_SLP,
                    chweight=SLP_weight, ifdipole=do_DLP, dpweight=DLP_weight)
        correction_mat = close_mat.real - naive_mat
        preparation = {
            'do_DLP'         : do_DLP,
            'do_SLP'         : do_SLP,
            'correction_mat' : correction_mat,
        }
        return preparation, _Laplace_Close_Correction_Function_Full_Preformed
    else:
        preparation = {
            'source'       : source,
            'target'       : target,
            'side'         : side,
            'do_DLP'       : do_DLP,
            'do_SLP'       : do_SLP,
            'DLP_weight'   : DLP_weight,
            'SLP_weight'   : SLP_weight,
            'backend'      : backend,
        }
        return preparation, _Laplace_Close_Correction_Function_Fly

################################################################################
#### Cauchy Compensated Evaluation - User Facing Functions #####################
################################################################################

def Compensated_Laplace_Full_Form(source, target, side, do_DLP=False,
                DLP_weight=None, do_SLP=False, SLP_weight=None, gradient=False):
    N = source.N
    M = target.N
    PM = np.zeros([N, N], dtype=complex)
    if do_DLP:
        DPM = compensated_laplace_dlp_full_preform(source, side)
        if DLP_weight is not None:
            DPM *= DLP_weight
        PM += DPM
    if do_SLP:
        SPM, AFTER_MATS = compensated_laplace_slp_preform(source, target, side)
        if gradient:
            AFTER_MAT = AFTER_MATS[0]
            AFTER_DER_MAT = AFTER_MATS[1]
        else:
            AFTER_MAT = AFTER_MATS
        if SLP_weight is not None:
            SPM *= SLP_weight
            AFTER_MAT *= SLP_weight
            if gradient:
                AFTER_DER_MAT *= SLP_weight
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
    ret = (MAT1, MATD) if gradient else MAT1
    return ret

def Compensated_Laplace_Form(source, target, side, do_DLP=False,
                DLP_weight=None, do_SLP=False, SLP_weight=None):
    N = source.N
    M = target.N
    if do_DLP:
        dlp_pre_mat, dlp_c1, dlp_c2 = \
            compensated_laplace_dlp_preform(source, side)
        if DLP_weight is not None:
            dlp_pre_mat *= DLP_weight
            dlp_c1      *= DLP_weight
            dlp_c2      *= DLP_weight
    else:
        dlp_pre_mat, dlp_c1, dlp_c2 = None, None, None
    if do_SLP:
        slp_pre_mat, slp_post_mat = compensated_laplace_slp_preform(source,
                                                                target, side)
        if SLP_weight is not None:
            slp_pre_mat  *= SLP_weight
            slp_post_mat *= SLP_weight
    else:
        slp_pre_mat, slp_post_mat = None, None
    cauchy_mat = compensated_cauchy_form(source, target, side,
                                                        derivative=False)
    return dlp_pre_mat, dlp_c1, dlp_c2, cauchy_mat, slp_pre_mat, slp_post_mat

def Compensated_Laplace_Apply(source, target, side, tau, do_DLP=False,
                DLP_weight=None, do_SLP=False, SLP_weight=None, backend='fly'):
    N = source.N
    M = target.N
    vb = np.zeros(source.N, dtype=complex)
    if do_DLP:
        Dvb = compensated_laplace_dlp_preapply(source, side, tau,
                                                                backend=backend)
        if DLP_weight is not None:
            Dvb *= DLP_weight
        vb += Dvb
    if do_SLP:
        Svb, after_adj = compensated_laplace_slp_preapply(source, target, side,
                                                                            tau)
        if SLP_weight is not None:
            Svb  *= SLP_weight
            after_adj *= SLP_weight
        vb += Svb            
    u = compensated_cauchy_apply(source, target, side, vb, derivative=False,
                                                                backend=backend)
    if do_SLP:
        u += after_adj
    return u

################################################################################
#### Cauchy Compensated Evaluation - The Ugly Internals ########################
################################################################################

##### CSLP Matrix Form/Apply ###################################################
def Complex_SLP_Kress_Split_Nystrom_Self_Form(source, side):
    weights = source.weights
    speed = source.speed
    scaled_weights = weights/(2*np.pi)
    source_c = source.c
    scT = source_c[:,None]
    kress_x = source.Kress_x
    kress_xT = kress_x[:,None]
    xd = ne.evaluate('kress_xT-kress_x')
    d = ne.evaluate('scT - source_c')
    S = ne.evaluate('log(xd/d)*scaled_weights') # appears to be no phase jumps!
    np.fill_diagonal(S, source.CSLP_limit)
    VR = source.Kress_VRi if side == 'i' else source.Kress_VRe
    R = sp.linalg.circulant(VR)
    S = ne.evaluate('S + R*speed')
    return S

@numba.njit(parallel=True)
def _Complex_SLP_Kress_Split_Nystrom_Self_Apply_numba(s, p, tau, pot):
    for i in range(s.shape[0]):
        pot[i] = 0.0
    for i in numba.prange(s.shape[0]):
        for j in range(i):
            ds = s[i] - s[j]
            dp = p[i] - p[j]
            pot[i] += np.log(dp/ds)*tau[j]
        for j in range(i+1,s.shape[0]):
            ds = s[i] - s[j]
            dp = p[i] - p[j]
            pot[i] += np.log(dp/ds)*tau[j]

# there is no FMM apply of this function for now
# since we don't know how to do the log(dp/ds) with FMM
# or the log(dp)-log(ds) with corrections for phase jumps
def Complex_SLP_Kress_Split_Nystrom_Self_Apply(source, side, tau):
    u1 = np.empty_like(source.c)
    _Complex_SLP_Kress_Split_Nystrom_Self_Apply_numba(source.c, \
                        source.Kress_x, tau*source.weights/(2*np.pi), u1)
    u2 = source.CSLP_limit*tau
    circ = source.Kress_VRi_hat if side == 'i' else source.Kress_VRe_hat
    u3 = apply_circulant_matrix(tau*source.speed, c_hat=circ, real_it=False)
    return u1 + u2 + u3

##### Compensated Cauchy Form/Apply ############################################
def compensated_cauchy_form(source, target, side, derivative=False):
    mysum = lambda x: x.dot(source.ones_vec)
    sc = source.c
    scT = sc[:,None]
    tcT = target.c[:,None]
    cw = source.complex_weights
    cwT = cw[:,None]

    comp = Cauchy_Layer_Form(source, target)
    J0 = mysum(comp)
    if side == 'e':
        # may have to check on this for SLPs!
        J0 += (2.0*np.pi)**2
    prefac = 1.0/J0
    prefacT = prefac[:,None]

    MAT = ne.evaluate('prefacT*comp')

    if derivative:
        # get Schneider-Werner derivative matrix
        DMAT = ne.evaluate('cw/(scT-sc')
        np.fill_diagonal(DMAT, 0.0)
        np.fill_diagonal(DMAT, -mysum(DMAT))
        if side == 'e':
            np.fill_diagonal(DMAT, DMAT.diagonal() - 2.0j*np.pi)
        ne.evaluate('DMAT/cwT', out=DMAT)
        ret = MAT, MAT.dot(DMAT)
    else:
        ret = MAT
    return ret
def compensated_cauchy_apply(source, target, side, tau, derivative=False,
                                                                backend='fly'):
    I0 = Cauchy_Layer_Apply(source, target, tau, backend=backend)
    J0 = Cauchy_Layer_Apply(source, target, source.ones_vec, backend=backend)
    if side == 'e':
        J0 += (2.0*np.pi)**2
    return I0/J0

##### Laplace DLP Prework ######################################################
def compensated_laplace_dlp_full_preform(source, side):
    A1 = Cauchy_Layer_Form(source, source)
    mysum = lambda x: x.dot(source.ones_vec)
    np.fill_diagonal(A1, -mysum(A1))
    scale = 1.0j/source.N
    MAT = A1 + scale*source.differentiation_matrix
    if side == 'i':
        np.fill_diagonal(MAT, MAT.diagonal()-1)
    return MAT

def compensated_laplace_dlp_preform(source, side):
    # to avoid the differentiation matrix...
    A1 = Cauchy_Layer_Form(source, source)
    mysum = lambda x: x.dot(source.ones_vec)
    np.fill_diagonal(A1, -mysum(A1))

    M1 = A1
    C1 = 1.0j/source.N
    C2 = -1.0 if side == 'i' else 0.0
    return M1, C1, C2

def compensated_laplace_dlp_preapply(source, side, tau, backend='fly'):
    u1 = Cauchy_Layer_Apply(source, source, tau, backend=backend)
    u2 = Cauchy_Layer_Apply(source, source, source.ones_vec, backend=backend)
    scale = 1j/source.N
    u = u1 - u2*tau + scale*differentiate(tau)
    if side == 'i':
        u -= tau
    return u

##### Laplace SLP Prework ######################################################
def compensated_laplace_slp_preform(source, target, side):
    # check if the CSLP Matrix was already generated
    if not hasattr(source, 'CSLP'):
        source.CSLP = Complex_SLP_Kress_Split_Nystrom_Self_Form(source, side)
    if side == 'e':
        # what gets done before cauchy
        MAT1 = source.CSLP + source.sawlog[:,None]*(source.weights/(2.0*np.pi))
        MAT2 = source.inf_scale.dot(MAT1)[:,None]
        MAT = MAT1 - MAT2.T
        # what gets done after cauchy
        LA = np.log(np.abs(source.inside_point_c-target.c))
        AFTER_MAT = -LA[:,None]*(source.weights/(2.0*np.pi))-MAT2.T
    else:
        MAT = CSLP
        AFTER_MAT = np.zeros([target.N, source.N])
    return MAT, AFTER_MAT
# only one version (no FMM version...)
def compensated_laplace_slp_preapply(source, target, side, tau):
    # how to reliably control that this only gets done once if there
    # are multiple apply calls? I.e. to multiple targets?
    # can't just store it and depend on that as in the form
    # because it may be called with different taus
    vb = Complex_SLP_Kress_Split_Nystrom_Self_Apply(source, side, tau)
    if side == 'e':
        # what gets done before cauchy
        totchgp = np.sum(source.weights*tau)/(2*np.pi)

        vb += totchgp*source.sawlog
        vinf = np.sum(source.inf_scale*vb)
        vb -= vinf

        after_adj = -totchgp*np.log(np.abs(source.inside_point_c-target.c)) - \
                                                                        vinf
    else:
        after_adj = np.zeros(trg.N, dtype=complex)
    return vb, after_adj

