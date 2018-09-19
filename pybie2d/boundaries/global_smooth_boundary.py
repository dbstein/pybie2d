import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os
import numba

from ..misc.basic_functions import interpolate_to_p, differentiate, \
                differentiation_matrix, apply_circulant_matrix, rowsum
from .boundary import Boundary
from ..kernels.high_level.laplace import Laplace_Layer_Form, Laplace_Layer_Apply
from ..kernels.high_level.stokes import Stokes_Layer_Form, Stokes_Layer_Apply
from ..kernels.high_level.cauchy import Cauchy_Layer_Form, Cauchy_Layer_Apply
from .. import have_fmm
if have_fmm:
    from .. import FMM

# decorater for any functions that require Kress Precomputations
def do_Kress_Preparations(func):
    def inner_function(self, *args, **kwargs):
        self.Kress_Preparations()
        return func(self, *args, **kwargs)
    return inner_function

# decorater for any functions that require CSLP/Close Eval Precomputations
def do_Close_Preparations(func):
    def inner_function(self, *args, **kwargs):
        self.Close_Preparations()
        return func(self, *args, **kwargs)
    return inner_function

class Global_Smooth_Boundary(Boundary):
    """
    This class impelements a "global smooth boundary" for use in
    Boundary Integral methods

    Instantiation: see documentation to self.__init__()
        the compute_quadrature method is called by default
    """
    def __init__(self, x=None, y=None, c=None):
        """
        This function initializes the boundary element.

        x (optional): real vector of x-coordinates
        y (optional): real vector of y-coordinates
        c (optional): complex vector with c.real giving x-coordinates
            and c.imag giving y-coordinates
        The user must provide at least one of the following sets of inputs:
            (1) x and y            
                (x and y positions, as real vectors)
            (2) c
                (x and y positions as a complex vector, x=c.real, y=cimag)
        If inside_point is not provided, it will be computed as the mean
            this may not actually be inside!

        As of now, its not clear to me that everything will
        work if n is odd. For now, I will throw an error if x/y/c have an
        odd number of elements in them
        """
        super(Global_Smooth_Boundary, self).__init__(x, y, c)
        if self.N % 2 != 0:
            raise Exception('The Global_Smooth_Boundary class only accepts \
                                                                    even N.')
        self.t, self.dt = np.linspace(0, 2*np.pi, self.N, \
                                                endpoint=False, retstep=True)
        self.k = np.fft.fftfreq(self.N, self.dt/(2.0*np.pi)) # fourier modes
        self.ik = 1j*self.k
        self.chat = np.fft.fft(self.c)
        self.cp = np.fft.ifft(self.chat*self.ik)
        self.cpp = np.fft.ifft(self.chat*self.ik**2)
        self.speed = np.abs(self.cp)
        self.tangent_c = self.cp/self.speed
        self.tangent_x = self.tangent_c.real
        self.tangent_y = self.tangent_c.imag
        self.normal_c = -1.0j*self.tangent_c
        self.normal_x = self.normal_c.real
        self.normal_y = self.normal_c.imag
        self.curvature = -(np.conj(self.cpp)*self.normal_c).real/self.speed**2
        self.weights = self.dt*self.speed
        self.complex_weights = self.dt*self.cp
        self.scaled_cp = self.cp/self.N
        self.scaled_speed = self.speed/self.N
        self.max_h = np.max(self.weights)
        self.area = self.dt*np.sum(self.x*self.cp.imag)
        self.perimeter = self.dt*np.sum(self.speed)
    # end __init__ function definition

    """
    Decorations
    """

    def Kress_Preparations(self):
        """
        Constructs vectors used to speed up Kress Routines
        """
        if not hasattr(self, 'Kress_Vectors_Formed'):
            v1 = 4.0*np.sin(np.pi*np.arange(self.N)/self.N)**2
            v1[0] = 1.0
            self.Kress_V1 = 0.5*np.log(v1)/self.dt
            self.Kress_V1_hat = np.fft.fft(self.Kress_V1)
            v2 = np.abs(self.k)
            v2[0] = np.Inf
            self.Kress_IV = 1.0/v2
            self.Kress_V2 = 0.5*np.fft.ifft(self.Kress_IV).real/self.dt
            self.Kress_V2_hat = 0.5*self.Kress_IV/self.dt
            self.Kress_V_hat = self.Kress_V1_hat/self.N + self.Kress_V2_hat
            self.Kress_V = self.Kress_V1/self.N + self.Kress_V2
            self.Kress_Vectors_Formed = True

    def Close_Preparations(self):
        """
        Constructs vectors used to speed up CSLP/Close Routines
        """
        if not hasattr(self, 'Close_Vectors_Formed'):
            self.Kress_Preparations()
            RVe = self.Kress_IV.copy()
            RVe[int(self.N/2)+1:] = 0.0
            RVe[int(self.N/2)] /= 2
            RVi = self.Kress_IV.copy()
            RVi[:int(self.N/2)] = 0.0
            self.Kress_VRe = np.fft.ifft(RVi)
            self.Kress_VRi = np.fft.ifft(RVe)
            self.Kress_VRe_hat = RVi
            self.Kress_VRi_hat = RVe
            self.Kress_x = (np.cos(self.t) + 1j*np.sin(self.t))
            self.CSLP_limit =  np.log(1j*self.Kress_x/self.cp) * \
                                                self.weights/(2*np.pi)
            self.sawlog = -1j*self.t + np.log(self.get_inside_point() - self.c)
            self.sawlog.imag = np.unwrap(self.sawlog.imag)
            self.inf_scale = self.complex_weights/(self.c-self.get_inside_point()) \
                                                                    / (2.0j*np.pi)
            self.NF = int(np.ceil(2.2*self.N)/2.0)*2
            self.sfc = sp.signal.resample(self.c, self.NF)
            self.fsrc = Global_Smooth_Boundary(c=self.sfc)
            self.Close_Vectors_Formed = True

    def get_differentiation_matrix(self):
        if not hasattr(self, 'differentiation_matrix'):
            self.differentiation_matrix = differentiation_matrix(self.N)
        return self.differentiation_matrix

    def set_inside_point(self, c):
        """
        Set an inside point, used in close evaluation schemes
        If self-eval schemes are called before this is set, this will be
        computed as the average of the boundary nodes, which may not be inside
        c should be an imaginary float, with c.real=x, c.imag=y
        """
        good = self._test_inside_point(c)
        if not good:
            warnings.warn('Inside point failed basic test, is it actually inside?')
        self.inside_point_c = c

    def get_inside_point(self):
        if not hasattr(self, 'inside_point_c'):
            candidate = np.sum(self.c)/self.N
            good = self._test_inside_point(candidate)
            if not good:
                warnings.warn('Inside point computed as mean, failed basic test, is it actually inside?')
            self.inside_point_c = candidate
        return self.inside_point_c

    #########################
    #### Public Methods ####
    #########################
    # self quadrature (apply) for Laplace SLP
    def Laplace_SLP_Self_Apply(self, tau, backend=None):
        """
        Apply Laplace SLP self-interaction kernel

        Inputs:
            tau,    required, dtype(ns): density (dtype can be float or complex)
        """
        return Laplace_SLP_Self_Kress_Apply(self, tau, backend)
    # end Laplace_SLP_Self_Apply function definition

    # self quadrature (form) for Laplace SLP
    def Laplace_SLP_Self_Form(self):
        """
        Form Laplace SLP self-interaction matrix
            (and normal derivative matrix, if requested)
        """
        return Laplace_SLP_Self_Kress_Form(self)
    # end Laplace_SLP_Self_Apply function definition

    # self quadrature (apply) for Stokes SLP
    def Stokes_SLP_Self_Apply(self, tau, backend=None):
        """
        Apply Stokes SLP self-interaction kernel

        Inputs:
            tau,    required, dtype(ns): density (dtype can be float or complex)
        """
        return Stokes_SLP_Self_Kress_Apply(self, tau, backend)
    # end Laplace_SLP_Self_Apply function definition

    # self quadrature (form) for Laplace SLP
    def Stokes_SLP_Self_Form(self):
        """
        Form Stokes SLP self-interaction matrix
            (and normal derivative matrix, if requested)
        """
        return Stokes_SLP_Self_Kress_Form(self)
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
        if kernel is 'stokes':
            return Get_Stokes_Close_Correction_Function(self, target, side,
                            do_DLP, DLP_weight, do_SLP, SLP_weight, backend)
        else:
            raise Exception('Close evaluation for the selected kernel has not \
                                                        yet been implemented.')

    #########################
    #### Private Methods ####
    #########################

    def _test_inside_point(self, candidate, eps=1e-10):
        """
        Test whether the provided or generated inside point is acceptable
        returns True if the point is okay, False if its not
        """
        test_value = np.sum(self.complex_weights/(self.c-candidate))
        return np.abs(test_value - 2.0j*np.pi) < eps
    # end _test_inside_point function
##### End of Global_Smooth_Boundary class definition ###########################

################################################################################
#### Kress Self-Evaluation routines for Laplace SLP ############################
################################################################################

@do_Kress_Preparations
def Laplace_SLP_Self_Kress_Form(source, store=True):
    if not hasattr(source, 'Laplace_Self_Kress_Matrix'):
        weights = source.weights
        C = sp.linalg.circulant(source.Kress_V)
        A = Laplace_Layer_Form(source, ifcharge=True)
        np.fill_diagonal(A, -np.log(source.speed)/(2*np.pi)*weights)
        Laplace_Self_Kress_Matrix = ne.evaluate('A + C*weights')
        if store:
           source.Laplace_Self_Kress_Matrix = Laplace_Self_Kress_Matrix 
        return Laplace_Self_Kress_Matrix 
    else:
        return source.Laplace_Self_Kress_Matrix 

@do_Kress_Preparations
def Laplace_SLP_Self_Kress_Apply(source, tau, backend='fly'):
    """
    source, required, global_smooth_boundary, source
    tau,   required, dtype(ns), density
    """
    weighted_tau = tau*source.weights
    u1 = Laplace_Layer_Apply(source, charge=tau, backend=backend)
    u1 -= np.log(source.speed)/(2*np.pi)*weighted_tau
    u2 = apply_circulant_matrix(weighted_tau, c_hat=source.Kress_V_hat,
                                                                real_it=True)
    return u1 + u2

################################################################################
#### Kress Self-Evaluation routines for Stokes  SLP ############################
################################################################################

def Stokes_SLP_Self_Kress_Form(source, store=True):
    if not hasattr(source, 'Stokes_Self_Kress_Matrix'):
        mu = 1.0
        sx = source.x
        sxT = sx[:,None]
        sy = source.y
        syT = sy[:,None]
        dx = ne.evaluate('sxT-sx')
        dy = ne.evaluate('syT-sy')
        irr = ne.evaluate('1.0/(dx*dx + dy*dy)')
        weights = source.weights/(4.0*np.pi*mu)
        A = np.empty((2*source.N,2*source.N), dtype=float)
        A00 = ne.evaluate('weights*dx*dx*irr', out=A[:source.N,:source.N])
        A01 = ne.evaluate('weights*dx*dy*irr', out=A[:source.N,source.N:])
        A10 = ne.evaluate('A01', out=A[source.N:,:source.N])
        A11 = ne.evaluate('weights*dy*dy*irr', out=A[source.N:,source.N:])
        tx = source.tangent_x
        ty = source.tangent_y
        np.fill_diagonal(A00, ne.evaluate('weights*tx*tx'))
        np.fill_diagonal(A01, ne.evaluate('weights*tx*ty'))
        np.fill_diagonal(A10, ne.evaluate('weights*tx*ty'))
        np.fill_diagonal(A11, ne.evaluate('weights*ty*ty'))
        S = Laplace_SLP_Self_Kress_Form(source, store)
        muS = ne.evaluate('0.5*mu*S')
        A00 = ne.evaluate('A00 + muS', out=A[:source.N,:source.N])
        A11 = ne.evaluate('A11 + muS', out=A[source.N:,source.N:])
        if store:
            source.Stokes_Self_Kress_Matrix = A
        return A
    else:
        return source.Stokes_Self_Kress_Matrix

def Stokes_SLP_Self_Kress_Apply(source, tau, backend='fly'):
    """
    source, required, global_smooth_boundary, source
    tau,   required, dtype(2*ns), density
    """
    mu = 1.0
    u1 = Stokes_Layer_Apply(source, charge=tau, backend=backend)
    diag1 = source.weights*source.tangent_x*source.tangent_x*tau[:source.N] + \
                source.weights*source.tangent_x*source.tangent_y*tau[source.N:]
    diag2 = source.weights*source.tangent_x*source.tangent_y*tau[:source.N] + \
                source.weights*source.tangent_y*source.tangent_y*tau[source.N:]
    u2 = np.concatenate([diag1, diag2])
    S1 = Laplace_SLP_Self_Kress_Apply(tau[:source.N])
    S2 = Laplace_SLP_Self_Kress_Apply(tau[source.N:])
    u3 = np.concatenate([S1, S2])
    return u1 + u2 + 0.5*mu*u3

################################################################################
#### Cauchy Compensated Evaluation                       #######################
#### Functions compatible with Close_Corrector interface #######################
################################################################################

# note that this function is somewhat more complex than it might seem to be
# this is to avoid forming the differentiation matrix, since
# the differentiate function gives better error than applying the
# differentiation matrix itself
# Also it avoids an expensive MAT-MAT in the formation stage
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

def _Stokes_Close_Correction_Function_Full_Preformed(tau, preparation):
    return preparation['correction_mat'].dot(tau)

def _Stokes_Close_Correction_Function_Apply(tau, preparation):
    v1 = Compensated_Stokes_Apply(
            preparation['source'],
            preparation['target'],
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
    v2 = Stokes_Layer_Apply(
            source     = preparation['source'],
            target     = preparation['target'],
            forces     = ch,
            dipstr     = ds,
            backend    = preparation['backend']
        )
    return v1 - v2

def Get_Stokes_Close_Correction_Function(source, target, side, do_DLP,
                                    DLP_weight, do_SLP, SLP_weight, backend):
    if backend == 'full preformed':
        close_mat = Compensated_Stokes_Full_Form(source, target, side, do_DLP, 
                                                DLP_weight, do_SLP, SLP_weight)
        naive_mat = Stokes_Layer_Form(source, target, ifforce=do_SLP,
                    fweight=SLP_weight, ifdipole=do_DLP, dpweight=DLP_weight)
        correction_mat = close_mat.real - naive_mat
        preparation = {
            'do_DLP'         : do_DLP,
            'do_SLP'         : do_SLP,
            'correction_mat' : correction_mat,
        }
        return preparation, _Stokes_Close_Correction_Function_Full_Preformed
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
        return preparation, _Stokes_Close_Correction_Function_Apply

################################################################################
#### Cauchy Compensated Evaluation - User Facing Functions #####################
################################################################################

@do_Close_Preparations
def Compensated_Stokes_Full_Form(source, target, side, do_DLP=False,
                            DLP_weight=None, do_SLP=False, SLP_weight=None):
    # adjust the SLP/DLP weights for calls to Laplace functions
    DLP_weight = 1.0 if DLP_weight is None else 1.0*DLP_weight
    SLP_weight = 0.5 if SLP_weight is None else 0.5*SLP_weight
    # arrays that manipulate form of density
    # there's almost certainly a faster way to deal with some of these things
    sh = (source.N, source.N)
    Mx1 = np.zeros(sh)
    np.fill_diagonal(Mx1, source.x)
    My1 = np.zeros(sh)
    np.fill_diagonal(My1, source.y)
    Mxy = np.array(np.bmat([Mx1, My1]))
    Mx = np.array(np.bmat([np.eye(source.N), np.zeros(sh)]))
    My = np.array(np.bmat([np.zeros(sh), np.eye(source.N)]))
    # get laplace compensated matrices
    _DL, DLG = Compensated_Laplace_Full_Form(source=source, target=target,
                    side=side, do_DLP=do_DLP, DLP_weight=DLP_weight,
                    do_SLP=do_SLP, SLP_weight=SLP_weight, gradient=True,
                    main_type='real', gradient_type='complex')
    if do_DLP:
        NF = source.NF
        fsrc = source.fsrc
        # construct resampling matrix
        RS = sp.signal.resample(np.eye(source.N), NF)
        # array that takes 2*src.N real density to resampled complex density
        Mc = Mx + 1j*My
        RSMc = RS.dot(Mc)
        FL = Compensated_Laplace_Full_Form(source=fsrc, target=target,
                side=side, do_DLP=do_DLP, DLP_weight=DLP_weight,
                main_type='complex')
    if do_SLP:
        if do_DLP:
            DL = Compensated_Laplace_Full_Form(source=source, target=target,
                    side=side, do_SLP=do_SLP, SLP_weight=SLP_weight,
                    main_type='real')
        else:
            DL = _DL
    # Step 1
    M1 = np.zeros([target.N, 2*source.N], dtype=complex)
    if do_DLP:
        IX = (fsrc.normal_x/fsrc.normal_c)[:,None]*RSMc
        IY = (fsrc.normal_y/fsrc.normal_c)[:,None]*RSMc
        M1 += FL.dot(IX).real + 1j*FL.dot(IY).real
    if do_SLP:
        M1 += DL.dot(Mc)
    # Step 2
    M2 = DLG.dot(Mxy)
    # Step 3 and 4
    M3 = target.x[:,None]*DLG.dot(Mx)
    M4 = target.y[:,None]*DLG.dot(My)
    # add these up
    MM = M1 + np.conj(M2 - M3 - M4)
    # construct matrix of the right size
    MAT = np.zeros([2*target.N, 2*source.N], dtype=float)
    MAT[:target.N, :] = MM.real
    MAT[target.N:, :] = MM.imag
    return MAT

@do_Close_Preparations
def Compensated_Stokes_Apply(source, target, side, tau, do_DLP=False,
                DLP_weight=None, do_SLP=False, SLP_weight=None, backend='fly'):
    # adjust the SLP/DLP weights for calls to Laplace functions
    DLP_weight = 1.0 if DLP_weight is None else 1.0*DLP_weight
    SLP_weight = 0.5 if SLP_weight is None else 0.5*SLP_weight
    NF = source.NF
    fsrc = source.fsrc
    taux = tau[:source.N]
    tauy = tau[source.N:]
    tauc = taux + 1j*tauy
    ftauc = sp.signal.resample(tauc, NF)
    # step 1
    u1 = np.zeros(target.N, dtype=complex)
    if do_DLP:
        IX = (fsrc.normal_x/fsrc.normal_c)*ftauc
        IY = (fsrc.normal_y/fsrc.normal_c)*ftauc
        u1a = Compensated_Laplace_Apply(fsrc, target, side, IX, do_DLP=True, DLP_weight=DLP_weight, backend=backend, main_type='complex')
        u1b = Compensated_Laplace_Apply(fsrc, target, side, IY, do_DLP=True, DLP_weight=DLP_weight, backend=backend, main_type='complex')
        u1 += u1a.real + 1j*u1b.real
    if do_SLP:
        u1a = Compensated_Laplace_Apply(source, target, side, taux, do_SLP=True, SLP_weight=SLP_weight, backend=backend)
        u1b = Compensated_Laplace_Apply(source, target, side, tauy, do_SLP=True, SLP_weight=SLP_weight, backend=backend)
        u1 += u1a + 1j*u1b
    # step 2
    tauh = source.x*taux + source.y*tauy
    _, u2 = Compensated_Laplace_Apply(source, target, side, tauh, do_SLP=do_SLP, do_DLP=do_DLP, SLP_weight=SLP_weight, DLP_weight=DLP_weight, gradient=True, gradient_type='complex', backend=backend)
    # step 3 and 4
    _, u3 = Compensated_Laplace_Apply(source, target, side, taux, do_SLP=do_SLP, do_DLP=do_DLP, SLP_weight=SLP_weight, DLP_weight=DLP_weight, gradient=True, gradient_type='complex', backend=backend)
    _, u4 = Compensated_Laplace_Apply(source, target, side, tauy, do_SLP=do_SLP, do_DLP=do_DLP, SLP_weight=SLP_weight, DLP_weight=DLP_weight, gradient=True, gradient_type='complex', backend=backend)
    u3 *= target.x
    u4 *= target.y
    # add everything up
    u = u1 + np.conj(u2 - u3 - u4)
    return np.concatenate([u.real, u.imag])

def Compensated_Laplace_Full_Form(source, target, side, do_DLP=False,
                DLP_weight=None, do_SLP=False, SLP_weight=None, gradient=False,
                main_type='real', gradient_type='real'):
    """
    Full Formation of Close-Eval Matrix for Laplace Problem

    Parameters:
    source     (required): Boundary, source
    target     (required): PointSet, target
    side       (required): 'i' or 'e' for interior/exterior evaluation
    do_DLP     (optional): whether to include DLP evaluation
    DLP_weight (optional): scalar weight to apply to the DLP evaluation
    do_SLP     (optional): whether to include SLP evaluation
    SLP_weight (optional): scalar weight to apply to the SLP evaluation
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
        DPM = compensated_laplace_dlp_full_preform(source, side)
        if DLP_weight is not None:
            DPM *= DLP_weight
        PM += DPM
    if do_SLP:
        SPM, AFTER_MATS = compensated_laplace_slp_preform(source, target, side,
                                                            gradient=gradient)
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
    MAT = MAT1.real if main_type == 'real' else MAT1
    if gradient:
        if gradient_type == 'real':
            ret = (MAT, MATD.real, -MATD.imag)
        else:
            ret = (MAT, MATD)
    else:
        ret = MAT
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
        DLP_weight=None, do_SLP=False, SLP_weight=None, gradient=False,
        main_type='real', gradient_type='real', backend='fly'):
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
        Svb, after_adjs = compensated_laplace_slp_preapply(source, target, side,
                                                        tau, gradient=gradient)
        if gradient:
            after_adj = after_adjs[0]
            after_der_adj = after_adjs[1]
        else:
            after_adj = after_adjs
        if SLP_weight is not None:
            Svb *= SLP_weight
            after_adj *= SLP_weight
            if gradient:
                after_der_adj *= SLP_weight
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

################################################################################
#### Cauchy Compensated Evaluation - The Ugly Internals ########################
################################################################################

##### CSLP Matrix Form/Apply ###################################################
@do_Close_Preparations
def Complex_SLP_Kress_Split_Nystrom_Self_Form(source, side, store=True):
    if not hasattr(source, 'CSLP_Base'):
        weights = source.weights
        speed = source.speed
        scaled_weights = weights/(2*np.pi)
        source_c = source.c
        scT = source_c[:,None]
        kress_x = source.Kress_x
        kress_xT = kress_x[:,None]
        xd = ne.evaluate('kress_xT-kress_x')
        d = ne.evaluate('scT - source_c')
        S = ne.evaluate('log(xd/d)') # appears to be no phase jumps!
            # this fails when x is not star-shaped?!
        SI = S.imag
        np.fill_diagonal(SI, np.concatenate([ SI.diagonal(1), (SI[-1,0],) ]) )
        SI = np.unwrap(SI)
        S = ne.evaluate('S*scaled_weights', out=S)
        np.fill_diagonal(S, source.CSLP_limit)
        if store:
            source.CSLP_Base = S
    else:
        S = source.CSLP_Base
    attr = 'CSLP_' + side
    if not hasattr(source, attr):
        RV = source.Kress_VRi if side == 'i' else source.Kress_VRe
        R = sp.linalg.circulant(RV)
        SR = ne.evaluate('S + R*speed')
        if store:
            setattr(source, attr, SR)
    else:
        SR = getattr(source, attr)
    return SR

@numba.njit(parallel=True)
def _Complex_SLP_Kress_Split_Nystrom_Self_Apply_numba(s, p, tau, diag, pot):
    # CSLP apply
    # this function correctly handles phase jumps in the imaginary portion
    # attempt to make this a bit cleaner and faster
    # it is a little faster than the other one
    pot[:] = 0.0
    for i in numba.prange(s.shape[0]):
        # evaluate the kernel for this target point
        temp1 = np.zeros(s.shape[0])
        temp2 = np.zeros(s.shape[0])
        for j in range(i):
            ds = s[i] - s[j]
            dp = p[i] - p[j]
            a = np.log(dp/ds)
            temp1[j] = a.real
            temp2[j] = a.imag
        temp1[i] = diag[i].real
        temp2[i] = diag[i].imag
        for j in range(i+1,s.shape[0]):
            ds = s[i] - s[j]
            dp = p[i] - p[j]
            a = np.log(dp/ds)
            temp1[j] = a.real
            temp2[j] = a.imag
        # fix phase jumps in this
        for j in range(s.shape[0]-1):
            if temp2[j+1] - temp2[j] > np.pi:
                temp2[j+1] -= 2*np.pi
            elif temp2[j] - temp2[j+1] > np.pi:
                temp2[j+1] += 2*np.pi
        # now evaluate the kernel
        for j in range(s.shape[0]):
            pot[i] += (temp1[j]+1j*temp2[j])*tau[j]

@numba.njit(parallel=True)
def _Complex_SLP_Kress_Split_Nystrom_Self_Apply_numba_(s, p, tau, diag, pot):
    # CSLP apply
    # this function correctly handles phase jumps in the imaginary portion
    # this can probably be made faster by (for each i) computing the vector
    # of log(dp/ds) first, looping through it to kill phase jumps, and then
    # applying that fixed vector
    pot[:] = 0.0
    for i in numba.prange(s.shape[0]):
        phasejump = 0.0
        if i == 0:
            prior = diag[i]
            pot[i] += prior*tau[i]
            for j in range(1,s.shape[0]):
                ds = s[i] - s[j]
                dp = p[i] - p[j]
                new = np.log(dp/ds)
                if new.imag-prior.imag > np.pi:
                    phasejump = phasejump - 2*np.pi
                elif prior.imag-new.imag > np.pi:
                    phasejump = phasejump + 2*np.pi
                pot[i] += (new + 1j*phasejump)*tau[j]
                prior = new
        else:
            ds = s[i] - s[0]
            dp = p[i] - p[0]
            prior = np.log(dp/ds)
            pot[i] += prior*tau[0]
            for j in range(1,i):
                ds = s[i] - s[j]
                dp = p[i] - p[j]
                new = np.log(dp/ds)
                if new.imag-prior.imag > np.pi:
                    phasejump = phasejump - 2*np.pi
                elif prior.imag-new.imag > np.pi:
                    phasejump = phasejump + 2*np.pi
                pot[i] += (new + 1j*phasejump)*tau[j]
                prior = new
            new = diag[i]
            if new.imag-prior.imag > np.pi:
                phasejump = phasejump - 2*np.pi
            elif prior.imag-new.imag > np.pi:
                phasejump = phasejump + 2*np.pi
            pot[i] += (new + 1j*phasejump)*tau[i]
            prior = new
            for j in range(i+1,s.shape[0]):
                ds = s[i] - s[j]
                dp = p[i] - p[j]
                new = np.log(dp/ds)
                if new.imag-prior.imag > np.pi:
                    phasejump = phasejump - 2*np.pi
                elif prior.imag-new.imag > np.pi:
                    phasejump = phasejump + 2*np.pi
                pot[i] += (new + 1j*phasejump)*tau[j]
                prior = new

@numba.njit(parallel=True)
def _Complex_SLP_Kress_Split_Nystrom_Self_Apply_numba_old(s, p, tau, diag, pot):
    # CSLP apply
    # this function does not correctly handle phase jumps
    for i in range(s.shape[0]):
        pot[i] = 0.0
    for i in numba.prange(s.shape[0]):
        for j in range(i):
            ds = s[i] - s[j]
            dp = p[i] - p[j]
            pot[i] += np.log(dp/ds)*tau[j]
        pot[i] += diag[i]*tau[i]
        for j in range(i+1,s.shape[0]):
            ds = s[i] - s[j]
            dp = p[i] - p[j]
            pot[i] += np.log(dp/ds)*tau[j]

# there is no FMM apply of this function for now
# since we don't know how to do the log(dp/ds) with FMM
# or the log(dp)-log(ds) with corrections for phase jumps
# def Complex_SLP_Kress_Split_Nystrom_Self_Apply(source, side, tau):
#     u1 = np.empty_like(source.c)
#     _Complex_SLP_Kress_Split_Nystrom_Self_Apply_numba(source.c, \
#                         source.Kress_x, tau*source.weights/(2*np.pi), u1)
#     u2 = source.CSLP_limit*tau
#     circ = source.Kress_VRi_hat if side == 'i' else source.Kress_VRe_hat
#     u3 = apply_circulant_matrix(tau*source.speed, c_hat=circ, real_it=False)
#     return u1 + u2 + u3

@do_Close_Preparations
def Complex_SLP_Kress_Split_Nystrom_Self_Apply(source, side, tau):
    u1 = np.empty_like(source.c)
    _Complex_SLP_Kress_Split_Nystrom_Self_Apply_numba(source.c, \
                        source.Kress_x, tau*source.weights/(2*np.pi),
                        source.CSLP_limit*(2*np.pi)/source.weights, u1)
    circ = source.Kress_VRi_hat if side == 'i' else source.Kress_VRe_hat
    u2 = apply_circulant_matrix(tau*source.speed, c_hat=circ, real_it=False)
    return u1 + u2

##### Compensated Cauchy Form/Apply ############################################
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

def compensated_cauchy_apply_barycentric(source, target, side, tau,
                                                            derivative=False):
    pass

##### Laplace DLP Prework ######################################################
def compensated_laplace_dlp_full_preform(source, side):
    A1 = Cauchy_Layer_Form(source, source)
    np.fill_diagonal(A1, -rowsum(A1))
    scale = 1.0j/source.N
    MAT = A1 + scale*source.get_differentiation_matrix()
    if side == 'i':
        np.fill_diagonal(MAT, MAT.diagonal()-1)
    return MAT

def compensated_laplace_dlp_preform(source, side):
    # to avoid the differentiation matrix...
    A1 = Cauchy_Layer_Form(source, source)
    np.fill_diagonal(A1, -rowsum(A1))

    M1 = A1
    C1 = 1.0j/source.N
    C2 = -1.0 if side == 'i' else 0.0
    return M1, C1, C2

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

##### Laplace SLP Prework ######################################################
@do_Close_Preparations
def compensated_laplace_slp_preform(source, target, side, gradient=False):
    # check if the CSLP Matrix was already generated
    CSLP = Complex_SLP_Kress_Split_Nystrom_Self_Form(source, side)
    if side == 'e':
        # what gets done before cauchy
        MAT1 = CSLP + source.sawlog[:,None]*(source.weights/(2.0*np.pi))
        MAT2 = source.inf_scale.dot(MAT1)[:,None]
        MAT = MAT1 - MAT2.T
        # what gets done after cauchy
        LA = np.log(np.abs(source.get_inside_point()-target.c))
        AFTER_MAT = -LA[:,None]*(source.weights/(2.0*np.pi))-MAT2.T
    else:
        MAT = CSLP
        AFTER_MAT = np.zeros([target.N, source.N]), np.zeros([target.N, source.N])
    if gradient:
        if side == 'e':
            LD = 1.0/(source.get_inside_point()-target.c)
            AFTER_DER_MAT = source.weights/(2.0*np.pi)*LD[:,None]
        else:
            AFTER_DER_MAT = np.zeros([target.M, soure.N])
        ret = MAT, (AFTER_MAT, AFTER_DER_MAT)
    else:
        ret = MAT, AFTER_MAT
    return ret
# only one version (no FMM version...)
@do_Close_Preparations
def compensated_laplace_slp_preapply(source, target, side, tau, gradient=False):
    # how to reliably control that this only gets done once if there
    # are multiple apply calls? I.e. to multiple targets?
    # can't just store it and depend on that as in the form
    # because it may be called with different taus
    # check to see if source has this computed, use it if it is!
    attr = 'CSLP_' + side
    if hasattr(source, attr):
        CSLP = getattr(source, attr)
        vb = CSLP.dot(tau)
    else:
        vb = Complex_SLP_Kress_Split_Nystrom_Self_Apply(source, side, tau)
    if side == 'e':
        # what gets done before cauchy
        totchgp = np.sum(source.weights*tau)/(2*np.pi)

        vb += totchgp*source.sawlog
        vinf = np.sum(source.inf_scale*vb)
        vb -= vinf

        after_adj = -totchgp*np.log(np.abs(source.get_inside_point()-target.c)) - \
                                                                        vinf
    else:
        after_adj = np.zeros(target.N, dtype=complex)
    if gradient:
        if side == 'e':
            after_der_adj = totchgp/(source.get_inside_point() - target.c)
        else:
            after_der_adj = np.zeros(target.N, dtype=complex)
        ret = vb, (after_adj, after_der_adj)
    else:
        ret = vb, after_adj
    return ret

