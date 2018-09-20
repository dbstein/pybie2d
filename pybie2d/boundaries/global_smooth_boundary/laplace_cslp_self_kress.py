import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import numba
import warnings
import os

from ...misc.basic_functions import apply_circulant_matrix

class Laplace_CSLP_Self_Kress(object):
    """
    Module providing Laplace CSLP Self-Evaluation based on Kress Quadrature
    """
    def __init__(self, GSB):
        """
        Initializes the CSLP Kress Quadrature module

        GSB (required): boundary of type Global_Smooth_Boundary
        """
        GSB.add_module('Laplace_SLP_Self_Kress')
        self.boundary = GSB
        K = self.boundary.Laplace_SLP_Self_Kress
        N = self.boundary.N
        Nh = int(N/2)
        t = self.boundary.t
        weights = self.boundary.weights
        complex_weights = self.boundary.complex_weights
        cp = self.boundary.cp
        inside_point = self.boundary.get_inside_point()

        RVe = K.IV.copy()
        RVe[Nh+1:] = 0.0
        RVe[Nh] /= 2
        RVi = K.IV.copy()
        RVi[:Nh] = 0.0
        self.VRe = np.fft.ifft(RVi)
        self.VRi = np.fft.ifft(RVe)
        self.VRe_hat = RVi
        self.VRi_hat = RVe
        self.Kx = (np.cos(t) + 1j*np.sin(t))
        self.CSLP_limit =  np.log(1j*self.Kx/cp) * weights/(2*np.pi)

    def Form(self, side):
        attr = 'CSLP_' + side
        if not hasattr(self, attr):
            if not hasattr(self, 'CSLP_Base'):
                weights = self.boundary.weights
                speed = self.boundary.speed
                scaled_weights = weights/(2*np.pi)
                source_c = self.boundary.c
                scT = source_c[:,None]
                kress_x = self.Kx
                kress_xT = kress_x[:,None]
                xd = ne.evaluate('kress_xT-kress_x')
                d = ne.evaluate('scT - source_c')
                S = ne.evaluate('log(xd/d)')
                SI = S.imag
                np.fill_diagonal(SI, np.concatenate([ SI.diagonal(1), (SI[-1,0],) ]) )
                SI = np.unwrap(SI)
                S = ne.evaluate('S*scaled_weights', out=S)
                np.fill_diagonal(S, self.CSLP_limit)
                self.CSLP_Base = S
            else:
                S = self.CSLP_Base
            RV = self.VRi if side == 'i' else self.VRe
            R = sp.linalg.circulant(RV)
            SR = ne.evaluate('S + R*speed')
            setattr(self, attr, SR)
        return getattr(self, attr)

    def Apply(self, side, tau):
        u1 = np.empty_like(self.boundary.c)
        aweights = self.boundary.weights/(2*np.pi)
        _CSLP(self.boundary.c, self.Kx, tau*aweights, self.CSLP_limit/aweights, u1)
        circ = self.VRi_hat if side == 'i' else self.VRe_hat
        u2 = apply_circulant_matrix(tau*self.boundary.speed, c_hat=circ,
                                                                real_it=False)
        return u1 + u2

@numba.njit(parallel=True)
def _CSLP(s, p, tau, diag, pot):
    # CSLP apply
    # this function correctly handles phase jumps in the imaginary portion
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
