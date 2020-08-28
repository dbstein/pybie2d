import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os

from ...misc.basic_functions import apply_circulant_matrix
from ...kernels.high_level.laplace import Laplace_Layer_Form, Laplace_Layer_Apply

class Laplace_SLP_Self_Kress(object):
    """
    Module providing Laplace SLP Self-Evaluation based on Kress Quadrature
    """
    def __init__(self, GSB):
        """
        Initializes the Kress Quadrature module

        GSB (required): boundary of type Global_Smooth_Boundary
        """
        self.boundary = GSB
        N = self.boundary.N
        dt = self.boundary.dt
        v1 = 4.0*np.sin(np.pi*np.arange(N)/N)**2
        v1[0] = 1.0
        self.V1 = 0.5*np.log(v1)/dt
        self.V1_hat = np.fft.fft(self.V1)
        v2 = np.abs(self.boundary.k)
        v2[0] = np.Inf
        v2[int(N/2)] = np.Inf # experimental!
        self.IV = 1.0/v2
        self.V2 = 0.5*np.fft.ifft(self.IV).real/dt
        self.V2_hat = 0.5*self.IV/dt
        self.V_hat = self.V1_hat/N + self.V2_hat
        self.V = self.V1/N + self.V2

    def Form(self):
        if not hasattr(self, 'MAT'):
            self.MAT = Laplace_SLP_Self_Kress_Form(self.boundary)
        return self.MAT

    def Apply(self, tau, backend='fly'):
        if backend is 'preformed':
            self.Form()
            return self.MAT.dot(tau)
        else:
            return self._Apply(self, tau, backend)

    def _Apply(self, tau, backend='fly'):
        """
        source, required, global_smooth_boundary, source
        tau,   required, dtype(ns), density
        """
        return Laplace_SLP_Self_Kress_Apply(self.boundary, tau, backend)

def Laplace_SLP_Self_Kress_Form(source):
    weights = source.weights
    speed = source.speed
    module = source.Laplace_SLP_Self_Kress
    C = sp.linalg.circulant(module.V)
    A = Laplace_Layer_Form(source, ifcharge=True)
    np.fill_diagonal(A, -np.log(speed)/(2*np.pi)*weights)
    return ne.evaluate('A + C*weights')

def Laplace_SLP_Self_Kress_Apply(source, tau, backend='fly'):
    module = source.Laplace_SLP_Self_Kress
    weighted_tau = tau*source.weights
    u1 = Laplace_Layer_Apply(source, charge=tau, backend=backend)
    u1 -= np.log(source.speed)/(2*np.pi)*weighted_tau
    u2 = apply_circulant_matrix(weighted_tau, c_hat=module.V_hat, real_it=True)
    return u1 + u2
