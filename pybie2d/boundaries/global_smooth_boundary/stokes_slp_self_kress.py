import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os

class Stokes_SLP_Self_Kress(object):
    """
    Module providing Stokes SLP Self-Evaluation based on Kress Quadrature
    """
    def __init__(self, GSB):
        """
        Initializes the Kress Quadrature module

        GSB (required): boundary of type Global_Smooth_Boundary
        """
        GSB.add_module('Laplace_SLP_Self_Kress')
        self.boundary = GSB

    def Form(self):
        if not hasattr(self, 'MAT'):
            self.MAT = Stokes_SLP_Self_Kress_Form(self.boundary)
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
        tau,   required, dtype(2*ns), density
        """
        return Stokes_SLP_Self_Kress_Apply(source, tau, backend)

def Stokes_SLP_Self_Kress_Form(source):
    N = source.N
    weights = source.weights
    mu = 1.0
    sx = source.x
    sxT = sx[:,None]
    sy = source.y
    syT = sy[:,None]
    dx = ne.evaluate('sxT-sx')
    dy = ne.evaluate('syT-sy')
    irr = ne.evaluate('1.0/(dx*dx + dy*dy)')
    weights = weights/(4.0*np.pi*mu)
    A = np.empty((2*N,2*N), dtype=float)
    A00 = ne.evaluate('weights*dx*dx*irr', out=A[:N,:N])
    A01 = ne.evaluate('weights*dx*dy*irr', out=A[:N,N:])
    A10 = ne.evaluate('A01', out=A[N:,:N])
    A11 = ne.evaluate('weights*dy*dy*irr', out=A[N:,N:])
    tx = source.tangent_x
    ty = source.tangent_y
    np.fill_diagonal(A00, ne.evaluate('weights*tx*tx'))
    np.fill_diagonal(A01, ne.evaluate('weights*tx*ty'))
    np.fill_diagonal(A10, ne.evaluate('weights*tx*ty'))
    np.fill_diagonal(A11, ne.evaluate('weights*ty*ty'))
    S = source.Laplace_SLP_Self_Kress.Form()
    muS = ne.evaluate('0.5*mu*S')
    A00 = ne.evaluate('A00 + muS', out=A[:N,:N])
    A11 = ne.evaluate('A11 + muS', out=A[N:,N:])
    return A

def Stokes_SLP_Self_Kress_Apply(source, tau, backend='fly'):
    mu = 1.0
    N = source.N
    weights = source.weights
    tx = source.tangent_x
    ty = source.tangent_y
    u1 = Stokes_Layer_Apply(source, charge=tau, backend=backend)
    diag1 = weights*tx*tx*tau[:N] + weights*tx*ty*tau[N:]
    diag2 = weights*tx*ty*tau[:N] + weights*ty*ty*tau[N:]
    u2 = np.concatenate([diag1, diag2])
    S1 = source.Laplace_SLP_Self_Kress.Apply(tau[:source.N], backend)
    S2 = source.Laplace_SLP_Self_Kress.Apply(tau[source.N:], backend)
    u3 = np.concatenate([S1, S2])
    return u1 + u2 + 0.5*mu*u3
