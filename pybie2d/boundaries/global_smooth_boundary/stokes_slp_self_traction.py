import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import numba
import warnings
import os

class Stokes_SLP_Self_Traction(object):
    """
    Module providing Stokes SLP Self-Evaluation for the traction kernel

    Note: this doesn't depend on the boundary so maybe should just be
            in the kernels?  This probably isn't the correct place to put it
    """
    def __init__(self, GSB):
        """
        Stokes SLP Self-Evaluation
        """
        self.boundary = GSB

    def Form(self):
        if not hasattr(self, 'MAT'):
            self.MAT = Stokes_SLP_Self_Traction_Form(self.boundary)
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
        return Stokes_SLP_Self_Traction_Apply(source, tau)

def Stokes_SLP_Self_Traction_Form(source):
    ns = source.N
    SX = source.x
    SY = source.y
    TX = source.x[:,None]
    TY = source.y[:,None]
    nx = source.normal_x[:,None]
    ny = source.normal_y[:,None]
    weights = np.concatenate([source.weights, source.weights])
    G = np.zeros([2*ns, 2*ns], dtype=float)
    dx = ne.evaluate('TX - SX')
    dy = ne.evaluate('TY - SY')
    id2 = ne.evaluate('1.0/(dx**2 + dy**2)')
    W = np.empty_like(dx)
    GH = np.empty_like(dx)
    d_dot_n_ir4 = ne.evaluate('(dx*nx + dy*ny)*id2*id2', out=W)
    ipi = 1.0/np.pi
    ne.evaluate('-ipi*d_dot_n_ir4*dx*dx', out=GH)
    G[:ns, :ns] += GH
    ne.evaluate('-ipi*d_dot_n_ir4*dx*dy', out=GH)
    G[ns:, :ns] += GH
    G[:ns, ns:] += GH
    ne.evaluate('-ipi*d_dot_n_ir4*dy*dy', out=GH)   
    G[ns:, ns:] += GH
    # now fill in the correct diagonal limits
    scale = -0.5*source.curvature/np.pi
    tx = source.tangent_x
    ty = source.tangent_y
    s01 = scale*tx*ty
    np.fill_diagonal(G[:ns, :ns], scale*tx*tx)
    np.fill_diagonal(G[ns:, :ns], s01)
    np.fill_diagonal(G[:ns, ns:], s01)
    np.fill_diagonal(G[ns:, ns:], scale*ty*ty)
    return G*weights

@numba.njit(parallel=True)
def _SSKT(sx, sy, fx, fy, nx, ny, u, v):
    for i in numba.prange(sx.shape[0]):
        for j in range(sx.shape[0]):
            if not (doself and i == j):
                dx = tx[i] - sx[j]
                dy = ty[i] - sy[j]
                dxdx = dx*dx
                dxdy = dx*dy
                dydy = dy*dy
                id2 = 1.0/(dxdx + dydy)
                d_dot_n = dx*nx[i] + dy*ny[i]
                d_dot_n_ir4 = d_dot_n*id2*id2
                Gd00 = d_dot_n_ir4*dxdx
                Gd01 = d_dot_n_ir4*dxdy
                Gd11 = d_dot_n_ir4*dydy
                u[i] += (Gd00*fx[j] + Gd01*fy[j])
                v[i] += (Gd01*fx[j] + Gd11*fy[j])

def Stokes_SLP_Self_Traction_Apply(source, tau):
    weights = -source.weights/np.pi
    sx = source[0]
    sy = source[1]
    velocity = np.zeros([2,source.N], dtype=float)
    u = velocity[0]
    v = velocity[1]
    fx = tau[:source.N]*weights
    fy = tau[source.N:]*weights
    nx = source.normal_x
    ny = source.normal_y
    _SSKT(sx, sy, fx, fy, nx, ny, u, v)
    return velocity
