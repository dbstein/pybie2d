import numpy as np
import numexpr as ne
import numba
import warnings

from ... import have_fmm
if have_fmm:
    from ... import FMM

################################################################################
# General Purpose Low Level Source --> Target Kernel Apply Functions

@numba.njit(parallel=True)
def _SKANB(sx, sy, tx, ty, fx, fy, dipx, dipy, nx, ny, u, v, ifforce, ifdipole):
    """
    Numba-jitted Stokes Kernel
    Case:
        Incoming: force, dipole
        Outgoing: potential
    Inputs:
        sx,  intent(in),  float(ns), x-coordinates of source
        sy,  intent(in),  float(ns), y-coordinates of source
        tx,  intent(in),  float(nt), x-coordinates of target
        ty,  intent(in),  float(nt), y-coordinates of target
        fx,  intent(in),  float(ns), x-component of force
        fy,  intent(in),  float(ns), y-component of force
        u,   intent(out), float(nt), x-component of velocity at targ locations
        v,   intent(out), float(nt), y-component of velocity at targ locations
    ns = number of source points; nt = number of target points
    all inputs are required

    This function should generally not be called direclty
    Instead call through the "Stokes_Kernel_Apply_numba" interface
    Note that a 0.25/pi weight is applied to fx, fy in that interface
    Note that a 1.0/pi weight is applied to dipx, dipy in that interface
    """
    doself = sx is tx and sy is ty
    for i in numba.prange(tx.shape[0]):
        temp = np.zeros(sx.shape[0])
        for j in range(sx.shape[0]):
            dx = tx[i] - sx[j]
            dy = ty[i] - sy[j]
            dxdx = dx*dx
            dxdy = dx*dy
            dydy = dy*dy
            temp[j] = dxdx + dydy
            id2 = 1.0/temp[j]
            if ifforce:
                G00 = dxdx*id2
                G01 = dxdy*id2
                G11 = dydy*id2
                if not (doself and i == j):
                    u[i] += (G00*fx[j] + G01*fy[j])
                    v[i] += (G01*fx[j] + G11*fy[j])
            if ifdipole:
                d_dot_n = dx*nx[j] + dy*ny[j]
                d_dot_n_ir4 = d_dot_n*id2*id2
                Gd00 = d_dot_n_ir4*dxdx
                Gd01 = d_dot_n_ir4*dxdy
                Gd11 = d_dot_n_ir4*dydy
                if not (doself and i == j):
                    u[i] += (Gd00*dipx[j] + Gd01*dipy[j])
                    v[i] += (Gd01*dipx[j] + Gd11*dipy[j])
        if ifforce:
            for j in range(sx.shape[0]):
                    temp[j] = np.log(temp[j])
            for j in range(sx.shape[0]):
                if not (doself and i == j):
                    u[i] -= 0.5*temp[j]*fx[j]
                    v[i] -= 0.5*temp[j]*fy[j]

def Stokes_Kernel_Apply_numba(source, target, forces=None, dipstr=None,
                                                    dipvec=None, weights=None):
    """
    Interface to numba-jitted Stokes Kernels
    Inputs:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        forces,   optional, float(2, ns),  forces at source locations
        dipstr,   optional, float(2, ns),  dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
    Outputs:
        float(2, nt), velocity at target coordinates
    ns = number of source points; nt = number of target points
    """
    weights = 1.0 if weights is None else weights
    weighted_weights1 = 0.25*weights/np.pi
    weighted_weights2 = weights/np.pi
    sx = source[0]
    sy = source[1]
    tx = target[0]
    ty = target[1]
    velocity = np.zeros([2,target.shape[1]], dtype=float)
    u = velocity[0]
    v = velocity[1]
    ifforces = forces is not None
    ifdipole = dipstr is not None
    zero_vec = np.zeros(source.shape[1], dtype=float)
    fx = zero_vec if forces is None else forces[0]*weighted_weights1
    fy = zero_vec if forces is None else forces[1]*weighted_weights1
    dipx = zero_vec if dipstr is None else dipstr[0]*weighted_weights2
    dipy = zero_vec if dipstr is None else dipstr[1]*weighted_weights2
    nx = zero_vec if dipvec is None else dipvec[0]
    ny = zero_vec if dipvec is None else dipvec[1]
    _SKANB(sx, sy, tx, ty, fx, fy, dipx, dipy, nx, ny, u, v, ifforces, ifdipole)
    return velocity

def Stokes_Kernel_Apply_FMM(source, target, forces=None, dipstr=None,
                                                    dipvec=None, weights=None):
    """
    Interface to FMM Stokes Kernels
    Inputs:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        forces,   optional, float(2, ns),  forces at source locations
        dipstr,   optional, float(2, ns),  dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
    Outputs:
        float(2, nt), velocity at target coordinates
    ns = number of source points; nt = number of target points
    """
    weights = 1.0 if weights is None else weights
    wf = None if forces is None else forces*weights
    wd = None if dipstr is None else dipstr*weights
    if source is target:
        out = FMM(kind='stokes', source=source, forces=wf, 
                dipstr=wd, dipvec=dipvec, compute_source_velocity=True)['source']
    else:
        out = FMM(kind='stokes', source=source, target=target, forces=wf, 
                dipstr=wd, dipvec=dipvec, compute_target_velocity=True)['target']
    return np.row_stack([out['u'], out['v']])

Stokes_Kernel_Applys = {}
Stokes_Kernel_Applys['numba'] = Stokes_Kernel_Apply_numba
Stokes_Kernel_Applys['FMM']   = Stokes_Kernel_Apply_FMM

def Stokes_Pressure_Apply_FMM(source, target, forces=None, dipstr=None,
                                                    dipvec=None, weights=None):
    """
    Interface to FMM Stokes Kernels
    Inputs:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        forces,   optional, float(2, ns),  forces at source locations
        dipstr,   optional, float(2, ns),  dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
    Outputs:
        float(2, nt), velocity at target coordinates
    ns = number of source points; nt = number of target points
    """
    weights = 1.0 if weights is None else weights
    wf = None if forces is None else forces*weights
    wd = None if dipstr is None else dipstr*weights
    out = FMM(kind='stokes', source=source, target=target, forces=wf, 
            dipstr=wd, dipvec=dipvec, compute_target_stress=True)['target']
    return out['p']

def Stokes_Kernel_Apply(source, target, forces=None, dipstr=None, dipvec=None,
                                                weights=None, backend='numba'):
    """
    Laplace Kernel Apply
    Inputs:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        forces,   optional, float(2, ns),  forces at source locations
        dipstr,   optional, float(2, ns),  dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
        backend,  optional, str,           backend, ('numba' or 'FMM')
    Outputs:
        float(2, nt), velocity at target coordinates
    ns = number of source points; nt = number of target points
    """
    return Stokes_Kernel_Applys[backend](source, target, forces, dipstr,
                                                                dipvec, weights)

################################################################################
# General Purpose Low Level Source --> Target Kernel Formation

def Stokes_Kernel_Form(source, target, ifforce=False, fweight=None,
                    ifdipole=False, dpweight=None, dipvec=None, weights=None):
    """
    Stokes Kernel Formation

    Parameters:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        ifforce,  optional, bool,          include force contribution
        fweight,  optional, float,         scalar weight to apply to forces
        ifdipole, optional, bool,          include dipole contribution
        dpweight, optional, float,         scalar weight to apply to dipoles
        dipvec,   optional, float(2, ns),  dipole orientations
        weights,  optional, float(ns),     quadrature weights

    This function assumes that source and target have no coincident points
    """
    ns = source.shape[1]
    nt = target.shape[1]
    SX = source[0]
    SY = source[1]
    TX = target[0][:,None]
    TY = target[1][:,None]
    if dipvec is not None:
        nx = dipvec[0]
        ny = dipvec[1]
    fscale = 0.25/np.pi
    dscale = 1.0/np.pi
    if weights is not None:
        fscale *= weights
        dscale *= weights
    if fweight is not None:
        fscale *= fweight
    if dpweight is not None:
        dscale *= dpweight
    G = np.zeros([2*nt, 2*ns], dtype=float)
    if not (ifforce or ifdipole):
        # no forces, no dipoles
        # just return appropriate zero matrix
        return G
    else:
        dx = ne.evaluate('TX - SX')
        dy = ne.evaluate('TY - SY')
        id2 = ne.evaluate('1.0/(dx**2 + dy**2)')
        W = np.empty_like(dx)
        GH = np.empty_like(dx)
        if ifforce:
            # forces effect on velocity
            logid = ne.evaluate('0.5*log(id2)', out=W)
            ne.evaluate('fscale*(logid + dx*dx*id2)', out=GH)
            G[:nt, :ns] += GH
            ne.evaluate('fscale*dx*dy*id2', out=GH)
            G[nt:, :ns] += GH
            G[:nt, ns:] += GH            
            GH = ne.evaluate('fscale*(logid + dy*dy*id2)')
            G[nt:, ns:] += GH
        if ifdipole:
            # dipoles effect on velocity
            d_dot_n_ir4 = ne.evaluate('(dx*nx + dy*ny)*id2*id2', out=W)
            ne.evaluate('dscale*d_dot_n_ir4*dx*dx', out=GH)
            G[:nt, :ns] += GH
            ne.evaluate('dscale*d_dot_n_ir4*dx*dy', out=GH)
            G[nt:, :ns] += GH
            G[:nt, ns:] += GH
            ne.evaluate('dscale*d_dot_n_ir4*dy*dy', out=GH)   
            G[nt:, ns:] += GH
        if source is target:
            np.fill_diagonal(G, 0.0)
            np.fill_diagonal(G[:ns, ns:], 0.0)
            np.fill_diagonal(G[ns:, :ns], 0.0)
        return G
