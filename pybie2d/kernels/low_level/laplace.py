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
def _LKANB(sx, sy, tx, ty, charge, dipstr, nx, ny, pot, ifcharge, ifdipole, doself):
    """
    Numba-jitted Laplace Kernel
    Case:
        Incoming: charge, dipole
        Outgoing: potential
    Inputs:
        sx,     intent(in),  float(ns), x-coordinates of source
        sy,     intent(in),  float(ns), y-coordinates of source
        tx,     intent(in),  float(nt), x-coordinates of target
        ty,     intent(in),  float(nt), y-coordinates of target
        charge, intent(in),  float(ns), charge at source locations
        dipstr, intent(in),  float(ns), dipole strength at source locations
        nx,     intent(in),  float(ns), dipole orientation vector (x-coord)
        ny,     intent(in),  float(ns), dipole orientation vector (y-coord)
        pot,    intent(out), float(nt), potential at target locations
    ns = number of source points; nt = number of target points
    all inputs are required

    This function should generally not be called direclty
    Instead call through the "Laplace_Kernel_Apply_numba" interface
    """
    for i in numba.prange(tx.shape[0]):
        temp = np.zeros(sx.shape[0])
        if ifdipole:
            id2 = np.zeros(sx.shape[0])
            n_dot_d = np.zeros(sx.shape[0])
        for j in range(sx.shape[0]):
            dx = tx[i] - sx[j]
            dy = ty[i] - sy[j]
            temp[j] = dx**2 + dy**2
            if ifdipole:
                n_dot_d[j] = nx[j]*dx + ny[j]*dy
        if ifdipole:
            for j in range(sx.shape[0]):
                if not (doself and i == j):
                    id2[j] = 1.0/temp[j]
                    pot[i] -= n_dot_d[j]*id2[j]*dipstr[j]
        if ifcharge:
            for j in range(sx.shape[0]):
                temp[j] = np.log(temp[j])
            for j in range(sx.shape[0]):
                if not (doself and i == j):
                    pot[i] += 0.5*charge[j]*temp[j]

@numba.njit(parallel=True)
def _LKANBG(sx, sy, tx, ty, charge, dipstr, nx, ny, pot, gradx, grady, ifcharge, ifdipole, doself):
    """
    Numba-jitted Laplace Kernel
    Case:
        Incoming: charge, dipole
        Outgoing: potential, gradient
    Inputs:
        sx,     intent(in),  float(ns), x-coordinates of source
        sy,     intent(in),  float(ns), y-coordinates of source
        tx,     intent(in),  float(nt), x-coordinates of target
        ty,     intent(in),  float(nt), y-coordinates of target
        charge, intent(in),  float(ns), charge at source locations
        dipstr, intent(in),  float(ns), dipole strength at source locations
        nx,     intent(in),  float(ns), dipole orientation vector (x-coord)
        ny,     intent(in),  float(ns), dipole orientation vector (y-coord)
        pot,    intent(out), float(nt), potential at target locations
        gradx,  intent(out), float(nt), x-derivative of potential
        grady,  intent(out), float(nt), y-derivative of potential
    ns = number of source points; nt = number of target points
    all inputs are required

    This function should generally not be called direclty
    Instead call through the "Laplace_Kernel_Apply_numba" interface
    """
    for i in numba.prange(tx.shape[0]):
        temp = np.zeros(sx.shape[0])
        dx = np.zeros(sx.shape[0])
        dy = np.zeros(sx.shape[0])
        id2 = np.zeros(sx.shape[0])
        if ifdipole:
            id4 = np.zeros(sx.shape[0])
            n_dot_d = np.zeros(sx.shape[0])
        for j in range(sx.shape[0]):
            dx[j] = tx[i] - sx[j]
            dy[j] = ty[i] - sy[j]
            temp[j] = dx[j]**2 + dy[j]**2
        for j in range(sx.shape[0]):
            id2[j] = 1.0/temp[j]
        if ifcharge:
            for j in range(sx.shape[0]):
                temp[j] = np.log(temp[j])
        if ifdipole:
            for j in range(sx.shape[0]):
                id4[j] = id2[j]*id2[j]
                n_dot_d[j] = nx[j]*dx[j] + ny[j]*dy[j]
        if ifcharge:
            for j in range(sx.shape[0]):
                if not (doself and i == j):
                    pot[i] += 0.5*charge[j]*temp[j]
                    gradx[i] += dx[j]*id2[j]*charge[j]
                    grady[i] += dy[j]*id2[j]*charge[j]
        if ifdipole:
            for j in range(sx.shape[0]):
                if not (doself and i == j):
                    pot[i] -= n_dot_d[j]*id2[j]*dipstr[j]
                    gradx[i] += (nx[j]*(dx[j]*dx[j] - dy[j]*dy[j]) + 2*ny[j]*dx[j]*dy[j])*id4[j]*dipstr[j]
                    grady[i] += (nx[j]*2*dx[j]*dy[j] + ny[j]*(dy[j]*dy[j] - dx[j]*dx[j]))*id4[j]*dipstr[j]

def Laplace_Kernel_Apply_numba(source, target, charge=None, dipstr=None,
                                dipvec=None, weights=None, gradient=False):
    """
    Interface to numba-jitted Laplace Kernels
    Inputs:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        charge,   optional, float(ns),     charge at source locations
        dipstr,   optional, float(ns),     dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
        gradient, optional, bool,          whether to compute gradient or not
    Outputs:
        if gradient == False:
            float(nt), potential at target coordinates
        if gradient == True:
            tuple of:
                float(nt), potential at target coordinates
                float(nt), x-derivative of potential at target coordinates
                float(nt), y-derivative of potential at target coordinates
    ns = number of source points; nt = number of target points
    """
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5*weights/np.pi
    sx = source[0]
    sy = source[1]
    tx = target[0]
    ty = target[1]
    ifcharge = charge is not None
    ifdipole = dipstr is not None
    pot = np.zeros(target.shape[1], dtype=float)
    zero_vec = np.zeros(source.shape[1], dtype=float)
    ch = zero_vec if charge is None else charge*weighted_weights
    ds = zero_vec if dipstr is None else dipstr*weighted_weights
    nx = zero_vec if dipvec is None else dipvec[0]
    ny = zero_vec if dipvec is None else dipvec[1]
    if gradient:
        gradx = np.zeros(target.shape[1], dtype=float)
        grady = np.zeros(target.shape[1], dtype=float)
        _LKANBG(sx, sy, tx, ty, ch, ds, nx, ny, pot, gradx, grady, ifcharge, ifdipole, False)
        return pot, gradx, grady
    else:
        _LKANB(sx, sy, tx, ty, ch, ds, nx, ny, pot, ifcharge, ifdipole, False)
        return pot

def Laplace_Kernel_Apply_FMM(source, target, charge=None, dipstr=None,
                                    dipvec=None, weights=None, gradient=False):
    """
    Interface to FMM Laplace Kernels
    Inputs:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        charge,   optional, float(ns),     charge at source locations
        dipstr,   optional, float(ns),     dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
        gradient, optional, bool,          whether to compute gradient or not
    Outputs:
        if gradient == False:
            float(nt), potential at target coordinates
        if gradient == True:
            tuple of:
                float(nt), potential at target coordinates
                float(nt), x-derivative of potential at target coordinates
                float(nt), y-derivative of potential at target coordinates
    ns = number of source points; nt = number of target points
    """
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5*weights/np.pi
    ch = charge*weighted_weights if charge is not None else None
    ds = dipstr*weighted_weights if dipstr is not None else None
    out = FMM(kind='laplace', source=source, target=target, charge=ch,
                dipstr=ds, dipvec=dipvec, compute_target_potential=True,
                compute_target_gradient=gradient)['target']
    if gradient:
        return out['u'], out['u_x'], out['u_y']
    else:
        return out['u']

Laplace_Kernel_Applys = {}
Laplace_Kernel_Applys['numba'] = Laplace_Kernel_Apply_numba
Laplace_Kernel_Applys['FMM']   = Laplace_Kernel_Apply_FMM

def Laplace_Kernel_Apply(source, target, charge=None, dipstr=None, dipvec=None,
                                weights=None, gradient=False, backend='numba'):
    """
    Laplace Kernel Apply
    Inputs:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        charge,   optional, float(ns),     charge at source locations
        dipstr,   optional, float(ns),     dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
        gradient, optional, bool,          whether to compute gradient or not
        backend,  optional, str,           backend ('FMM' or 'numba')
    Outputs:
        if gradient == False:
            float(nt), potential at target coordinates
        if gradient == True:
            tuple of:
                float(nt), potential at target coordinates
                float(nt), x-derivative of potential at target coordinates
                float(nt), y-derivative of potential at target coordinates
    ns = number of source points; nt = number of target points
    """
    return Laplace_Kernel_Applys[backend](source, target, charge, dipstr,
                                                    dipvec, weights, gradient)

################################################################################
# General Purpose Low Level Source --> Source Kernel Apply Functions
# These are naive quadratures with no self interaction!
# only potentials are currently implemented for self-interaction

def Laplace_Kernel_Self_Apply_numba(source, charge=None, dipstr=None,
                                    dipvec=None, weights=None, gradient=False):
    """
    Interface to numba-jitted Laplace Kernels (self, naive quadrature)
    Inputs:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        charge,   optional, float(ns),     charge at source locations
        dipstr,   optional, float(ns),     dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
        gradient, optional, bool,          whether to compute gradient or not
    Outputs:
        if gradient == False:
            float(nt), potential at target coordinates
        if gradient == True:
            tuple of:
                float(nt), potential at target coordinates
                float(nt), x-derivative of potential at target coordinates
                float(nt), y-derivative of potential at target coordinates
    ns = number of source points; nt = number of target points
    """
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5*weights/np.pi
    sx = source[0]
    sy = source[1]
    ifcharge = charge is not None
    ifdipole = dipstr is not None
    pot = np.zeros(source.shape[1], dtype=float)
    zero_vec = np.zeros(source.shape[1], dtype=float)
    ch = zero_vec if charge is None else charge*weighted_weights
    ds = zero_vec if dipstr is None else dipstr*weighted_weights
    nx = zero_vec if dipvec is None else dipvec[0]
    ny = zero_vec if dipvec is None else dipvec[1]
    if gradient:
        gradx = np.zeros(source.shape[1], dtype=float)
        grady = np.zeros(source.shape[1], dtype=float)
        _LKANBG(sx, sy, sx, sy, ch, ds, nx, ny, pot, gradx, grady, ifcharge, ifdipole, True)
        return pot, gradx, grady
    else:
        _LKANB(sx, sy, sx, sy, ch, ds, nx, ny, pot, ifcharge, ifdipole, True)
        return pot

def Laplace_Kernel_Self_Apply_FMM(source, charge=None, dipstr=None,
                                    dipvec=None, weights=None, gradient=False):
    """
    Interface to FMM Laplace Kernels (self, naive quadrature)
    Inputs:
        source,   required, float(2, ns),  source coordinates
        charge,   optional, float(ns),     charge at source locations
        dipstr,   optional, float(ns),     dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
        gradient, optional, bool,          whether to compute gradient or not
    Outputs:
        if gradient == False:
            float(ns), potential at target coordinates
        if gradient == True:
            tuple of:
                float(ns), potential at target coordinates
                float(ns), x-derivative of potential at target coordinates
                float(ns), y-derivative of potential at target coordinates
    ns = number of source points
    """
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5*weights/np.pi
    ch = charge*weighted_weights if charge is not None else None
    ds = dipstr*weighted_weights if dipstr is not None else None
    out = FMM(kind='laplace', source=source, target=source, charge=ch,
        dipstr=ds, dipvec=dipvec, compute_source_potential=True,
        compute_source_gradient=gradient)['source']
    if gradient:
        return out['u'], out['u_x'], out['u_y']
    else:
        return out['u']

Laplace_Kernel_Self_Applys = {}
Laplace_Kernel_Self_Applys['numba'] = Laplace_Kernel_Self_Apply_numba
Laplace_Kernel_Self_Applys['FMM']   = Laplace_Kernel_Self_Apply_FMM

def Laplace_Kernel_Self_Apply(source, charge=None, dipstr=None,
            dipvec=None, weights=None, backend='numba', gradient=False):
    """
    Interface to Laplace Kernels (self, naive quadrature)
    Inputs:
        source,   required, float(2, ns),  source coordinates
        charge,   optional, float(ns),     charge at source locations
        dipstr,   optional, float(ns),     dipole strength at source locations
        dipvec,   optional, float(2, ns),  dipole orientation at source loc
        weights,  optional, float(ns),     quadrature weights
        gradient, optional, bool,          whether to compute gradient or not
        backend,  optional, str,           backend ('FMM' or 'numba')
    Outputs:
        if gradient == False:
            float(ns), potential at target coordinates
        if gradient == True:
            tuple of:
                float(ns), potential at target coordinates
                float(ns), x-derivative of potential at target coordinates
                float(ns), y-derivative of potential at target coordinates
    ns = number of source points
    """
    return Laplace_Kernel_Self_Applys[backend](source, charge, dipstr, dipvec,
                                                            weights, gradient)

################################################################################
# General Purpose Low Level Source --> Target Kernel Formation

def Laplace_Kernel_Form(source, target, ifcharge=False, chweight=None,
                            ifdipole=False, dpweight=None, dipvec=None,
                            weights=None, gradient=False):
    """
    Laplace Kernel Formation
    Computes the matrix:
        [ chweight*G_ij + dpweight*(n_j dot grad G_ij) ] weights_j
        where G is the Laplace Greens function and other parameters described
        below
    Also returns the matrices for the x and y derivatives, if requested

    Parameters:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        ifcharge, optional, bool,          include charge contribution
        chweight, optional, float,         scalar weight to apply to charges
        ifdipole, optional, bool,          include dipole contribution
        dpweight, optional, float,         scalar weight to apply to dipoles
        dipvec,   optional, float(2, ns),  dipole orientations
        weights,  optional, float(ns),     quadrature weights
        gradient, optional, bool,          compute gradients or not

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
    scale = -1.0/(2*np.pi)
    scale = scale*np.ones(ns) if weights is None else scale*weights
    chscale = scale if chweight is None else scale*chweight
    dpscale = scale if dpweight is None else scale*dpweight
    G = np.zeros([nt, ns], dtype=float)
    if gradient:
        Gx = np.zeros([nt, ns], dtype=float)
        Gy = np.zeros([nt, ns], dtype=float)
    if not (ifcharge or ifdipole):
        # no charges, no dipoles
        # just return appropriate zero matrices
        if gradient:
            return G, Gx, Gy
        else:
            return G
    else:
        dx = ne.evaluate('TX - SX')
        dy = ne.evaluate('TY - SY')
        id2 = ne.evaluate('1.0/(dx**2 + dy**2)')
        if ifcharge:
            # charges effect on potential
            ne.evaluate('G - 0.5*log(id2)*chscale', out=G)
        if ifdipole:
            # dipoles effect on potential
            ne.evaluate('G - (nx*dx + ny*dy)*id2*dpscale', out=G)
        if gradient:
            Gx1 = np.empty_like(dx)
            Gy1 = np.empty_like(dx)
            if ifcharge:
                # charges effect on gradient
                ne.evaluate('Gx + dx*id2*chscale', out=Gx)
                ne.evaluate('Gy + dy*id2*chscale', out=Gy)
            if ifdipole:
                # dipoles effect on gradient
                id4 = ne.evaluate('id2*id2')
                ne.evaluate('Gx + (nx*(dx*dx-dy*dy) + 2*ny*dx*dy)*id4*dpscale', out=Gx)
                ne.evaluate('Gy + (2*nx*dx*dy + ny*(dy*dy-dx*dx))*id4*dpscale', out=Gy)
            return G, Gx, Gy
        else:
            return G

################################################################################
# General Purpose Low Level Source --> Source Kernel Formation
# These are naive quadratures with no self interaction!
# only potentials are currently implemented for self-interaction

def Laplace_Kernel_Self_Form(source, ifcharge=False, chweight=None,
    ifdipole=False, dpweight=None, dipvec=None, weights=None, gradient=False):
    """
    Laplace Kernel Formation
    Computes the matrix:
        [ chweight*G_ij + dpweight*(n_j dot grad G_ij) ] weights_j
        where G is the Laplace Greens function and other parameters described
        below, and the diagonal is set to 0

    Parameters:
        source,   required, float(2, ns),  source coordinates
        ifcharge, optional, bool,          include charge contribution
        chweight, optional, float,         scalar weight to apply to charges
        ifdipole, optional, bool,          include dipole contribution
        dpweight, optional, float,         scalar weight to apply to dipoles
        dipvec,   optional, float(2, ns),  dipole orientations
        weights,  optional, float(ns),     quadrature weights
    """
    Gs = Laplace_Kernel_Form(source, source, ifcharge, chweight, ifdipole,
                                            dpweight, dipvec, weights, gradient)
    if gradient:
        for i in range(3):
            np.fill_diagonal(Gs[i], 0.0)
        return Gs
    else:
        np.fill_diagonal(Gs, 0.0)
        return Gs
