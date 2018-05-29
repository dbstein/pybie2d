import numpy as np
import numexpr as ne
import numba
import warnings

from ..backend_defaults import get_backend
from .. import have_fmm
if have_fmm:
    from .. import FMM

################################################################################
# General Purpose Source --> Target Kernel Calls

def Laplace_Kernel_Apply(source, target, charge=None, dipstr=None, dipvec=None,
                    weights=None, gradient=False, dtype=float, backend='fly'):
    """
    Laplace Kernel Apply (potential and gradient) in 2D
    Computes the sum:
        u_i = sum_j[G_ij charge_j + (dipvec_j dot grad G_ij) dipstr_j] weights_j
    and appropriate derivatives, where:

    Parameters:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        charge,   optional, dtype(ns),     charge
        dipstr,   optional, dtype(ns),     dipole strength
        dipvec,   optional, float(2, ns),  dipole orientations
        weights,  optional, float(ns),     quadrature weights
        gradient, optional, bool,          compute gradients or not
        dtype,    optional, float/complex, whether this transform is for float
                                            only, or complex densities
        backend,  optional, str,           'fly', 'numba', 'fmm'

        This function assumes that source and target have no coincident points
    """
    backend = get_backend(source.shape[1], target.shape[1], backend)
    return Laplace_Kernel_Applys[backend](source, target, charge, dipstr,
                                            dipvec, weights, gradient, dtype)

@numba.njit(parallel=True)
def _Laplace_Kernel_Apply_numba_charge(sx, sy, tx, ty, charge, pot):
    scale = -0.5/np.pi
    for i in numba.prange(tx.shape[0]):
        for j in range(sx.shape[0]):
            dx = tx[i] - sx[j]
            dy = ty[i] - sy[j]
            d2 = dx**2 + dy**2
            pot[i] += 0.5*np.log(d2)*charge[j]
@numba.njit(parallel=True)
def _Laplace_Kernel_Apply_numba_charge_gradient(sx, sy, tx, ty, charge, pot,
                                                                gradx, grady):
    for i in numba.prange(tx.shape[0]):
        for j in range(sx.shape[0]):
            dx = tx[i] - sx[j]
            dy = ty[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            pot[i] += 0.5*np.log(d2)*charge[j]
            gradx[i] += dx*id2*charge[j]
            grady[i] += dy*id2*charge[j]
@numba.njit(parallel=True)
def _Laplace_Kernel_Apply_numba_dipole(sx, sy, tx, ty, dipstr, nx, ny, pot):
    for i in numba.prange(tx.shape[0]):
        for j in range(sx.shape[0]):
            dx = tx[i] - sx[j]
            dy = ty[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] -= n_dot_d*id2*dipstr[j]
@numba.njit(parallel=True)
def _Laplace_Kernel_Apply_numba_dipole_gradient(sx, sy, tx, ty, dipstr, nx, ny,
                                                            pot, gradx, grady):
    for i in numba.prange(tx.shape[0]):
        for j in range(sx.shape[0]):
            dx = tx[i] - sx[j]
            dy = ty[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] -= n_dot_d*id2*dipstr[j]
            id4 = id2*id2
            gradx[i] += (nx[j]*(dx*dx - dy*dy) + 2*ny[j]*dx*dy)*id4*dipstr[j]
            grady[i] += (nx[j]*2*dx*dy + ny[j]*(dy*dy - dx*dx))*id4*dipstr[j]
@numba.njit(parallel=True)
def _Laplace_Kernel_Apply_numba_both(sx, sy, tx, ty, charge, dipstr, nx, ny,
                                                                        pot):
    for i in numba.prange(tx.shape[0]):
        for j in range(sx.shape[0]):
            dx = tx[i] - sx[j]
            dy = ty[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] += 0.5*np.log(d2)*charge[j]
            pot[i] -= n_dot_d*id2*dipstr[j]
@numba.njit(parallel=True)
def _Laplace_Kernel_Apply_numba_both_gradient(sx, sy, tx, ty, charge, dipstr,
                                                    nx, ny, pot, gradx, grady):
    for i in numba.prange(tx.shape[0]):
        for j in range(sx.shape[0]):
            dx = tx[i] - sx[j]
            dy = ty[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] += 0.5*np.log(d2)*charge[j]
            pot[i] -= n_dot_d*id2*dipstr[j]
            id4 = id2*id2
            gradx[i] += dx*id2*charge[j]
            grady[i] += dy*id2*charge[j]
            gradx[i] += (nx[j]*(dx*dx - dy*dy) + 2*ny[j]*dx*dy)*id4*dipstr[j]
            grady[i] += (nx[j]*2*dx*dy + ny[j]*(dy*dy - dx*dx))*id4*dipstr[j]

def Laplace_Kernel_Apply_numba(source, target, charge=None, dipstr=None,
                    dipvec=None, weights=None, gradient=False, dtype=float):
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5*weights/np.pi
    sx = source[0]
    sy = source[1]
    tx = target[0]
    ty = target[1]
    code = 0
    pot = np.zeros(target.shape[1], dtype=dtype)
    if charge is not None:
        code += 1
        ch = charge*weighted_weights
    if dipstr is not None:
        code += 2
        ds = dipstr*weighted_weights
        nx = dipvec[0]
        ny = dipvec[1]
    if gradient:
        gradx = np.zeros(target.shape[1], dtype=dtype)
        grady = np.zeros(target.shape[1], dtype=dtype)
        if code == 1:
            _Laplace_Kernel_Apply_numba_charge_gradient(sx, sy, tx, ty, ch, pot,
                                                                gradx, grady)
        if code == 2:
            _Laplace_Kernel_Apply_numba_dipole_gradient(sx, sy, tx, ty, ds, 
                                                    nx, ny, pot, gradx, grady)
        if code == 3:
            _Laplace_Kernel_Apply_numba_both_gradient(sx, sy, tx, ty, ch, ds, 
                                                    nx, ny, pot, gradx, grady)
        return pot, gradx, grady
    else:
        if code == 1:
            _Laplace_Kernel_Apply_numba_charge(sx, sy, tx, ty, ch, pot)
        if code == 2:
            _Laplace_Kernel_Apply_numba_dipole(sx, sy, tx, ty, ds, nx, ny, pot)
        if code == 3:
            _Laplace_Kernel_Apply_numba_both(sx, sy, tx, ty, ch, ds, nx, ny,
                                                                            pot)
        return pot

def Laplace_Kernel_Apply_FMM(source, target, charge=None, dipstr=None,
                    dipvec=None, weights=None, gradient=False, dtype=float):
    kind = 'laplace' if dtype is float else 'laplace-complex'
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5*weights/np.pi
    ch = charge*weighted_weights if charge is not None else None
    ds = dipstr*weighted_weights if dipstr is not None else None
    out = FMM(kind=kind, source=source, target=target, charge=ch, dipstr=ds,
                dipvec=dipvec, compute_target_potential=True,
                compute_target_gradient=gradient)['target']
    if gradient:
        return out['u'], out['u_x'], out['u_y']
    else:
        return out['u']

Laplace_Kernel_Applys = {}
Laplace_Kernel_Applys['numba'] = Laplace_Kernel_Apply_numba
Laplace_Kernel_Applys['FMM']   = Laplace_Kernel_Apply_FMM

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
    scale = -1.0/(2*np.pi)
    SX = source[0]
    SY = source[1]
    TX = target[0][:,None]
    TY = target[1][:,None]
    if dipvec is not None:
        nx = dipvec[0]
        ny = dipvec[1]
    if weights is not None:
        scale *= weights
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
        G1 = np.empty_like(dx)
        if ifcharge:
            # charges effect on potential
            ne.evaluate('-0.5*log(id2)', out=G1)
            if chweight is not None:
                ne.evaluate('G1*chweight', out=G1)
            ne.evaluate('G+G1', out=G)
        if ifdipole:
            # dipoles effect on potential
            ne.evaluate('-(nx*dx + ny*dy)*id2', out=G1)
            if dpweight is not None:
                ne.evaluate('G1*dpweight', out=G1)
            ne.evaluate('G+G1', out=G)
        ne.evaluate('G*scale', out=G)
        if gradient:
            Gx1 = G1
            Gy1 = np.empty_like(dx)
            if ifcharge:
                # charges effect on gradient
                ne.evaluate('dx*id2', out=Gx1)
                ne.evaluate('dy*id2', out=Gy1)
                if chweight is not None:
                    ne.evaluate('Gx1*chweight', out=Gx1)
                    ne.evaluate('Gy1*chweight', out=Gy1)
                ne.evaluate('Gx+Gx1', out=Gx)
                ne.evaluate('Gy+Gy1', out=Gy)
            if ifdipole:
                # dipoles effect on gradient
                id4 = ne.evaluate('id2*id2')
                ne.evaluate('(nx*(dx*dx-dy*dy) + 2*ny*dx*dy)*id4', out=Gx1)
                ne.evaluate('(2*nx*dx*dy + ny*(dy*dy-dx*dx))*id4', out=Gy1)
                if dpweight is not None:
                    ne.evaluate('Gx1*dpweight', out=Gx1)
                    ne.evaluate('Gy1*dpweight', out=Gy1)
                ne.evaluate('Gx+Gx1', out=Gx)
                ne.evaluate('Gy+Gy1', out=Gy)
            ne.evaluate('Gx*scale', out=Gx)
            ne.evaluate('Gy*scale', out=Gy)
            return G, Gx, Gy
        else:
            return G


################################################################################
# General Purpose Source --> Source Kernel Calls
# These are naive quadratures with no self interaction!

def Laplace_Kernel_Self_Apply(source, charge=None, dipstr=None, dipvec=None,
                                    weights=None, dtype=float, backend='fly'):
    """
    Laplace Kernel Apply (potential) in 2D
    Computes the sum:
        u_i = sum_j[G_ij charge_j + (dipvec_j dot grad G_ij) dipstr_j] weights_j
            for i != j
    where:

    Parameters:
        source,   required, float(2, ns),  source coordinates
        charge,   optional, dtype(ns),     charge
        dipstr,   optional, dtype(ns),     dipole strength
        dipvec,   optional, float(2, ns),  dipole orientations
        weights,  optional, float(ns),     quadrature weights
        dtype,    optional, float/complex, whether this transform is for float
                                            only, or complex densities
        backend,  optional, str,           'fly', 'numba', 'fmm'
    """
    backend = get_backend(source.shape[1], source.shape[1], backend)
    return Laplace_Kernel_Self_Applys[backend](source, charge, dipstr,
                                                    dipvec, weights, dtype)

@numba.njit(parallel=True)
def _Laplace_Kernel_Self_Apply_numba_charge(sx, sy, charge, pot):
    scale = -0.5/np.pi
    for i in numba.prange(sx.shape[0]):
        for j in range(i):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            d2 = dx**2 + dy**2
            pot[i] += 0.5*np.log(d2)*charge[j]
        for j in range(i+1,sx.shape[0]):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            d2 = dx**2 + dy**2
            pot[i] += 0.5*np.log(d2)*charge[j]
@numba.njit(parallel=True)
def _Laplace_Kernel_Self_Apply_numba_dipole(sx, sy, dipstr, nx, ny, pot):
    for i in numba.prange(sx.shape[0]):
        for j in range(i):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] -= n_dot_d*id2*dipstr[j]
        for j in range(i+1,sx.shape[0]):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] -= n_dot_d*id2*dipstr[j]
@numba.njit(parallel=True)
def _Laplace_Kernel_Self_Apply_numba_both(sx, sy, charge, dipstr, nx, ny, pot):
    for i in numba.prange(sx.shape[0]):
        for j in range(i):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] += 0.5*np.log(d2)*charge[j]
            pot[i] -= n_dot_d*id2*dipstr[j]
        for j in range(i+1,sx.shape[0]):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] += 0.5*np.log(d2)*charge[j]
            pot[i] -= n_dot_d*id2*dipstr[j]

def Laplace_Kernel_Self_Apply_numba(source, charge=None, dipstr=None,
                                        dipvec=None, weights=None, dtype=float):
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5*weights/np.pi
    sx = source[0]
    sy = source[1]
    code = 0
    pot = np.zeros(source.shape[1], dtype=dtype)
    if charge is not None:
        code += 1
        ch = charge*weighted_weights
    if dipstr is not None:
        code += 2
        ds = dipstr*weighted_weights
        nx = dipvec[0]
        ny = dipvec[1]
    if code == 1:
        _Laplace_Kernel_Self_Apply_numba_charge(sx, sy, ch, pot)
    if code == 2:
        _Laplace_Kernel_Self_Apply_numba_dipole(sx, sy, ds, nx, ny, pot)
    if code == 3:
        _Laplace_Kernel_Self_Apply_numba_both(sx, sy, ch, ds, nx, ny, pot)
    return pot

def Laplace_Kernel_Self_Apply_FMM(source, charge=None, dipstr=None,
                                    dipvec=None, weights=None, dtype=float):
    kind = 'laplace' if dtype is float else 'laplace-complex'
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5*weights/np.pi
    ch = charge*weighted_weights if charge is not None else None
    ds = dipstr*weighted_weights if dipstr is not None else None
    out = FMM(kind=kind, source=source, target=source, charge=ch, dipstr=ds,
                dipvec=dipvec, compute_source_potential=True)['source']
    return out['u']

Laplace_Kernel_Self_Applys = {}
Laplace_Kernel_Self_Applys['numba'] = Laplace_Kernel_Self_Apply_numba
Laplace_Kernel_Self_Applys['FMM']   = Laplace_Kernel_Self_Apply_FMM

def Laplace_Kernel_Self_Form(source, ifcharge=False, chweight=None,
                    ifdipole=False, dpweight=None, dipvec=None, weights=None):
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
    G = Laplace_Kernel_Form(source, source, ifcharge, chweight, ifdipole,
                                            dpweight, dipvec, weights, False)
    np.fill_diagonal(G, 0.0)
    return G

################################################################################
# Layer evaluation/form functions
#     note that unlike the above functions, these take source and target
#     not as pure coordinate arrays, but of class(boundary_element)

def Laplace_Layer_Form(source, target, ifcharge=False, chweight=None,
                                ifdipole=False, dpweight=None, gradient=False):
    """
    Laplace Layer Evaluation (potential and gradient in 2D)
    Assumes that source is not target (see function Laplace_Layer_Self_Form)

    Parameters:
        source, required, class(boundary_element), source
        target, required, class(boundary_element), target
        ifcharge, optional, bool,  include effect of charge (SLP)
        chweight, optional, float, scalar weight for the SLP portion
        ifdipole, optional, bool,  include effect of dipole (DLP)
        dpweight, optional, float, scalar weight for the DLP portion
        gradient, optional, bool,  whether to compute the gradient matrices
    """
    if source is not target:
        sourcex = source.stacked_boundary_T
        targetx = target.stacked_boundary_T
        dipvec  = source.stacked_normal_T if ifdipole is not None else None
        weights = source.weights
        return Laplace_Kernel_Form(sourcex, targetx, ifcharge, chweight,
            ifdipole, dpweight, dipvec, weights, gradient)
    else:
        raise Exception('To do source-->source evaluations, use the function \
                                                    Laplace_Layer_Self_Form')

def Laplace_Layer_Apply(source, target, charge=None, dipstr=None,
                                    gradient=False, dtype=float, backend='fly'):
    """
    Laplace Layer Evaluation (potential and gradient in 2D)
    Assumes that source is not target (see function Laplace_Layer_Self_Form)

    Parameters:
        source, required, class(boundary_element), source
        target, required, class(boundary_element), target
        charge,   optional, dtype(ns),     charge
        dipstr,   optional, dtype(ns),     dipole strength
        gradient, optional, bool,          compute gradients or not
        dtype,    optional, float/complex, whether this transform is for float
                                            only, or complex densities
        backend,  optional, str,           'fly', 'numba', 'fmm'
    """
    if source is not target:
        sourcex = source.stacked_boundary_T
        targetx = target.stacked_boundary_T
        dipvec  = source.stacked_normal_T if dipstr is not None else None
        weights = source.weights
        return Laplace_Kernel_Apply(sourcex, targetx, charge, dipstr, dipvec,
                                weights, gradient, dtype=float, backend=backend)
    else:
        raise Exception('To do source-->source evaluations, use the function \
                                                    Laplace_Layer_Self_Apply')

################################################################################
# Layer self evaluation/form functions

def Laplace_Layer_Self_Form(source, ifcharge=False, chweight=None,
                            ifdipole=False, dpweight=None, self_type='naive'):
    """
    Laplace Layer Evaluation (potential in 2D)

    Parameters:
        source, required, class(boundary_element), source
        ifcharge, optional, bool,  include effect of charge (SLP)
        chweight, optional, float, scalar weight for the SLP portion
        ifdipole, optional, bool,  include effect of dipole (DLP)
        dpweight, optional, float, scalar weight for the DLP portion
        self_type, optional, string, 'naive' or 'singular'
    """
    if self_type is 'naive':
        sourcex = source.stacked_boundary_T
        dipvec  = source.stacked_normal_T if ifdipole else None
        weights = source.weights
        return Laplace_Kernel_Self_Form(source.stacked_boundary_T, ifcharge,
                                chweight, ifdipole, dpweight, dipvec, weights)
    else:
        ALP = np.zeros([source.N, source.N], dtype=float)
        if ifdipole:
            # form the DLP Matrix
            sourcex = source.stacked_boundary_T
            dipvec  = source.stacked_normal_T
            weights = source.weights
            DLP = Laplace_Kernel_Self_Form(sourcex, ifdipole=True,
                                                dipvec=dipvec, weights=weights)
            # fix the diagonal
            np.fill_diagonal(DLP, -0.25*source.curvature*source.weights/np.pi)
            # weight, if necessary
            if dpweight != None:
                DLP *= dp_weight
            ne.evaluate('ALP + DLP', out=ALP)
        if ifcharge:
            # form the SLP Matrix
            # because this is singular, this depends on the type of layer itself
            # and the SLP formation must be implemented in that class!
            SLP = source.Laplace_SLP_Self_Form()
            # weight, if necessary
            if chweight != None:
                SLP *= chweight
            ne.evaluate('ALP + SLP', out=ALP)
        return ALP

@numba.njit(parallel=True)
def _Laplace_Kernel_Self_Apply_DLP(sx, sy, dipstr, nx, ny, pot, curvature,
                                                                    weights):
    for i in numba.prange(sx.shape[0]):
        for j in range(i):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] -= n_dot_d*id2*dipstr[j]
        for j in range(i+1,sx.shape[0]):
            dx = sx[i] - sx[j]
            dy = sy[i] - sy[j]
            d2 = dx**2 + dy**2
            id2 = 1.0/d2
            n_dot_d = nx[j]*dx + ny[j]*dy
            pot[i] -= n_dot_d*id2*dipstr[j]
        pot[i] -= 0.25*curvature*weights[i]/np.pi

def Laplace_Layer_Self_Apply(source, charge=None, dipstr=None,
                                dtype=float, self_type='naive', backend='fly'):
    """
    Laplace Layer Self Evaluation (potential in 2D)

    Parameters:
        source, required, class(boundary_element), source
        charge,   optional, dtype(ns),     charge
        dipstr,   optional, dtype(ns),     dipole strength
        dtype,    optional, float/complex, whether this transform is for float
                                            only, or complex densities
        self_type, optional, string, 'naive' or 'singular'
        backend,  optional, str,           'fly', 'numba', 'fmm'
    """
    if self_type is 'naive':
        sourcex = source.stacked_boundary_T
        dipvec  = source.stacked_normal_T
        weights = source.weights
        return Laplace_Kernel_Self_Apply(sourcex, charge, dipstr, dipvec,
                                                    weights, dtype, backend)
    else:
        uALP = np.zeros([source.N,], dtype=dtype)
        if dipstr is not None:
            # evaluate the DLP
            sourcex = source.stacked_boundary_T
            dipvec  = source.stacked_normal_T
            weights = source.weights
            uDLP = Laplace_Kernel_Self_Apply(sourcex, dipstr=dipstr,
                dipvec=dipvec, weights=weights, dtype=dtype, backend=backend)
            uDLP -= 0.25*source.curvature*weights/np.pi*dipstr
            ne.evaluate('uALP+uDLP', out=uALP)
        if charge is not None:
            # form the SLP Matrix
            # because this is singular, this depends on the type of layer itself
            # and the SLP formation must be implemented in that class!
            uSLP = source.Laplace_SLP_Self_Apply(charge)
            ne.evaluate('uALP+uSLP', out=uALP)
        return uALP

