import numpy as np
import numexpr as ne
import numba
import warnings

from ..backend_defaults import get_backend
from .. import have_fmm
if have_fmm:
    from .. import FMM


################################################################################
# General Purpose Source --> Source Kernel Calls
# These are not singular quadratures!  They merely ignore the self interaction

def Cauchy_Kernel_Self_Apply(source, dipstr, weights=None, backend='fly'):
    """
    Cauchy Kernel Apply in 2D
    Computes the sum:
        u_i = 1/(2i*pi) * sum_j cw_j*p_j/(z_j-z_i) for i != j
    where:

    Parameters:
        source,   required, complex(ns),   source coordinates (z_j)
        dipstr,   optional, complex(ns),   dipole strength (p_j)
        weights,  optional, complex(ns),   complex quadrature weights (cw_j)
        backend,  optional, str,           'fly', 'numba', 'fmm'

        This function assumes that source and target have no coincident points
    """
    backend = get_backend(source.shape[0], source.shape[0], backend)
    return Cauchy_Kernel_Self_Applys[backend](source, dipstr, weights)

@numba.njit(parallel=True)
def _Cauchy_Kernel_Self_Apply_numba_dipole(s, dipstr, pot):
    for i in range(s.shape[0]):
        pot[i] = 0.0
    for i in numba.prange(s.shape[0]):
        for j in range(i):
            pot[i] += dipstr[j]/(s[i]-s[j])
        for j in range(i+1, s.shape[0]):
            pot[i] += dipstr[j]/(s[i]-s[j])

def Cauchy_Kernel_Self_Apply_numba(source, dipstr, weights=None):
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5j*weights/np.pi
    pot = np.empty(source.shape[0], dtype=complex)
    ds = dipstr*weighted_weights
    _Cauchy_Kernel_Self_Apply_numba_dipole(source, ds, pot)
    return pot

def Cauchy_Kernel_Self_Apply_FMM(source, dipstr, weights=None):
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5j*weights/np.pi
    ds = dipstr*weighted_weights
    src = np.row_stack([source.real, source.imag])
    # note the cauchy FMM hangs if not given a target...
    # is this my wrapper or the code?
    out = FMM(kind='cauchy', source=src, target=src, dipstr=ds,
                                    compute_source_potential=True)['source']
    return out['u']

Cauchy_Kernel_Self_Applys = {
    'numba' : Cauchy_Kernel_Self_Apply_numba,
    'FMM'   : Cauchy_Kernel_Self_Apply_FMM,
}

def Cauchy_Kernel_Self_Form(source, weights=None):
    """
    Cauchy Kernel Formation
    Computes the matrix:
        1/(2i*pi) * cw_j/(z_j-z_i) for i != j

    Parameters:
        source,   required, complex(ns),   source coordinates (z_j)
        weights,  optional, complex(ns),   complex quadrature weights (cw_j)
        weights,  optional, complex(ns),   quadrature weights
    """
    scale = -0.5j/np.pi
    T = source[:,None]
    if weights is not None:
        scale *= weights
    G = ne.evaluate('scale/(T-source)')
    np.fill_diagonal(G, 0.0)
    return G

################################################################################
# General Purpose Source --> Target Kernel Calls

def Cauchy_Kernel_Apply(source, target, dipstr, weights=None, backend='fly'):
    """
    Cauchy Kernel Apply (potential and gradient) in 2D
    Computes the sum:
        u_i = 1/(2i*pi) * sum_j cw_j*p_j/(z_j-z_i)
    where:

    Parameters:
        source,   required, complex(ns),   source coordinates (z_j)
        target,   required, complex(nt),   target coordinates (z_i)
        dipstr,   optional, complex(ns),   dipole strength (p_j)
        weights,  optional, complex(ns),   complex quadrature weights (cw_j)
        backend,  optional, str,           'fly', 'numba', 'fmm'

        This function assumes that source and target have no coincident points
    """
    backend = get_backend(source.shape[0], target.shape[0], backend)
    return Cauchy_Kernel_Applys[backend](source, target, dipstr, weights)

@numba.njit(parallel=True)
def _Cauchy_Kernel_Apply_numba_dipole(s, t, dipstr, pot):
    for i in numba.prange(t.shape[0]):
        for j in range(s.shape[0]):
            if t[i] != s[j]: # this is ugly!
                pot[i] += dipstr[j]/(t[i]-s[j])
            else:
                pot[i] += np.Inf

def Cauchy_Kernel_Apply_numba(source, target, dipstr, weights=None):
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5j*weights/np.pi
    pot = np.zeros(target.shape[0], dtype=complex)
    ds = dipstr*weighted_weights
    _Cauchy_Kernel_Apply_numba_dipole(source, target, ds, pot)
    return pot

def Cauchy_Kernel_Apply_FMM(source, target, dipstr, weights=None):
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5j*weights/np.pi
    ds = dipstr*weighted_weights
    src = np.row_stack([source.real, source.imag])
    trg = np.row_stack([target.real, target.imag])
    out = FMM(kind='cauchy', source=src, target=trg, dipstr=ds,
                                    compute_target_potential=True)['target']
    return out['u']

Cauchy_Kernel_Applys = {}
Cauchy_Kernel_Applys['numba'] = Cauchy_Kernel_Apply_numba
Cauchy_Kernel_Applys['FMM']   = Cauchy_Kernel_Apply_FMM

def Cauchy_Kernel_Form(source, target, weights=None):
    """
    Cauchy Kernel Formation
    Computes the matrix:
        1/(2i*pi) * cw_j/(z_j-z_i)
    Also returns the matrices for the x and y derivatives, if requested

    Parameters:
        source,   required, complex(ns),   source coordinates (z_j)
        target,   required, complex(nt),   target coordinates (z_i)
        weights,  optional, complex(ns),   complex quadrature weights (cw_j)
        weights,  optional, complex(ns),   quadrature weights

    This function assumes that source and target have no coincident points
    """
    ns = source.shape[0]
    nt = target.shape[0]
    scale = -0.5j/np.pi
    T = target[:,None]
    if weights is not None:
        scale *= weights
    G = ne.evaluate('scale/(T-source)')
    return G

################################################################################
# Layer evaluation/form functions
#     note that unlike the above functions, these take source and target
#     not as pure coordinate arrays, but of class(boundary_element)

def Cauchy_Layer_Form(source, target):
    """
    Cauchy Layer Formation (potential only)
    Calls Cauchy_Kernel_Self_Form if source is target
    Otherwise calls Cauchy_Kernel_Form

    Note that if this is computing a self-quadrature, then it is a naive
    quadrature (ignoring self-interaction)!

    Parameters:
        source, required, class(boundary_element), source
        target, required, class(boundary_element), target
    """
    if source is target:
        return Cauchy_Kernel_Self_Form(source.c, weights=source.complex_weights)
    else:
        return Cauchy_Kernel_Form(source.c, target.c, 
                                        weights=source.complex_weights)

def Cauchy_Layer_Apply(source, target, dipstr, backend='fly'):
    """
    Cauchy Layer Evaluation (potential only)
    Only handles the case where source is not target

    Parameters:
        source,  required, class(boundary_element), source
        target,  required, class(boundary_element), target
        dipstr,  optional, complex(ns),             dipole strength
        backend,  optional, str,                    'fly', 'numba', 'fmm'
    """
    if source is target:
        return Cauchy_Kernel_Self_Apply(source.c, dipstr,
                                weights=source.complex_weights, backend=backend)
    else:
        return Cauchy_Kernel_Apply(source.c, target.c, dipstr, 
                                        source.complex_weights, backend)
