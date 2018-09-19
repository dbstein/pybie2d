import numpy as np
import numexpr as ne
import numba
import warnings

from ... import have_fmm
if have_fmm:
    from ... import FMM

################################################################################
# General Purpose Source --> Target Kernel Apply Functions

@numba.njit(parallel=True)
def _cauchy(s, t, dipstr, pot):
    """
    Numba-jitted Cauchy Kernel
    Inputs:
        s,      intent(in),  complex(ns), coordinates of source
        t,      intent(in),  complex(nt), coordinates of target
        dipstr, intent(in),  complex(ns), dipole strength at source locations
        pot,    intent(out), complex(nt), potential at target locations
    ns = number of source points; nt = number of target points
    all inputs are required

    This function should generally not be called direclty
    Instead call through the "Cauchy_Kernel_Apply_numba" interface
    """
    doself = s is t
    for i in numba.prange(t.shape[0]):
        for j in range(s.shape[0]):
            # if not (doself and i == j):
            pot[i] += dipstr[j]/(t[i]-s[j])
            # if t[i] != s[j]: # this is ugly! (should be handled not here!)
            #     pot[i] += dipstr[j]/(t[i]-s[j])
            # else:
            #     pot[i] += np.Inf

def Cauchy_Kernel_Apply_numba(source, target, dipstr, weights=None):
    """
    Interface to numba-jitted Cauchy Kernel
    Inputs:
        source,   required, complex(ns), coordinates of source
        target,   required, complex(nt), coordinates of target
        dipstr,   required, complex(ns), dipole strength at source locations
        weights,  optional, float(ns),   quadrature weights
    Outputs:
        complex(nt), potential at target coordinates
    ns = number of source points; nt = number of target points
    """
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5j*weights/np.pi
    pot = np.zeros(target.shape[0], dtype=complex)
    ds = dipstr*weighted_weights
    _cauchy(source, target, ds, pot)
    return pot

def Cauchy_Kernel_Apply_FMM(source, target, dipstr, weights=None):
    """
    Interface to FMM Cauchy Kernel
    Inputs:
        source,   required, complex(ns), coordinates of source
        target,   required, complex(nt), coordinates of target
        dipstr,   required, complex(ns), dipole strength at source locations
        weights,  optional, float(ns),   quadrature weights
    Outputs:
        complex(nt), potential at target coordinates
    ns = number of source points; nt = number of target points
    """
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5j*weights/np.pi
    ds = dipstr*weighted_weights
    src = np.row_stack([source.real, source.imag])
    trg = np.row_stack([target.real, target.imag])
    if source is target:
        out = FMM(kind='cauchy', source=src, dipstr=ds,
                                    compute_source_potential=True)['source']
    else:
        out = FMM(kind='cauchy', source=src, target=trg, dipstr=ds,
                                    compute_target_potential=True)['target']
    return out['u']

Cauchy_Kernel_Applys = {}
Cauchy_Kernel_Applys['numba'] = Cauchy_Kernel_Apply_numba
Cauchy_Kernel_Applys['FMM']   = Cauchy_Kernel_Apply_FMM

def Cauchy_Kernel_Apply(source, target, dipstr, weights=None, backend='numba'):
    """
    Interface to Cauchy Kernel
    Inputs:
        source,   required, complex(ns), coordinates of source
        target,   required, complex(nt), coordinates of target
        dipstr,   required, complex(ns), dipole strength at source locations
        weights,  optional, float(ns),   quadrature weights
        backend,  optional, str,         backend ('FMM' or 'numba')
    Outputs:
        complex(nt), potential at target coordinates
    ns = number of source points; nt = number of target points
    """
    return Cauchy_Kernel_Applys[backend](source, target, dipstr, weights)

################################################################################
# General Purpose Low Level Source --> Target Kernel Formation

def Cauchy_Kernel_Form(source, target, weights=None):
    """
    Cauchy Kernel Formation
    Computes the matrix:
        1/(2i*pi) * w_j/(z_j-z_i)

    Inputs:
        source,   required, complex(ns),   source coordinates (z_j)
        target,   required, complex(nt),   target coordinates (z_i)
        weights,  optional, numeric(ns),   quadrature weights (w_j)

    This function assumes that source and target have no coincident points
    """
    scale = -0.5j/np.pi
    T = target[:,None]
    if weights is not None:
        scale *= weights
    G = ne.evaluate('scale/(T-source)')
    if source is target:
        np.fill_diagonal(G, 0.0)
    return G
