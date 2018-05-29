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
def _CKAND(s, t, dipstr, pot):
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
    for i in numba.prange(t.shape[0]):
        for j in range(s.shape[0]):
            if t[i] != s[j]: # this is ugly! (should be handled not here!)
                pot[i] += dipstr[j]/(t[i]-s[j])
            else:
                pot[i] += np.Inf

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
    _CKAND(source, target, ds, pot)
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
# General Purpose Source --> Source Kernel Calls
# These are not singular quadratures!  They merely ignore the self interaction

@numba.njit(parallel=True)
def _CKSAND(s, dipstr, pot):
    """
    Numba-jitted Cauchy Kernel (self, naive quadrature)
    Inputs:
        s,      intent(in),  complex(ns), coordinates of source
        dipstr, intent(in),  complex(ns), dipole strength at source locations
        pot,    intent(out), complex(nt), potential at target locations
    ns = number of source points
    all inputs are required

    This function should generally not be called direclty
    Instead call through the "Cauchy_Kernel_Self_Apply_numba" interface
    """
    for i in range(s.shape[0]):
        pot[i] = 0.0
    for i in numba.prange(s.shape[0]):
        for j in range(i):
            pot[i] += dipstr[j]/(s[i]-s[j])
        for j in range(i+1, s.shape[0]):
            pot[i] += dipstr[j]/(s[i]-s[j])

def Cauchy_Kernel_Self_Apply_numba(source, dipstr, weights=None):
    """
    Interface to numba-jitted Cauchy Kernel (self, naive quadrature)
    Inputs:
        source,   required, complex(ns), coordinates of source
        dipstr,   required, complex(ns), dipole strength at source locations
        weights,  optional, float(ns),   quadrature weights
    Outputs:
        complex(ns), potential at target coordinates
    ns = number of source points
    """
    weights = 1.0 if weights is None else weights
    weighted_weights = -0.5j*weights/np.pi
    pot = np.empty(source.shape[0], dtype=complex)
    ds = dipstr*weighted_weights
    _CKSAND(source, ds, pot)
    return pot

def Cauchy_Kernel_Self_Apply_FMM(source, dipstr, weights=None):
    """
    Interface to FMM Cauchy Kernel (self, naive quadrature)
    Inputs:
        source,   required, complex(ns), coordinates of source
        dipstr,   required, complex(ns), dipole strength at source locations
        weights,  optional, float(ns),   quadrature weights
    Outputs:
        complex(ns), potential at target coordinates
    ns = number of source points
    """
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

def Cauchy_Kernel_Self_Apply(source, dipstr, weights=None, backend='numba'):
    """
    Interface to FMM Cauchy Kernel (self, naive quadrature)
    Inputs:
        source,   required, complex(ns), coordinates of source
        dipstr,   required, complex(ns), dipole strength at source locations
        weights,  optional, float(ns),   quadrature weights
        backend,  optional, str,         backend ('FMM' or 'numba')
    Outputs:
        complex(ns), potential at target coordinates
    ns = number of source points
    """
    return Cauchy_Kernel_Self_Applys[backend](source, dipstr, weights)


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
    return ne.evaluate('scale/(T-source)')

################################################################################
# General Purpose Low Level Source --> Source Kernel Formation
# These are naive quadratures with no self interaction!
# only potentials are currently implemented for self-interaction

def Cauchy_Kernel_Self_Form(source, weights=None):
    """
    Cauchy Kernel Formation
    Computes the matrix:
        1/(2i*pi) * w_j/(z_j-z_i) for i != j
        0                         for i == j

    Parameters:
        source,   required, complex(ns),   source coordinates (z_j)
        weights,  optional, numeric(ns),   quadrature weights (w_j)
    """
    G = Cauchy_Kernel_Form(source, source, weights)
    np.fill_diagonal(G, 0.0)
    return G
