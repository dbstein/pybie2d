"""
This submodule provides higher-level wrappers for the Cauchy Kernel Functions
"""

import numpy as np
import numexpr as ne
import numba
import warnings

from ...backend_defaults import get_backend
from ... import have_fmm
if have_fmm:
    from ... import FMM

from ..low_level.cauchy import Cauchy_Kernel_Apply, Cauchy_Kernel_Form

################################################################################
# Applies

def Cauchy_Layer_Apply(source, target=None, dipstr=None, backend='fly'):
    """
    Cauchy Layer Apply
    Computes the sum:
        u_i = 1/(2i*pi) * sum_j cw_j*p_j/(z_j-z_i) for i != j
    where:

    Parameters:
        source,   required, Boundary,      source coordinates (z_j)
        target,   optional, PointSet,      target coordinates (z_i)
        dipstr,   optional, complex(ns),   dipole strength (p_j)
        backend,  optional, str,           'fly', 'numba', 'FMM'
            cw_j are the complex_weights in the boundary source

    If source is not target, then this function assumes that source and
        target have no coincident points
    If source is target, this function computes a naive quadrature,
        ignoring the i=j term in the sum
    """
    if target is None:
        target = source
    backend = get_backend(source.N, source.N, backend)
    return Cauchy_Kernel_Apply(
                source  = source.c,
                target  = target.c,
                dipstr  = dipstr,
                weights = source.complex_weights,
                backend = backend,
            )

def Cauchy_Layer_Form(source, target=None):
    """
    Cauchy Layer Form

    Parameters:
        source, required, class(boundary_element), source
        target, optional, class(boundary_element), target
    If source is not target, then this function assumes that source and
        target have no coincident points
    If source is target, this function computes a naive quadrature,
        ignoring the i=j term in the sum
    """
    if target is None:
        target = source
    return Cauchy_Kernel_Form(
                source  = source.c,
                target  = target.c,
                weights = source.complex_weights,
            )
