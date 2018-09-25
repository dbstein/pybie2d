"""
This submodule provides higher-level wrappers for the Modified Helmholtz Kernel Functions
"""

import numpy as np
import numexpr as ne
import numba
import warnings

from ...backend_defaults import get_backend
from ... import have_fmm
if have_fmm:
    from ... import FMM

from ..low_level.modified_helmholtz import Modified_Helmholtz_Kernel_Apply
from ..low_level.modified_helmholtz import Modified_Helmholtz_Kernel_Form

################################################################################
# Applies

def Modified_Helmholtz_Layer_Apply(source, target=None, k=1.0, charge=None,
                                                    dipstr=None, backend='fly'):
    """
    Laplace Layer Apply (potential and gradient) in 2D
    Computes the sum:
        u_i = sum_j[G_ij charge_j + (dipvec_j dot grad G_ij) dipstr_j] weights_j
    and appropriate derivatives, where:

    Parameters:
        source,   required, Boundary,       source
        target,   optional, PointSet,       target
        charge,   optional, dtype(ns),      charge
        dipstr,   optional, dtype(ns),      dipole strength
        backend,  optional, str,            'fly', 'numba', 'FMM'

    If source is not target, then this function assumes that source and
        target have no coincident points
    If source is target, this function computes a naive quadrature,
        ignoring the i=j term in the sum
    """
    dipvec = None if dipstr is None else source.get_stacked_normal(T=True)
    if target is None:
        target = source
    backend = get_backend(source.N, source.N, backend)
    return Modified_Helmholtz_Kernel_Apply(
                source   = source.get_stacked_boundary(T=True),
                target   = target.get_stacked_boundary(T=True),
                k        = k,
                charge   = charge,
                dipstr   = dipstr,
                dipvec   = dipvec,
                weights  = source.weights,
                backend  = backend,
            )

################################################################################
# Formations

def Modified_Helmholtz_Layer_Form(source, target=None, k=1.0, ifcharge=False,
                                                                ifdipole=False):
    """
    Laplace Layer Evaluation (potential and gradient in 2D)
    Assumes that source is not target (see function Laplace_Layer_Self_Form)

    Parameters:
        source,   required, Boundary, source
        target,   optional, Boundary, target
        ifcharge, optional, bool,  include effect of charge (SLP)
        chweight, optional, float, scalar weight for the SLP portion
        ifdipole, optional, bool,  include effect of dipole (DLP)
        dpweight, optional, float, scalar weight for the DLP portion

    If source is not target, then this function assumes that source and
        target have no coincident points
    If source is target, this function computes a naive quadrature,
        ignoring the i=j term in the sum
    """
    dipvec = None if ifdipole is None else source.get_stacked_normal(T=True)
    if target is None:
        target = source
    return Modified_Helmholtz_Kernel_Form(
            source   = source.get_stacked_boundary(T=True),
            target   = target.get_stacked_boundary(T=True),
            k        = k,
            ifcharge = ifcharge,
            ifdipole = ifdipole,
            dipvec   = dipvec,
            weights  = source.weights,
        )
