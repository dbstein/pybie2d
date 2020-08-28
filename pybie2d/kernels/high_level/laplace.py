"""
This submodule provides higher-level wrappers for the Laplace Kernel Functions
"""

import numpy as np
import numexpr as ne
import numba
import warnings

from ...backend_defaults import get_backend
from ... import have_fmm
if have_fmm:
    from ... import FMM

from ..low_level.laplace import Laplace_Kernel_Apply, Laplace_Kernel_Form

################################################################################
# Applies

def Laplace_Layer_Apply(source, target=None, charge=None, dipstr=None,
                                                gradient=False, backend='fly', **kwargs):
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
        gradient, optional, bool,           compute gradients or not
            gradient only implemented for source-->target
        backend,  optional, str,            'fly', 'numba', 'FMM'

    If source is not target, then this function assumes that source and
        target have no coincident points
    If source is target, this function computes a naive quadrature,
        ignoring the i=j term in the sum
    """
    dipvec = None if dipstr is None else source.get_stacked_normal(T=True)
    if target is None:
        target = source
    backend = get_backend(source.N, target.N, backend)
    return Laplace_Kernel_Apply(
                source   = source.get_stacked_boundary(T=True),
                target   = target.get_stacked_boundary(T=True),
                charge   = charge,
                dipstr   = dipstr,
                dipvec   = dipvec,
                weights  = source.weights,
                gradient = gradient,
                backend  = backend,
                **kwargs,
            )

def Laplace_Layer_Singular_Apply(source, charge=None, dipstr=None,
                                                                backend='fly'):
    """
    Laplace Layer Singular Apply (potential only)

    Parameters:
        source,   required, Boundary,       source
        charge,   optional, dtype(ns),      charge
        dipstr,   optional, dtype(ns),      dipole strength
        backend,  optional, str,            'fly', 'numba', 'FMM'
    """
    uALP = np.zeros([source.N,], dtype=float)
    if dipstr is not None:
        # evaluate the DLP
        uDLP = Laplace_Layer_Apply(source, dipstr=dipstr, backend=backend)
        uDLP -= 0.25*source.curvature*source.weights/np.pi*dipstr
        ne.evaluate('uALP+uDLP', out=uALP)
    if charge is not None:
        # form the SLP Matrix
        # because this is singular, this depends on the type of layer itself
        # and the SLP formation must be implemented in that class!
        backend = get_backend(source.N, source.N, backend)
        uSLP = source.Laplace_SLP_Self_Apply(charge, backend=backend)
        ne.evaluate('uALP+uSLP', out=uALP)
    return uALP

################################################################################
# Formations

def Laplace_Layer_Form(source, target=None, ifcharge=False, chweight=None,
                                ifdipole=False, dpweight=None, gradient=False):
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
        gradient, optional, bool,  whether to compute the gradient matrices
            gradient only implemented for source-->target

    If source is not target, then this function assumes that source and
        target have no coincident points
    If source is target, this function computes a naive quadrature,
        ignoring the i=j term in the sum
    """
    dipvec = None if ifdipole is None else source.get_stacked_normal(T=True)
    if target is None:
        target = source
    return Laplace_Kernel_Form(
            source   = source.get_stacked_boundary(T=True),
            target   = target.get_stacked_boundary(T=True),
            ifcharge = ifcharge,
            chweight = chweight,
            ifdipole = ifdipole,
            dpweight = dpweight,
            dipvec   = dipvec,
            weights  = source.weights,
            gradient = gradient,
        )

def Laplace_Layer_Singular_Form(source, ifcharge=False, chweight=None,
                                                ifdipole=False, dpweight=None):
    """
    Laplace Layer Singular Form (potential only)

    Parameters:
        source,   required, Boundary,   source
        ifcharge, optional, bool,       include effect of charge (SLP)
        chweight, optional, float,      scalar weight for the SLP portion
        ifdipole, optional, bool,       include effect of dipole (DLP)
        dpweight, optional, float,      scalar weight for the DLP portion
    """
    ALP = np.zeros([source.N, source.N], dtype=float)
    if ifdipole:
        # form the DLP Matrix
        DLP = Laplace_Layer_Form(source, ifdipole=True)
        # fix the diagonal
        np.fill_diagonal(DLP, -0.25*source.curvature*source.weights/np.pi)
        # weight, if necessary, and add to ALP
        if dpweight == None:
            ne.evaluate('ALP + DLP', out=ALP)
        else:
            ne.evaluate('ALP + DLP*dpweight', out=ALP)
    if ifcharge:
        # form the SLP Matrix
        # because this is singular, this depends on the type of layer itself
        # and the SLP formation must be implemented in that class!
        SLP = source.Laplace_SLP_Self_Form()
        # weight, if necessary, and add to ALP
        if chweight == None:
            ne.evaluate('ALP + SLP', out=ALP)
        else:
            ne.evaluate('ALP + SLP*chweight', out=ALP)
    return ALP
