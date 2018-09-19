"""
This submodule provides higher-level wrappers for the Stokes Kernel Functions
"""

import numpy as np
import numexpr as ne
import numba
import warnings

from ...backend_defaults import get_backend
from ... import have_fmm
if have_fmm:
    from ... import FMM

from ..low_level.stokes import Stokes_Kernel_Apply, Stokes_Kernel_Form

################################################################################
# Applies

def check_and_convert(x, bdy):
    """
    utility function to convert sources between linear/stacked forms
    """
    if x is not None and len(x.shape) == 1:
        return x.reshape(2, bdy.N)
    else:
        return x

def Stokes_Layer_Apply(source, target=None, forces=None, dipstr=None,
                                            backend='fly', out_type='flat'):
    """
    Stokes Layer Apply

    Parameters:
        source,   required, Boundary,       source
        target,   optional, PointSet,       target
        forces,   optional, float(2, ns),   forces
        dipstr,   optional, float(2, ns),   dipole strength
        weights,  optional, float(ns),      weights
        backend,  optional, str,            'fly', 'numba', 'FMM'
        out_type, optional, str,            'flat' or 'stacked'

    forces/dipstr can also be given as float(2*ns)

    If source is not target, then this function assumes that source and
        target have no coincident points
    If source is target, this function computes a naive quadrature,
        ignoring the i=j term in the sum
    """
    forces = check_and_convert(forces, source)
    dipstr = check_and_convert(dipstr, source)
    dipvec = None if dipstr is None else source.get_stacked_normal(T=True)
    if target is None:
        target = source
    backend = get_backend(source.N, target.N, backend)
    out = Stokes_Kernel_Apply(
                source  = source.get_stacked_boundary(T=True),
                target  = target.get_stacked_boundary(T=True),
                forces  = forces,
                dipstr  = dipstr,
                dipvec  = dipvec,
                weights = source.weights,
                backend = backend,
            )
    if out_type == 'flat':
        return out.reshape(2*target.N)
    else:
        return out

def Stokes_Layer_Singular_Apply(source, forces=None, dipstr=None,
                                                                backend='fly'):
    """
    Stokes Layer Singular Apply

    Parameters:
        source,   required, Boundary,       source
        forces,   optional, float(2, ns),   forces
        dipstr,   optional, float(2, ns),   dipole strength
        weights,  optional, float(ns),      weights
        backend,  optional, str,            'fly', 'numba', 'FMM'

    forces/dipstr can also be given as float(2*ns)
    """
    forces = check_and_convert(forces, source)
    dipstr = check_and_convert(dipstr, source)
    uALP = np.zeros([2, source.N], dtype=float)
    if dipstr is not None:
        # evaluate the DLP
        uDLP = Stokes_Layer_Apply(source, dipstr=dipstr, backend=backend)
        tx = source.tangent_x
        ty = source.tangent_y
        scale = -0.5*source.curvature*source.weights/np.pi
        s01 = scale*tx*ty
        uDLP[0] += (scale*tx*tx*dipstr[0] + s01*dipstr[1])
        uDLP[1] += (s01*dipstr[0] + scale*ty*ty*dipstr[1])
        ne.evaluate('uALP+uDLP', out=uALP)
    if forces is not None:
        # form the SLP Matrix
        # because this is singular, this depends on the type of layer itself
        # and the SLP formation must be implemented in that class!
        backend = get_backend(source.N, source.N, backend)
        uSLP = source.Stokes_SLP_Self_Apply(forces, backend=backend)
        ne.evaluate('uALP+uSLP', out=uALP)
    if out_type == 'flat':
        return uALP.reshape(2*source.N)
    else:
        return uALP

################################################################################
# Formations

def Stokes_Layer_Form(source, target=None, ifforce=False, fweight=None,
                                                ifdipole=False, dpweight=None):
    """
    Stokes Layer Evaluation (potential and gradient in 2D)

    Parameters:
        source,   required, Boundary, source
        target,   optional, Boundary, target
        ifforce,  optional, bool,  include effect of force (SLP)
        fweight,  optional, float, scalar weight for the SLP portion
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
    return Stokes_Kernel_Form(
            source   = source.get_stacked_boundary(T=True),
            target   = target.get_stacked_boundary(T=True),
            ifforce  = ifforce,
            fweight  = fweight,
            ifdipole = ifdipole,
            dpweight = dpweight,
            dipvec   = dipvec,
            weights  = source.weights,
        )

def Stokes_Layer_Singular_Form(source, ifforce=False, fweight=None,
                                                ifdipole=False, dpweight=None):
    """
    Stokes Layer Singular Form

    Parameters:
        source,   required, Boundary,   source
        ifforce,  optional, bool,       include effect of force (SLP)
        fweight,  optional, float,      scalar weight for the SLP portion
        ifdipole, optional, bool,       include effect of dipole (DLP)
        dpweight, optional, float,      scalar weight for the DLP portion
    """
    sn = source.N
    ALP = np.zeros([2*sn, 2*sn], dtype=float)
    if ifdipole:
        # form the DLP Matrix
        DLP = Stokes_Layer_Form(source, ifdipole=True)
        # fix the diagonal
        scale = -0.5*source.curvature*source.weights/np.pi
        tx = source.tangent_x
        ty = source.tangent_y
        s01 = scale*tx*ty
        np.fill_diagonal(DLP[:sn, :sn], scale*tx*tx)
        np.fill_diagonal(DLP[sn:, :sn], s01)
        np.fill_diagonal(DLP[:sn, sn:], s01)
        np.fill_diagonal(DLP[sn:, sn:], scale*ty*ty)
        # weight, if necessary, and add to ALP
        if dpweight is None:
            ne.evaluate('ALP + DLP', out=ALP)
        else:
            ne.evaluate('ALP + DLP*dpweight', out=ALP)
    if ifforce:
        # form the SLP Matrix
        # because this is singular, this depends on the type of layer itself
        # and the SLP formation must be implemented in that class!
        SLP = source.Stokes_SLP_Self_Form()
        # weight, if necessary, and add to ALP
        if fweight is None:
            ne.evaluate('ALP + SLP', out=ALP)
        else:
            ne.evaluate('ALP + SLP*fweight', out=ALP)
    return ALP
