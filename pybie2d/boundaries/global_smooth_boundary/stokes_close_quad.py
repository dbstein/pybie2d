import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os

from ...kernels.high_level.stokes import Stokes_Layer_Form, Stokes_Layer_Apply

class Stokes_Close_Quad(object):
    """
    Module providing Stokes Close Eval based on Globaly Compensated Cauchy Quad
    """
    def __init__(self, GSB):
        """
        Initializes the Close Quad Module

        GSB (required): boundary of type Global_Smooth_Boundary
        """
        self.boundary = GSB
        GSB.add_module('Laplace_Close_Quad')
        self.NF = int(np.ceil(2.2*self.boundary.N)/2.0)*2
        self.fsrc = self.boundary.generate_resampled_boundary(self.NF)
        self.fsrc.add_module('Laplace_Close_Quad')

    def Get_Close_Correction_Function(self, target, side, do_DLP, do_SLP, backend):
        """
        Given target, kernel, and type ('preformed', 'fly', 'numba', or 'fmm'),

        Returns a dictionary 'preparation'
        And a function that takes as parameters tau, preparation that
        is repsonsible for the close correction

        Note that this does not check if target points need close evaluation!
        """
        return Get_Stokes_Close_Correction_Function(self.boundary, target, side,
                                                        do_DLP, do_SLP, backend)

def _Stokes_Close_Correction_Function_Preformed(tau, preparation):
    return preparation['correction_mat'].dot(tau)

def _Stokes_Close_Correction_Function_Apply(tau, preparation):
    v1 = Compensated_Stokes_Apply(
            preparation['source'],
            preparation['target'],
            side       = preparation['side'],
            tau        = tau,
            do_DLP     = preparation['do_DLP'], 
            do_SLP     = preparation['do_SLP'], 
            backend    = preparation['backend']
        )
    ch = tau*int(preparation['do_SLP'])
    ds = tau*int(preparation['do_DLP'])
    v2 = Stokes_Layer_Apply(
            source     = preparation['source'],
            target     = preparation['target'],
            forces     = ch,
            dipstr     = ds,
            backend    = preparation['backend']
        )
    return v1 - v2

def Get_Stokes_Close_Correction_Function(source, target, side, do_DLP, do_SLP, backend):
    if backend == 'preformed':
        close_mat = Compensated_Stokes_Form(source, target, side, do_DLP, do_SLP)
        naive_mat = Stokes_Layer_Form(source, target, ifforce=do_SLP, ifdipole=do_DLP)
        correction_mat = close_mat.real - naive_mat
        preparation = {
            'do_DLP'         : do_DLP,
            'do_SLP'         : do_SLP,
            'correction_mat' : correction_mat,
        }
        return preparation, _Stokes_Close_Correction_Function_Preformed
    else:
        preparation = {
            'source'       : source,
            'target'       : target,
            'side'         : side,
            'do_DLP'       : do_DLP,
            'do_SLP'       : do_SLP,
            'backend'      : backend,
        }
        return preparation, _Stokes_Close_Correction_Function_Apply

def Compensated_Stokes_Form(source, target, side, do_DLP=False, do_SLP=False):
    # arrays that manipulate form of density
    # there's almost certainly a faster way to deal with some of these things
    sh = (source.N, source.N)
    Mx1 = np.zeros(sh)
    np.fill_diagonal(Mx1, source.x)
    My1 = np.zeros(sh)
    np.fill_diagonal(My1, source.y)
    Mxy = np.array(np.bmat([Mx1, My1]))
    Mx = np.array(np.bmat([np.eye(source.N), np.zeros(sh)]))
    My = np.array(np.bmat([np.zeros(sh), np.eye(source.N)]))
    # get laplace compensated matrices
    _DL, DLG = source.Laplace_Close_Quad.Form(
                target, side, do_DLP=do_DLP, do_SLP=do_SLP, gradient=True,
                main_type='real', gradient_type='complex', forstokes=True)
    if do_DLP:
        NF = source.Stokes_Close_Quad.NF
        fsrc = source.Stokes_Close_Quad.fsrc
        # construct resampling matrix
        RS = sp.signal.resample(np.eye(source.N), NF)
        # array that takes 2*src.N real density to resampled complex density
        Mc = Mx + 1j*My
        RSMc = RS.dot(Mc)
        FL = source.Stokes_Close_Quad.fsrc.Laplace_Close_Quad.Form(
                target, side, do_DLP=True, main_type='complex', forstokes=True)
    if do_SLP:
        if do_DLP:
            DL = source.Laplace_Close_Quad.Form(
                    target, side, do_SLP=do_SLP, main_type='real', forstokes=True)
        else:
            DL = _DL
    # Step 1
    M1 = np.zeros([target.N, 2*source.N], dtype=complex)
    if do_DLP:
        IX = (fsrc.normal_x/fsrc.normal_c)[:,None]*RSMc
        IY = (fsrc.normal_y/fsrc.normal_c)[:,None]*RSMc
        M1 += FL.dot(IX).real + 1j*FL.dot(IY).real
    if do_SLP:
        M1 += DL.dot(Mc)
    # Step 2
    M2 = DLG.dot(Mxy)
    # Step 3 and 4
    M3 = target.x[:,None]*DLG.dot(Mx)
    M4 = target.y[:,None]*DLG.dot(My)
    # add these up
    MM = M1 + np.conj(M2 - M3 - M4)
    # construct matrix of the right size
    MAT = np.zeros([2*target.N, 2*source.N], dtype=float)
    MAT[:target.N, :] = MM.real
    MAT[target.N:, :] = MM.imag
    return MAT

def Compensated_Stokes_Form2(source, target, side, tau, do_DLP=False, do_SLP=False):
    # arrays that manipulate form of density
    # there's almost certainly a faster way to deal with some of these things
    sh = (source.N, source.N)
    Mx1 = np.zeros(sh)
    np.fill_diagonal(Mx1, source.x)
    My1 = np.zeros(sh)
    np.fill_diagonal(My1, source.y)
    Mxy = np.array(np.bmat([Mx1, My1]))
    Mx = np.array(np.bmat([np.eye(source.N), np.zeros(sh)]))
    My = np.array(np.bmat([np.zeros(sh), np.eye(source.N)]))
    # get laplace compensated matrices
    _DL, DLG = source.Laplace_Close_Quad.Form(
                target, side, do_DLP=do_DLP, do_SLP=do_SLP, gradient=True,
                main_type='real', gradient_type='complex', forstokes=True)
    if do_DLP:
        NF = source.Stokes_Close_Quad.NF
        fsrc = source.Stokes_Close_Quad.fsrc
        # construct resampling matrix
        RS = sp.signal.resample(np.eye(source.N), NF)
        # array that takes 2*src.N real density to resampled complex density
        Mc = Mx + 1j*My
        RSMc = RS.dot(Mc)
        FL = source.Stokes_Close_Quad.fsrc.Laplace_Close_Quad.Form(
                target, side, do_DLP=True, main_type='complex', forstokes=True)
    if do_SLP:
        if do_DLP:
            DL = source.Laplace_Close_Quad.Form(
                    target, side, do_SLP=do_SLP, main_type='real', forstokes=True)
        else:
            DL = _DL
    # Step 1
    M1 = np.zeros([target.N, 2*source.N], dtype=complex)
    if do_DLP:
        IX = (fsrc.normal_x/fsrc.normal_c)[:,None]*RSMc
        IY = (fsrc.normal_y/fsrc.normal_c)[:,None]*RSMc
        M1 += FL.dot(IX).real + 1j*FL.dot(IY).real
    if do_SLP:
        M1 += DL.dot(Mc)
    # Step 2
    M2 = DLG.dot(Mxy)
    print(M2.dot(tau))
    # Step 3 and 4
    M3 = target.x[:,None]*DLG.dot(Mx)
    M4 = target.y[:,None]*DLG.dot(My)
    # add these up
    MM = M1 + np.conj(M2 - M3 - M4)
    # construct matrix of the right size
    MAT = np.zeros([2*target.N, 2*source.N], dtype=float)
    MAT[:target.N, :] = MM.real
    MAT[target.N:, :] = MM.imag
    return MAT

def Compensated_Stokes_Apply(source, target, side, tau, do_DLP=False,
                                                do_SLP=False, backend='fly'):
    NF = source.Stokes_Close_Quad.NF
    fsrc = source.Stokes_Close_Quad.fsrc
    taux = tau[:source.N]
    tauy = tau[source.N:]
    tauc = taux + 1j*tauy
    ftauc = sp.signal.resample(tauc, NF)
    # step 1
    u1 = np.zeros(target.N, dtype=complex)
    if do_DLP:
        IX = (fsrc.normal_x/fsrc.normal_c)*ftauc
        IY = (fsrc.normal_y/fsrc.normal_c)*ftauc
        u1a = fsrc.Laplace_Close_Quad.Apply(target, side, IX, do_DLP=True, backend=backend, main_type='complex', forstokes=True)
        u1b = fsrc.Laplace_Close_Quad.Apply(target, side, IY, do_DLP=True, backend=backend, main_type='complex', forstokes=True)
        u1 += u1a.real + 1j*u1b.real
    if do_SLP:
        u1a = source.Laplace_Close_Quad.Apply(target, side, taux, do_SLP=True, backend=backend, forstokes=True)
        u1b = source.Laplace_Close_Quad.Apply(target, side, tauy, do_SLP=True, backend=backend, forstokes=True)
        u1 += u1a + 1j*u1b
    # step 2
    tauh = source.x*taux + source.y*tauy
    _, u2 = source.Laplace_Close_Quad.Apply(target, side, tauh, do_SLP=do_SLP, do_DLP=do_DLP, gradient=True, gradient_type='complex', backend=backend, forstokes=True)
    # step 3 and 4
    _, u3 = source.Laplace_Close_Quad.Apply(target, side, taux, do_SLP=do_SLP, do_DLP=do_DLP, gradient=True, gradient_type='complex', backend=backend, forstokes=True)
    _, u4 = source.Laplace_Close_Quad.Apply(target, side, tauy, do_SLP=do_SLP, do_DLP=do_DLP, gradient=True, gradient_type='complex', backend=backend, forstokes=True)
    u3 *= target.x
    u4 *= target.y
    # add everything up
    u = u1 + np.conj(u2 - u3 - u4)
    return np.concatenate([u.real, u.imag])

