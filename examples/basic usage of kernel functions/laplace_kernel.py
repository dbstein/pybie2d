import numpy as np
import time
import pybie2d
from pybie2d.kernels.high_level.laplace import Laplace_Kernel_Apply, Laplace_Kernel_Form

"""
Demonstrate usage of the basic Laplace Kernels
Also timing/consistency checks
"""

def get_random(sh, dtype):
    r = np.random.rand(*sh).astype(dtype)
    if dtype is complex:
        r += 1j*np.random.rand(*sh)
    return r

dtype=float

ns = 5000
nt = 5000
source = get_random([2, ns], float)
target = get_random([2, nt], float)
dipvec = get_random([2, ns], float)
charge = get_random([ns,], dtype)
dipstr = get_random([ns,], dtype)

print('\n-- Laplace 2D Kernel Tests, Charge Only, No Derivatives --\n')

# using numba
print('Testing Numba (Apply)')
pot_numba = Laplace_Kernel_Apply(source, target, charge=charge, backend='numba')
time_numba =  %timeit -o Laplace_Kernel_Apply(source, target, charge=charge, backend='numba')

# using FMM
print('Testing FMM (Apply)')
pot_fmm = Laplace_Kernel_Apply(source, target, charge=charge, backend='FMM')
time_fmm =  %timeit -o Laplace_Kernel_Apply(source, target, charge=charge, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
MAT = Laplace_Kernel_Form(source, target, ifcharge=True)
time_numexpr_form = time.time() - st
pot_numexpr = MAT.dot(charge)
time_apply = %timeit -o MAT.dot(charge)

# print comparison
print('')
print('Maximum difference, potential,    numba vs. FMM:  {:0.1e}'.format(np.abs(pot_numba-pot_fmm).max()))
print('Maximum difference, potential,    numba vs. Form: {:0.1e}'.format(np.abs(pot_numba-pot_numexpr).max()))
print('Maximum difference, potential,    FMM   vs. Form: {:0.1e}'.format(np.abs(pot_fmm-pot_numexpr).max()))
print('')
print('Time for numba apply     (ms): {:0.2f}'.format(time_numba.average*1000))
print('Time for FMM apply       (ms): {:0.2f}'.format(time_fmm.average*1000))
print('Time for numexpr form    (ms): {:0.2f}'.format(time_numexpr_form*1000))
print('Time for preformed apply (ms): {:0.2f}'.format(time_apply.average*1000))

print('\n-- Laplace 2D Kernel Tests, Dipole Only, No Derivatives --\n')

# using numba
print('Testing Numba (Apply)')
pot_numba = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba')
time_numba =  %timeit -o Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba')

# using FMM
print('Testing FMM (Apply)')
pot_fmm = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM')
time_fmm =  %timeit -o Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
MAT = Laplace_Kernel_Form(source, target, ifdipole=True, dipvec=dipvec)
time_numexpr_form = time.time() - st
pot_numexpr = MAT.dot(dipstr)
time_apply = %timeit -o MAT.dot(dipstr)

# print comparison
print('')
print('Maximum difference, potential,    numba vs. FMM:  {:0.1e}'.format(np.abs(pot_numba-pot_fmm).max()))
print('Maximum difference, potential,    numba vs. Form: {:0.1e}'.format(np.abs(pot_numba-pot_numexpr).max()))
print('Maximum difference, potential,    FMM   vs. Form: {:0.1e}'.format(np.abs(pot_fmm-pot_numexpr).max()))
print('')
print('Time for numba apply     (ms): {:0.2f}'.format(time_numba.average*1000))
print('Time for FMM apply       (ms): {:0.2f}'.format(time_fmm.average*1000))
print('Time for numexpr form    (ms): {:0.2f}'.format(time_numexpr_form*1000))
print('Time for preformed apply (ms): {:0.2f}'.format(time_apply.average*1000))

print('\n-- Laplace 2D Kernel Tests, Charge and Dipole, No Derivatives --\n')

# using numba
print('Testing Numba (Apply)')
pot_numba = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='numba')
time_numba =  %timeit -o Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='numba')

# using FMM
print('Testing FMM (Apply)')
pot_fmm = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='FMM')
time_fmm =  %timeit -o Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
MATc = Laplace_Kernel_Form(source, target, ifcharge=True)
MATd = Laplace_Kernel_Form(source, target, ifdipole=True, dipvec=dipvec)
time_numexpr_form = time.time() - st
pot_numexpr = MATc.dot(charge) + MATd.dot(dipstr)
time_apply = %timeit -o MATc.dot(charge) + MATd.dot(dipstr)

# print comparison
print('')
print('Maximum difference, potential,    numba vs. FMM:  {:0.1e}'.format(np.abs(pot_numba-pot_fmm).max()))
print('Maximum difference, potential,    numba vs. Form: {:0.1e}'.format(np.abs(pot_numba-pot_numexpr).max()))
print('Maximum difference, potential,    FMM   vs. Form: {:0.1e}'.format(np.abs(pot_fmm-pot_numexpr).max()))
print('')
print('Time for numba apply     (ms): {:0.2f}'.format(time_numba.average*1000))
print('Time for FMM apply       (ms): {:0.2f}'.format(time_fmm.average*1000))
print('Time for numexpr form    (ms): {:0.2f}'.format(time_numexpr_form*1000))
print('Time for preformed apply (ms): {:0.2f}'.format(time_apply.average*1000))








print('\n-- Laplace 2D Kernel Tests, Charge Only, With Derivatives --\n')

# using numba
print('Testing Numba (Apply)')
pot_numba, gx_numba, gy_numba = Laplace_Kernel_Apply(source, target, charge=charge, backend='numba', gradient=True)
time_numba =  %timeit -o Laplace_Kernel_Apply(source, target, charge=charge, backend='numba', gradient=True)

# using FMM
print('Testing FMM (Apply)')
pot_fmm, gx_fmm, gy_fmm = Laplace_Kernel_Apply(source, target, charge=charge, backend='FMM', gradient=True)
time_fmm =  %timeit -o Laplace_Kernel_Apply(source, target, charge=charge, backend='FMM', gradient=True)

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
MAT, MATx, MATy = Laplace_Kernel_Form(source, target, ifcharge=True, gradient=True)
time_numexpr_form = time.time() - st
pot_numexpr = MAT.dot(charge)
gx_numexpr = MATx.dot(charge)
gy_numexpr = MATy.dot(charge)
time_apply = %timeit -o MAT.dot(charge); MATx.dot(charge); MATy.dot(charge)

# print comparison
print('')
print('Maximum difference, potential,    numba vs. FMM:  {:0.1e}'.format(np.abs(pot_numba-pot_fmm).max()))
print('Maximum difference, potential,    numba vs. Form: {:0.1e}'.format(np.abs(pot_numba-pot_numexpr).max()))
print('Maximum difference, potential,    FMM   vs. Form: {:0.1e}'.format(np.abs(pot_fmm-pot_numexpr).max()))
print('Maximum difference, gradient_x,   numba vs. FMM:  {:0.1e}'.format(np.abs(gx_numba-gx_fmm).max()))
print('Maximum difference, gradient_x,   numba vs. Form: {:0.1e}'.format(np.abs(gx_numba-gx_numexpr).max()))
print('Maximum difference, gradient_x,   FMM   vs. Form: {:0.1e}'.format(np.abs(gx_fmm-gx_numexpr).max()))
print('Maximum difference, gradient_y,   numba vs. FMM:  {:0.1e}'.format(np.abs(gy_numba-gy_fmm).max()))
print('Maximum difference, gradient_y,   numba vs. Form: {:0.1e}'.format(np.abs(gy_numba-gy_numexpr).max()))
print('Maximum difference, gradient_y,   FMM   vs. Form: {:0.1e}'.format(np.abs(gy_fmm-gy_numexpr).max()))
print('')
print('Time for numba apply     (ms): {:0.2f}'.format(time_numba.average*1000))
print('Time for FMM apply       (ms): {:0.2f}'.format(time_fmm.average*1000))
print('Time for numexpr form    (ms): {:0.2f}'.format(time_numexpr_form*1000))
print('Time for preformed apply (ms): {:0.2f}'.format(time_apply.average*1000))

print('\n-- Laplace 2D Kernel Tests, Dipole Only, With Derivatives --\n')

# using numba
print('Testing Numba (Apply)')
pot_numba, gx_numba, gy_numba = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba', gradient=True)
time_numba =  %timeit -o Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba', gradient=True)

# using FMM
print('Testing FMM (Apply)')
pot_fmm, gx_fmm, gy_fmm = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM', gradient=True)
time_fmm =  %timeit -o Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM', gradient=True)

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
MAT, MATx, MATy = Laplace_Kernel_Form(source, target, ifdipole=True, dipvec=dipvec, gradient=True)
time_numexpr_form = time.time() - st
pot_numexpr = MAT.dot(dipstr)
gx_numexpr = MATx.dot(dipstr)
gy_numexpr = MATy.dot(dipstr)
time_apply = %timeit -o MAT.dot(dipstr); MATx.dot(dipstr); MATy.dot(dipstr)

# print comparison
print('')
print('Maximum difference, potential,    numba vs. FMM:  {:0.1e}'.format(np.abs(pot_numba-pot_fmm).max()))
print('Maximum difference, potential,    numba vs. Form: {:0.1e}'.format(np.abs(pot_numba-pot_numexpr).max()))
print('Maximum difference, potential,    FMM   vs. Form: {:0.1e}'.format(np.abs(pot_fmm-pot_numexpr).max()))
print('Maximum difference, gradient_x,   numba vs. FMM:  {:0.1e}'.format(np.abs(gx_numba-gx_fmm).max()))
print('Maximum difference, gradient_x,   numba vs. Form: {:0.1e}'.format(np.abs(gx_numba-gx_numexpr).max()))
print('Maximum difference, gradient_x,   FMM   vs. Form: {:0.1e}'.format(np.abs(gx_fmm-gx_numexpr).max()))
print('Maximum difference, gradient_y,   numba vs. FMM:  {:0.1e}'.format(np.abs(gy_numba-gy_fmm).max()))
print('Maximum difference, gradient_y,   numba vs. Form: {:0.1e}'.format(np.abs(gy_numba-gy_numexpr).max()))
print('Maximum difference, gradient_y,   FMM   vs. Form: {:0.1e}'.format(np.abs(gy_fmm-gy_numexpr).max()))
print('')
print('Time for numba apply     (ms): {:0.2f}'.format(time_numba.average*1000))
print('Time for FMM apply       (ms): {:0.2f}'.format(time_fmm.average*1000))
print('Time for numexpr form    (ms): {:0.2f}'.format(time_numexpr_form*1000))
print('Time for preformed apply (ms): {:0.2f}'.format(time_apply.average*1000))

print('\n-- Laplace 2D Kernel Tests, Charge and Dipole, No Derivatives --\n')

