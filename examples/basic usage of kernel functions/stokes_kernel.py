import numpy as np
import time
import pybie2d
from pybie2d.kernels.high_level.stokes import Stokes_Kernel_Apply, Stokes_Kernel_Form

"""
Demonstrate usage of the basic Stokes Kernels
Also timing/consistency checks
"""

def get_random(sh, dtype):
    r = np.random.rand(*sh).astype(dtype)
    if dtype is complex:
        r += 1j*np.random.rand(*sh)
    return r

ns = 10000
nt = 10000
test_self = True

source = get_random([2, ns], float)
target = source if test_self else get_random([2, nt], float)
dipvec = get_random([2, ns], float)
forces = get_random([2, ns], float)
dipstr = get_random([2, ns], float)

print('\n-- Stokes 2D Kernel Tests, Forces Only --\n')

# using numba
print('Testing Numba (Apply)')
vel_numba = Stokes_Kernel_Apply(source, target, forces=forces, backend='numba')
time_numba =  %timeit -o Stokes_Kernel_Apply(source, target, forces=forces, backend='numba')

# using FMM
print('Testing FMM (Apply)')
vel_fmm = Stokes_Kernel_Apply(source, target, forces=forces, backend='FMM')
time_fmm =  %timeit -o Stokes_Kernel_Apply(source, target, forces=forces, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
MAT = Stokes_Kernel_Form(source, target, ifforce=True)
time_numexpr_form = time.time() - st
vel_numexpr = MAT.dot(forces.ravel()).reshape(target.shape)
time_apply = %timeit -o MAT.dot(forces.ravel()).reshape(target.shape)

# print comparison
print('')
print('Maximum difference, velocity,    numba vs. FMM:  {:0.1e}'.format(np.abs(vel_numba-vel_fmm).max()))
print('Maximum difference, velocity,    numba vs. Form: {:0.1e}'.format(np.abs(vel_numba-vel_numexpr).max()))
print('Maximum difference, velocity,    FMM   vs. Form: {:0.1e}'.format(np.abs(vel_fmm-vel_numexpr).max()))
print('')
print('Time for numba apply     (ms): {:0.2f}'.format(time_numba.average*1000))
print('Time for FMM apply       (ms): {:0.2f}'.format(time_fmm.average*1000))
print('Time for numexpr form    (ms): {:0.2f}'.format(time_numexpr_form*1000))
print('Time for preformed apply (ms): {:0.2f}'.format(time_apply.average*1000))

print('\n-- Stokes 2D Kernel Tests, Dipoles Only --\n')

# using numba
print('Testing Numba (Apply)')
vel_numba = Stokes_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba')
time_numba =  %timeit -o Stokes_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba')

# using FMM
print('Testing FMM (Apply)')
vel_fmm = Stokes_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM')
time_fmm =  %timeit -o Stokes_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
MAT = Stokes_Kernel_Form(source, target, dipvec=dipvec, ifdipole=True)
time_numexpr_form = time.time() - st
vel_numexpr = MAT.dot(dipstr.ravel()).reshape(target.shape)
time_apply = %timeit -o MAT.dot(dipstr.ravel()).reshape(target.shape)

# print comparison
print('')
print('Maximum difference, velocity,    numba vs. FMM:  {:0.1e}'.format(np.abs(vel_numba-vel_fmm).max()))
print('Maximum difference, velocity,    numba vs. Form: {:0.1e}'.format(np.abs(vel_numba-vel_numexpr).max()))
print('Maximum difference, velocity,    FMM   vs. Form: {:0.1e}'.format(np.abs(vel_fmm-vel_numexpr).max()))
print('')
print('Time for numba apply     (ms): {:0.2f}'.format(time_numba.average*1000))
print('Time for FMM apply       (ms): {:0.2f}'.format(time_fmm.average*1000))
print('Time for numexpr form    (ms): {:0.2f}'.format(time_numexpr_form*1000))
print('Time for preformed apply (ms): {:0.2f}'.format(time_apply.average*1000))

print('\n-- Stokes 2D Kernel Tests, Forces and Dipoles --\n')

# using numba
print('Testing Numba (Apply)')
vel_numba = Stokes_Kernel_Apply(source, target, forces=forces, dipstr=dipstr, dipvec=dipvec, backend='numba')
time_numba =  %timeit -o Stokes_Kernel_Apply(source, target, forces=forces, dipstr=dipstr, dipvec=dipvec, backend='numba')

# using FMM
print('Testing FMM (Apply)')
vel_fmm = Stokes_Kernel_Apply(source, target, forces=forces, dipstr=dipstr, dipvec=dipvec, backend='FMM')
time_fmm =  %timeit -o Stokes_Kernel_Apply(source, target, forces=forces, dipstr=dipstr, dipvec=dipvec, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
fMAT = Stokes_Kernel_Form(source, target, ifforce=True)
dMAT = Stokes_Kernel_Form(source, target, dipvec=dipvec, ifdipole=True)
time_numexpr_form = time.time() - st
vel_numexpr = (fMAT.dot(forces.ravel()) + dMAT.dot(dipstr.ravel())).reshape(target.shape)
time_apply = %timeit -o (fMAT.dot(forces.ravel()) + dMAT.dot(dipstr.ravel())).reshape(target.shape)

# print comparison
print('')
print('Maximum difference, velocity,    numba vs. FMM:  {:0.1e}'.format(np.abs(vel_numba-vel_fmm).max()))
print('Maximum difference, velocity,    numba vs. Form: {:0.1e}'.format(np.abs(vel_numba-vel_numexpr).max()))
print('Maximum difference, velocity,    FMM   vs. Form: {:0.1e}'.format(np.abs(vel_fmm-vel_numexpr).max()))
print('')
print('Time for numba apply     (ms): {:0.2f}'.format(time_numba.average*1000))
print('Time for FMM apply       (ms): {:0.2f}'.format(time_fmm.average*1000))
print('Time for numexpr form    (ms): {:0.2f}'.format(time_numexpr_form*1000))
print('Time for preformed apply (ms): {:0.2f}'.format(time_apply.average*1000))


