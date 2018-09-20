import numpy as np
import time
import pybie2d
from pybie2d.kernels.high_level.cauchy import Cauchy_Kernel_Apply, Cauchy_Kernel_Form

"""
Demonstrate usage of the basic Cauchy Kernels
Also timing/consistency checks
"""

def get_random(sh, dtype):
    r = np.random.rand(*sh).astype(dtype)
    if dtype is complex:
        r += 1j*np.random.rand(*sh)
    return r

dtype=float

ns = 2000
nt = 2000
test_self = True

source = get_random([ns,], complex)
target = source if test_self else get_random([nt,], complex)
dipstr = get_random([ns,], complex)

print('\n-- Cauchy 2D Kernel Tests --\n')

# using numba
print('Testing Numba (Apply)')
pot_numba = Cauchy_Kernel_Apply(source, target, dipstr, backend='numba')
time_numba =  %timeit -o Cauchy_Kernel_Apply(source, target, dipstr, backend='numba')

# using FMM
print('Testing FMM (Apply)')
pot_fmm = Cauchy_Kernel_Apply(source, target, dipstr, backend='FMM')
time_fmm =  %timeit -o Cauchy_Kernel_Apply(source, target, dipstr, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
MAT = Cauchy_Kernel_Form(source, target)
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
