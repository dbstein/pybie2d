import numpy as np
from scipy.special import i0, k0, i1, k1
import time
import pybie2d
from pybie2d.kernels.high_level.modified_helmholtz import Modified_Helmholtz_Kernel_Apply, Modified_Helmholtz_Kernel_Form
from pybie2d.misc.numba_special_functions import numba_k0, numba_k1
from pybie2d.misc.function_generator_functions import fg_k0, fg_k1

print('\n-- Testing numba special function implementation --\n')

# test the underlying numba implementations of i0, k0
# function generator versions give good relative error up to x=200, then good absolute error
x = np.linspace(0,100,1000000)

y1 = k0(x)
y2 = numba_k0(x)
y3 = fg_k0(x)
print('Timing scipy k0')
st=time.time(); k0(x); print('   (ms): {:0.2f}'.format((time.time()-st)*1000))
print('Timing numba k0')
st=time.time(); numba_k0(x); print('   (ms): {:0.2f}'.format((time.time()-st)*1000))
print('Timing function generator k0')
st=time.time(); fg_k0(x); print('   (ms): {:0.2f}'.format((time.time()-st)*1000))
print('Max relative difference in k0 (numba--scipy): {:0.2e}'.format(np.abs((y1[1:]-y2[1:])/np.abs(y1[1:])).max()))
print('Max relative difference in k1 (funcg--scipy): {:0.2e}'.format(np.abs((y1[1:]-y3[1:])/np.abs(y1[1:])).max()))

y1 = k1(x)
y2 = numba_k1(x)
y3 = fg_k1(x)
print('Timing scipy k1')
st=time.time(); k1(x); print('   (ms): {:0.2f}'.format((time.time()-st)*1000))
print('Timing numba k1')
st=time.time(); numba_k1(x); print('   (ms): {:0.2f}'.format((time.time()-st)*1000))
print('Timing function generator k1')
st=time.time(); fg_k1(x); print('   (ms): {:0.2f}'.format((time.time()-st)*1000))
print('Max relative difference in k0 (numba--scipy): {:0.2e}'.format(np.abs((y1[1:]-y2[1:])/np.abs(y1[1:])).max()))
print('Max adj rel  difference in k1 (funcg--scipy): {:0.2e}'.format(np.abs((y1[1:]-y3[1:])/np.abs(y1[1:]+1)).max()))

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

ns = 2000
nt = 2000
test_self = False
helmk = 10.0

source = get_random([2, ns], float)
target = source if test_self else get_random([2, nt], float)
dipvec = get_random([2, ns], float)
charge = get_random([ns,], dtype)
dipstr = get_random([ns,], dtype)

print('\n-- Modified Helmholtz 2D Kernel Tests, Charge Only, No Derivatives --\n')

# using numba
print('Testing Numba (Apply)')
pot_numba = Modified_Helmholtz_Kernel_Apply(source, target, helmk, charge=charge, backend='numba')
time_numba =  %timeit -o Modified_Helmholtz_Kernel_Apply(source, target, helmk, charge=charge, backend='numba')

# using FMM
print('Testing FMM (Apply)')
pot_fmm = Modified_Helmholtz_Kernel_Apply(source, target, helmk, charge=charge, backend='FMM')
time_fmm =  %timeit -o Modified_Helmholtz_Kernel_Apply(source, target, helmk, charge=charge, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
MAT = Modified_Helmholtz_Kernel_Form(source, target, helmk, ifcharge=True)
st = time.time()
MAT = Modified_Helmholtz_Kernel_Form(source, target, helmk, ifcharge=True)
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

print('\n-- Modified Helmholtz 2D Kernel Tests, Dipole Only, No Derivatives --\n')

# using numba
print('Testing Numba (Apply)')
pot_numba = Modified_Helmholtz_Kernel_Apply(source, target, helmk, dipstr=dipstr, dipvec=dipvec, backend='numba')
time_numba =  %timeit -o Modified_Helmholtz_Kernel_Apply(source, target, helmk, dipstr=dipstr, dipvec=dipvec, backend='numba')

# using FMM
print('Testing FMM (Apply)')
pot_fmm = Modified_Helmholtz_Kernel_Apply(source, target, helmk, dipstr=dipstr, dipvec=dipvec, backend='FMM')
time_fmm =  %timeit -o Modified_Helmholtz_Kernel_Apply(source, target, helmk, dipstr=dipstr, dipvec=dipvec, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
MAT = Modified_Helmholtz_Kernel_Form(source, target, helmk, ifdipole=True, dipvec=dipvec)
st = time.time()
MAT = Modified_Helmholtz_Kernel_Form(source, target, helmk, ifdipole=True, dipvec=dipvec)
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


print('\n-- Modified Helmholtz 2D Kernel Tests, Charge and Dipole, No Derivatives --\n')

# using numba
print('Testing Numba (Apply)')
pot_numba = Modified_Helmholtz_Kernel_Apply(source, target, helmk, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='numba')
time_numba =  %timeit -o Modified_Helmholtz_Kernel_Apply(source, target, helmk, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='numba')

# using FMM
print('Testing FMM (Apply)')
pot_fmm = Modified_Helmholtz_Kernel_Apply(source, target, helmk, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='FMM')
time_fmm =  %timeit -o Modified_Helmholtz_Kernel_Apply(source, target, helmk, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='FMM')

# using numexpr
print('Testing Numexpr (Form)')
st = time.time()
MATc = Modified_Helmholtz_Kernel_Form(source, target, helmk, ifcharge=True)
MATd = Modified_Helmholtz_Kernel_Form(source, target, helmk, ifdipole=True, dipvec=dipvec)
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
