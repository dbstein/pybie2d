import numpy as np
import time
import pybie2d
from pybie2d.kernels.laplace import Laplace_Kernel_Apply, Laplace_Kernel_Form

"""
Demonstrate usage of the basic Laplace Kernels
"""

def get_random(sh, dtype):
    r = np.random.rand(*sh).astype(dtype)
    if dtype is complex:
        r += 1j*np.random.rand(*sh)
    return r

dtype=float

ns = 2000
nt = 5000
source = get_random([2, ns], float)
target = get_random([2, nt], float)
dipvec = get_random([2, ns], float)
charge = get_random([ns,], dtype)

# using numba
st = time.time()
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, charge=charge,
                            dipstr=0.5*charge, dipvec=dipvec, backend='numba',
                            dtype=dtype, gradient=True)
time_first_numba = time.time() - st
st = time.time()
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, charge=charge,
                            dipstr=0.5*charge, dipvec=dipvec, backend='numba',
                            dtype=dtype, gradient=True)
time_second_numba = time.time() - st

# using FMM
st = time.time()
pot2, gradx2, grady2 = Laplace_Kernel_Apply(source, target, charge=charge,
                            dipstr=0.5*charge, dipvec=dipvec, backend='FMM',
                            dtype=dtype, gradient=True)
time_fmm = time.time() - st

# form the matrices (have to make two calls since charge, dipstr different...)
st = time.time()
MAT, MATx, MATy = Laplace_Kernel_Form(source, target, ifcharge=True,
                        ifdipole=True, dpweight=0.5, dipvec=dipvec,
                        gradient=True, dtype=dtype)
time_form = time.time() - st
st = time.time()
pot3 = MAT.dot(charge)
gradx3 = MATx.dot(charge)
grady3 = MATy.dot(charge)
time_apply = time.time() - st

# print comparison
print('')
print('Maximum difference, potential,    numba vs. FMM:  {:0.1e}'.format(np.abs(pot1-pot2).max()))
print('Maximum difference, potential,    numba vs. Form: {:0.1e}'.format(np.abs(pot1-pot3).max()))
print('Maximum difference, x-derivative, numba vs. FMM:  {:0.1e}'.format(np.abs(gradx1-gradx2).max()))
print('Maximum difference, x-derivative, numba vs. Form: {:0.1e}'.format(np.abs(gradx1-gradx3).max()))
print('Maximum difference, y-derivative, numba vs. FMM:  {:0.1e}'.format(np.abs(grady1-grady2).max()))
print('Maximum difference, y-derivative, numba vs. Form: {:0.1e}'.format(np.abs(grady1-grady3).max()))
print('')
print('Time for first  numba apply (ms): {:0.1f}'.format(time_first_numba*1000))
print('Time for second numba apply (ms): {:0.1f}'.format(time_second_numba*1000))
print('Time for FMM based    apply (ms): {:0.1f}'.format(time_fmm*1000))
print('Time for matrix formation   (ms): {:0.1f}'.format(time_form*1000))
print('Time for preformed apply    (ms): {:0.1f}'.format(time_apply*1000))

