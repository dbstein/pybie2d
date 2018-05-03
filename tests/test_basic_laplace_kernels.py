import numpy as np
import pybie2d
from pybie2d.kernels.laplace import Laplace_Kernel_Apply, Laplace_Kernel_Form

"""
Test all of the Laplace Kernel Functions against each other
"""

def get_random(sh, dtype):
    r = np.random.rand(*sh).astype(dtype)
    if dtype is complex:
        r += 1j*np.random.rand(*sh)
    return r

ns = 100
nt = 200

################################################################################
# Float ########################################################################
################################################################################

dtype = float

source = get_random([2, ns], float)
target = get_random([2, nt], float)
dipvec = get_random([2, ns], float)
charge = get_random([ns,], dtype)
dipstr = get_random([ns,], dtype)

############################################################################
# Without gradients

# Laplace Kernel, charge only
# force usage of numba
pot1 = Laplace_Kernel_Apply(source, target, charge=charge, backend='numba', dtype=dtype)
# force usage of FMM
pot2 = Laplace_Kernel_Apply(source, target, charge=charge, backend='FMM', dtype=dtype)
# form the matrix
MAT = Laplace_Kernel_Form(source, target, ifcharge=True)
pot3 = MAT.dot(charge)

def test1():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)

# Laplace Kernel, dipole only
# force usage of numba
pot1 = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba', dtype=dtype)
# force usage of FMM
pot2 = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM', dtype=dtype)
# form the matrix
MAT = Laplace_Kernel_Form(source, target, ifdipole=True, dipvec=dipvec)
pot3 = MAT.dot(dipstr)

def test2():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)

# Laplace Kernel, charge + dipstr
# force usage of numba
pot1 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='numba', dtype=dtype)
# force usage of FMM
pot2 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='FMM', dtype=dtype)

def test3():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)

# Laplace Kernel, charge + dipstr, but one a scalar multiple of the other
# force usage of numba
pot1 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=0.5*charge, dipvec=dipvec, backend='numba', dtype=dtype)
# force usage of FMM
pot2 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=0.5*charge, dipvec=dipvec, backend='FMM', dtype=dtype)
# form the matrix
MAT = Laplace_Kernel_Form(source, target, ifcharge=True, chweight=1.0, ifdipole=True, dpweight=0.5, dipvec=dipvec)
pot3 = MAT.dot(charge)

def test4():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)

############################################################################
# With gradients now

# Laplace Kernel, charge only
# force usage of numba
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, charge=charge, backend='numba', gradient=True, dtype=dtype)
# force usage of FMM
pot2, gradx2, grady2 = Laplace_Kernel_Apply(source, target, charge=charge, backend='FMM', gradient=True, dtype=dtype)
# form the matrix
MAT, MATX, MATY = Laplace_Kernel_Form(source, target, ifcharge=True, gradient=True)
pot3 = MAT.dot(charge)
gradx3 = MATX.dot(charge)
grady3 = MATY.dot(charge)

def test5():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx2, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx3, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady2, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady3, atol=0.0, rtol=1e-8)

# Laplace Kernel, dipole only
# force usage of numba
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba', gradient=True, dtype=dtype)
# force usage of FMM
pot2, gradx2, grady2 = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM', gradient=True, dtype=dtype)
# form the matrix
MAT, MATX, MATY = Laplace_Kernel_Form(source, target, ifdipole=True, dipvec=dipvec, gradient=True)
pot3 = MAT.dot(dipstr)
gradx3 = MATX.dot(dipstr)
grady3 = MATY.dot(dipstr)

def test6():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx2, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx3, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady2, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady3, atol=0.0, rtol=1e-8)

# Laplace Kernel, charge + dipstr
# force usage of numba
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='numba', gradient=True, dtype=dtype)
# force usage of FMM
pot2, gradx2, grady2 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='FMM', gradient=True, dtype=dtype)

def test7():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx2, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady2, atol=0.0, rtol=1e-8)

# Laplace Kernel, charge + dipstr, but one a scalar multiple of the other
# force usage of numba
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=0.5*charge, dipvec=dipvec, backend='numba', gradient=True, dtype=dtype)
# force usage of FMM
pot2, gradx2, grady2 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=0.5*charge, dipvec=dipvec, backend='FMM', gradient=True, dtype=dtype)
# form the matrix
MAT, MATX, MATY = Laplace_Kernel_Form(source, target, ifcharge=True, chweight=1.0, ifdipole=True, dpweight=0.5, dipvec=dipvec, gradient=True)
pot3 = MAT.dot(charge)
gradx3 = MATX.dot(charge)
grady3 = MATY.dot(charge)

def test8():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx2, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx3, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady2, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady3, atol=0.0, rtol=1e-8)

################################################################################
# Complex ######################################################################
################################################################################

dtype = complex

source = get_random([2, ns], float)
target = get_random([2, nt], float)
dipvec = get_random([2, ns], float)
charge = get_random([ns,], dtype)
dipstr = get_random([ns,], dtype)

############################################################################
# Without gradients

# Laplace Kernel, charge only
# force usage of numba
pot1 = Laplace_Kernel_Apply(source, target, charge=charge, backend='numba', dtype=dtype)
# force usage of FMM
pot2 = Laplace_Kernel_Apply(source, target, charge=charge, backend='FMM', dtype=dtype)
# form the matrix
MAT = Laplace_Kernel_Form(source, target, ifcharge=True)
pot3 = MAT.dot(charge)

def test9():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)

# Laplace Kernel, dipole only
# force usage of numba
pot1 = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba', dtype=dtype)
# force usage of FMM
pot2 = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM', dtype=dtype)
# form the matrix
MAT = Laplace_Kernel_Form(source, target, ifdipole=True, dipvec=dipvec)
pot3 = MAT.dot(dipstr)

def test10():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)

# Laplace Kernel, charge + dipstr
# force usage of numba
pot1 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='numba', dtype=dtype)
# force usage of FMM
pot2 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='FMM', dtype=dtype)

def test11():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)

# Laplace Kernel, charge + dipstr, but one a scalar multiple of the other
# force usage of numba
pot1 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=0.5*charge, dipvec=dipvec, backend='numba', dtype=dtype)
# force usage of FMM
pot2 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=0.5*charge, dipvec=dipvec, backend='FMM', dtype=dtype)
# form the matrix
MAT = Laplace_Kernel_Form(source, target, ifcharge=True, chweight=1.0, ifdipole=True, dpweight=0.5, dipvec=dipvec)
pot3 = MAT.dot(charge)

def test12():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)

############################################################################
# With gradients now

# Laplace Kernel, charge only
# force usage of numba
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, charge=charge, backend='numba', gradient=True, dtype=dtype)
# force usage of FMM
pot2, gradx2, grady2 = Laplace_Kernel_Apply(source, target, charge=charge, backend='FMM', gradient=True, dtype=dtype)
# form the matrix
MAT, MATX, MATY = Laplace_Kernel_Form(source, target, ifcharge=True, gradient=True)
pot3 = MAT.dot(charge)
gradx3 = MATX.dot(charge)
grady3 = MATY.dot(charge)

def test13():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx2, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx3, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady2, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady3, atol=0.0, rtol=1e-8)

# Laplace Kernel, dipole only
# force usage of numba
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='numba', gradient=True, dtype=dtype)
# force usage of FMM
pot2, gradx2, grady2 = Laplace_Kernel_Apply(source, target, dipstr=dipstr, dipvec=dipvec, backend='FMM', gradient=True, dtype=dtype)
# form the matrix
MAT, MATX, MATY = Laplace_Kernel_Form(source, target, ifdipole=True, dipvec=dipvec, gradient=True)
pot3 = MAT.dot(dipstr)
gradx3 = MATX.dot(dipstr)
grady3 = MATY.dot(dipstr)

def test14():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx2, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx3, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady2, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady3, atol=0.0, rtol=1e-8)

# Laplace Kernel, charge + dipstr
# force usage of numba
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='numba', gradient=True, dtype=dtype)
# force usage of FMM
pot2, gradx2, grady2 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=dipstr, dipvec=dipvec, backend='FMM', gradient=True, dtype=dtype)

def test15():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx2, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady2, atol=0.0, rtol=1e-8)

# Laplace Kernel, charge + dipstr, but one a scalar multiple of the other
# force usage of numba
pot1, gradx1, grady1 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=0.5*charge, dipvec=dipvec, backend='numba', gradient=True, dtype=dtype)
# force usage of FMM
pot2, gradx2, grady2 = Laplace_Kernel_Apply(source, target, charge=charge, dipstr=0.5*charge, dipvec=dipvec, backend='FMM', gradient=True, dtype=dtype)
# form the matrix
MAT, MATX, MATY = Laplace_Kernel_Form(source, target, ifcharge=True, chweight=1.0, ifdipole=True, dpweight=0.5, dipvec=dipvec, gradient=True)
pot3 = MAT.dot(charge)
gradx3 = MATX.dot(charge)
grady3 = MATY.dot(charge)

def test16():
    assert np.allclose(pot1, pot2, atol=0.0, rtol=1e-8)
    assert np.allclose(pot1, pot3, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx2, atol=0.0, rtol=1e-8)
    assert np.allclose(gradx1, gradx3, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady2, atol=0.0, rtol=1e-8)
    assert np.allclose(grady1, grady3, atol=0.0, rtol=1e-8)

