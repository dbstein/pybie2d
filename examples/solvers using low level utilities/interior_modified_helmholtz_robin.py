import numpy as np
import scipy as sp
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import time
plt.ion()
import pybie2d

"""Solve an interior Modified Helmholtz problem
On a simple domain using a global quadrature
Demonstrate how to use the pybie2d package

This example demonstrates how to do this entirely using low-level routines,
To demonstrate both how to use these low level routines
And to give you an idea what is going on under the hood in the
	higher level routines
"""

NB = 1000
NG = 100
helmholtz_k = 0.1

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Find_Near_Points = pybie2d.misc.near_points.find_near_points
Pairing = pybie2d.pairing.Pairing
k0 = pybie2d.misc.numba_special_functions.numba_k0
k1 = pybie2d.misc.numba_special_functions.numba_k1
Modified_Helmholtz_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply

################################################################################
# define problem

# robin constants (αu + βu_n)
α = 2
β = 0.5

# boundary
boundary = GSB(c=squish(NB,r=2,b=0.3,rot=np.pi/4.0))
# solution
def _solution_func(x, y):
	dx = x - (-0.5)
	dy = y - 1.0
	r = np.sqrt(dx**2 + dy**2)
	return k0(helmholtz_k*r)/(2*np.pi)
def _solution_func_dn(x, y, nx, ny):
	dx = x - (-0.5)
	dy = y - 1.0
	r = np.sqrt(dx**2 + dy**2)
	dd = helmholtz_k*k1(helmholtz_k*r)/(2*np.pi*r)
	return (nx*dx+ny*dy)*dd
def boundary_condition(x, y, nx, ny):
	bcu = _solution_func(x, y)
	bcun = _solution_func_dn(boundary.x, boundary.y, boundary.normal_x, boundary.normal_y)
	return α*bcu + β*bcun
bc = boundary_condition(boundary.x, boundary.y, boundary.normal_x, boundary.normal_y)
bcmax = np.abs(bc).max()
bc /= bcmax
def solution_func(x, y):
	return _solution_func(x, y)/bcmax

def err_plot(up):
	# compute the error
	errorp = up - solution_func(gridp.x, gridp.y)
	digitsp = -np.log10(np.abs(errorp)+1e-16)
	digits = np.zeros_like(full_grid.xg)
	digits[phys] = digitsp
	mdigits = np.ma.array(digits, mask=ext)

	# plot the error as a function of space (only good in interior)
	fig, ax = plt.subplots(1,1)
	clf = ax.imshow(mdigits[:,::-1].T, extent=[-2,2,-2,2],
												cmap=mpl.cm.viridis_r)
	ax.set_aspect('equal')
	fig.colorbar(clf)

	print('Error: {:0.2e}'.format(np.abs(errorp).max()))

################################################################################
##### solve problem the hard way ###############################################
################################################################################

################################################################################
# find physical region
# (this implements a fast way to tell if points are in or out of the boundary)
# (and of course, for the squish boundary, we could easily figure out something
#      faster, but this illustrates a general purpose routine)

full_grid = Grid([-2,2], NG, [-2,2], NG)
# this is hiding a lot of stuff!
phys, ext = boundary.find_interior_points(full_grid)
phys = full_grid.reshape(phys)
ext = full_grid.reshape(ext)

################################################################################
# solve for the density

SLP = boundary.Modified_Helmholtz_SLP_Self_Form(helmholtz_k)
DLP = boundary.Modified_Helmholtz_DLP_Self_Form(helmholtz_k)
SLPp = -(DLP/boundary.weights).T*boundary.weights
A = α*SLP + β*(-0.5*np.eye(boundary.N) + SLPp)
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([-2,2], NG, [-2,2], NG, mask=phys)

# evaluate at the target points
u = np.zeros_like(gridp.xg)
up = Modified_Helmholtz_Layer_Apply(boundary, gridp, helmholtz_k, charge=tau)
u[phys] = up
err_plot(up)

################################################################################
# correct with close evaluation (oversampling)

oversample_factor = 6
fbdy = GSB(c=sp.signal.resample(boundary.c, oversample_factor*boundary.N))
ftau = sp.signal.resample(tau, oversample_factor*boundary.N)
up = Modified_Helmholtz_Layer_Apply(fbdy, gridp, helmholtz_k, charge=ftau, backend='FMM')
u[phys] = up
err_plot(up)

################################################################################
# check on near boundary grid

close_x = boundary.x - boundary.max_h*boundary.normal_x
close_y = boundary.y - boundary.max_h*boundary.normal_y
trg = PointSet(x=close_x, y=close_y)
ub = Modified_Helmholtz_Layer_Apply(fbdy, trg, helmholtz_k, charge=ftau, backend='numba')
ua = solution_func(close_x, close_y)
err = np.abs(ua-ub)

print('Error a distance h from the boundary {:0.2e}'.format(err.max()))
