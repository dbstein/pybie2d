import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import time
plt.ion()
import pybie2d

"""
Demonstrate how to use the pybie2d package to solve an interior Stokes problem
On a complicated domain using a global quadrature

This example demonstrates how to do this entirely using low-level routines,
To demonstrate both how to use these low level routines
And to give you an idea what is going on under the hood in the
	higher level routines
"""

NG = 200
NB = 200
h_max = 1.0/NG
helmholtz_k = 5.0

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
PPB = pybie2d.boundaries.panel_polygon_boundary.panel_polygon_boundary.Panel_Polygon_Boundary
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Modified_Helmholtz_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
Modified_Helmholtz_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply
k0 = pybie2d.misc.numba_special_functions.numba_k0
k1 = pybie2d.misc.numba_special_functions.numba_k1
Boundary_Collection = pybie2d.boundaries.collection.BoundaryCollection

################################################################################
# define problem

# boundary
boundary1 = PPB([0,1,1,0], [0,0,1,1], [h_max]*4, [True]*4, order=16, dyadic_levels=20, dyadic_base=3)
boundary2 = GSB(c=squish(2*int(1.5*NG/2),x=0.5,y=0.5,r=0.3,b=0.5,rot=0.0))
boundary = Boundary_Collection()
boundary.add([boundary1, boundary2], ['i', 'e'])
boundary.amass_information()
# get oversampled boundary for close eval and interpolation routines
upsample_factor_1 = 6
upsample_factor_2 = boundary2.max_h
upsample = lambda f: sp.signal.resample(f, 6*f.shape[0])
h = h_max
fbdy1, IMAT = boundary1.prepare_oversampling(h/6.0)
IMAT = sp.sparse.csr_matrix(IMAT)
fbdy2 = GSB(c=upsample(boundary2.c))
fbdy = Boundary_Collection()
fbdy.add([fbdy1, fbdy2], ['i', 'e'])
try:
	fbdy.amass_information()
except:
	pass
def upsample_scalar(f):
	f1 = f[:boundary1.N]
	f2 = f[boundary1.N:]
	return np.concatenate([ IMAT.dot(f1), upsample(f2) ])

# solution
def _solution_func_base(x, y, x0, y0):
	dx = x - x0
	dy = y - y0
	r = np.sqrt(dx**2 + dy**2)
	return k0(helmholtz_k*r)/(2*np.pi)
def _solution_func_dn_base(x, y, nx, ny, x0, y0):
	dx = x - x0
	dy = y - y0
	r = np.sqrt(dx**2 + dy**2)
	dd = helmholtz_k*k1(helmholtz_k*r)/(2*np.pi*r)
	return (nx*dx+ny*dy)*dd
_solution_func_1 = lambda x, y: _solution_func_base(x, y, 0.5, 0.5)
_solution_func_dn_1 = lambda x, y, nx, ny: _solution_func_dn_base(x, y, nx, ny, 0.5, 0.5)
_solution_func_2 = lambda x, y: _solution_func_base(x, y, -0.1, 0.5)
_solution_func_dn_2 = lambda x, y, nx, ny: _solution_func_dn_base(x, y, nx, ny, -0.1, 0.5)
_solution_func = lambda x, y: _solution_func_1(x, y) + _solution_func_2(x, y)
_solution_func_dn = lambda x, y, nx, ny: _solution_func_dn_1(x, y, nx, ny) + _solution_func_dn_2(x, y, nx, ny)
bc = _solution_func_dn(boundary.x, boundary.y, boundary.normal_x, boundary.normal_y)
bcmax = np.abs(bc).max()
bc /= bcmax
solution_func = lambda x, y: _solution_func(x, y)/bcmax

def err_plot(up, func):
	# compute the error
	errorp = up - func(full_grid.xg[phys], full_grid.yg[phys])
	digitsp = -np.log10(np.abs(errorp)+1e-16)
	digits = np.zeros_like(full_grid.xg)
	digits[phys] = digitsp
	mdigits = np.ma.array(digits, mask=ext)

	# plot the error as a function of space (only good in interior)
	fig, ax = plt.subplots(1,1)
	clf = ax.imshow(mdigits[:,::-1].T, extent=[0,1,0,1],
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

full_grid = Grid([0,1], NG, [0,1], NG, x_endpoints=[False,False], y_endpoints=[False,False])
x = full_grid.xg
y = full_grid.yg
phys, ext = boundary.compute_physical_region(target=full_grid)

################################################################################
# solve for the density

DLP11 = Modified_Helmholtz_Layer_Form(boundary1, k=helmholtz_k, ifdipole=True)
DLP21 = Modified_Helmholtz_Layer_Form(boundary1, boundary2, k=helmholtz_k, ifdipole=True)
DLP12 = Modified_Helmholtz_Layer_Form(boundary2, boundary1, k=helmholtz_k, ifdipole=True)
DLP22 = boundary2.Modified_Helmholtz_DLP_Self_Form(helmholtz_k)
DLP = np.array(np.bmat([[DLP11, DLP12], [DLP21, DLP22]]))
SLPp = -(DLP/boundary.weights).T*boundary.weights
M = SLPp.copy()
M[:boundary1.N,:boundary1.N] -= 0.5*np.eye(boundary1.N)
M[boundary1.N:,boundary1.N:] += 0.5*np.eye(boundary2.N)
MI = np.linalg.inv(M)
tau = MI.dot(bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([0,1], NG, [0,1], NG, mask=phys, x_endpoints=[False,False], y_endpoints=[False,False])

# evaluate at the target points
up = Modified_Helmholtz_Layer_Apply(boundary, gridp, k=helmholtz_k, charge=tau)
err_plot(up, solution_func)

################################################################################
# use the oversampling features

ftau = upsample_scalar(tau)
up = Modified_Helmholtz_Layer_Apply(fbdy, gridp, k=helmholtz_k, charge=ftau)
err_plot(up, solution_func)

################################################################################
# test the oversampling feature where its actually supposed to work

new_bdy = GSB(boundary2.x + boundary2.normal_x*h, boundary2.y + boundary2.normal_y*h)
ub = Modified_Helmholtz_Layer_Apply(fbdy, new_bdy, k=helmholtz_k, charge=ftau)
uba = solution_func(new_bdy.x, new_bdy.y)
print('Error a distance h from boundary: {:0.4e}'.format(np.abs(ub-uba).max()))

u = np.zeros_like(x)
u[phys] = up

ua = np.zeros_like(x)
ua[phys] = solution_func(x[phys], y[phys])


