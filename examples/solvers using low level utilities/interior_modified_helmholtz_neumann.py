import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
plt.ion()
import pybie2d

"""o solve an interior Modified Helmholtz problem
On a complicated domain using a global quadr
Demonstrate how to use the pybie2d package tature

This example demonstrates how to do this entirely using low-level routines,
To demonstrate both how to use these low level routines
And to give you an idea what is going on under the hood in the
	higher level routines
"""

NB = 600
NG = 200
helmholtz_k = 30.0

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
bc = _solution_func_dn(boundary.x, boundary.y, boundary.normal_x, boundary.normal_y)
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

DLP = boundary.Modified_Helmholtz_DLP_Self_Form(helmholtz_k)
SLPp = -(DLP/boundary.weights).T*boundary.weights
A = -0.5*np.eye(boundary.N) + SLPp
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
# correct with close evaluation (preformed)

uph = up.copy()
close_distance = boundary.tolerance_to_distance(1.0e-12)
close_pts = gridp.find_near_points(boundary, boundary.max_h*5.7)
close_trg = PointSet(gridp.x[close_pts], gridp.y[close_pts])
close_corrector = boundary.Get_Close_Corrector('modified_helmholtz', close_trg, 'i', k=helmholtz_k, do_SLP=True)
close_corrector(uph, tau, close_pts)
err_plot(uph)

################################################################################
# correct with pair routines (preformed)

uph = up.copy()
# to show how much easier the Pairing utility makes things
pair = Pairing(boundary, gridp, 'i', 1e-12)
code2 = pair.Setup_Close_Corrector('modified_helmholtz', k=helmholtz_k, do_SLP=True)
pair.Close_Correction(uph, tau, code2)
err_plot(uph)

################################################################################
# check on near boundary grid

adjr = np.arange(10)[::-1]
adj = 1/10**adjr*boundary.max_h
close_x = boundary.x - boundary.normal_x*adj[:,None]
close_y = boundary.y - boundary.normal_y*adj[:,None]
trg = PointSet(x=close_x.flatten(), y=close_y.flatten())

up = Modified_Helmholtz_Layer_Apply(boundary, trg, helmholtz_k, charge=tau)
uph = up.copy()
# to show how much easier the Pairing utility makes things
pair = Pairing(boundary, trg, 'i', 1e-12)
code2 = pair.Setup_Close_Corrector('modified_helmholtz', k=helmholtz_k, do_SLP=True)
pair.Close_Correction(uph, tau, code2)

uph = uph.reshape([adjr.shape[0], boundary.N])
ua = solution_func(close_x, close_y)
print(np.max(np.abs(ua-uph),1))



