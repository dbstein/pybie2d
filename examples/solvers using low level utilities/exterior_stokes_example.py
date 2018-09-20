import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
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

N = 250

# extract some functions for easy calling
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Compensated_Stokes_Form = pybie2d.boundaries.global_smooth_boundary.stokes_close_quad.Compensated_Stokes_Form
Compensated_Stokes_Apply = pybie2d.boundaries.global_smooth_boundary.stokes_close_quad.Compensated_Stokes_Apply
Pairing = pybie2d.pairing.Pairing

################################################################################
# define problem

# boundary
boundary = GSB(c=star(N,x=0,y=0,r=1.0,a=0.3,f=4,rot=np.pi/3.0))
boundary.add_module('Stokes_SLP_Self_Kress')
boundary.add_module('Stokes_Close_Quad')

# solution that is the evaluation of a point force of (1,1) at (0,0)
def solution_function(x, y):
	r = np.sqrt(x**2 + y**2)
	u = -np.log(r) + x*x/r**2 + x*y/r**2
	v = -np.log(r) + x*y/r**2 + y*y/r**2
	p = x/r**2 + y/r**2
	return u/(4*np.pi), v/(4*np.pi), p/(2*np.pi)
solution_func_u = lambda x, y: solution_function(x,y)[0]
solution_func_v = lambda x, y: solution_function(x,y)[1]

bcu, bcv, bcp = solution_function(boundary.x, boundary.y)
bc = np.concatenate([bcu, bcv])

def err_plot(up, func):
	# compute the error
	errorp = up - func(full_grid.xg[phys], full_grid.yg[phys])
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

full_grid = Grid([-2,2], N, [-2,2], N, periodic=True)
# this is hiding a lot of stuff!
ext, phys = boundary.find_interior_points(full_grid)
phys = full_grid.reshape(phys)
ext = full_grid.reshape(ext)

################################################################################
# solve for the density

ALP = Stokes_Layer_Singular_Form(boundary, ifdipole=True, ifforce=True)
A = 0.5*np.eye(2*boundary.N) + ALP
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

# evaluate at the target points
u = np.zeros_like(gridp.xg)
v = np.zeros_like(gridp.xg)
Up = Stokes_Layer_Apply(boundary, gridp, dipstr=tau, forces=tau, backend='FMM',
															out_type='stacked')
up = Up[0]
vp = Up[1]
u[phys] = up
v[phys] = vp

err_plot(up, solution_func_u)
err_plot(vp, solution_func_v)

################################################################################
# correct with close evaluation (preformed)

uph = up.copy()
vph = vp.copy()
# get distance to do close evaluation on from tolerance
close_distance = boundary.tolerance_to_distance(1.0e-16)
close_pts = gridp.find_near_points(boundary, close_distance)
close_trg = PointSet(gridp.x[close_pts], gridp.y[close_pts])
# generate close eval matrix
close_mat = Compensated_Stokes_Form(boundary, close_trg, 'e', do_DLP=True, do_SLP=True)
# generate naive matrix
naive_mat = Stokes_Layer_Form(boundary, close_trg, ifdipole=True, ifforce=True)
# construct close correction matrix
correction_mat = close_mat - naive_mat
# get correction
correction = correction_mat.dot(tau)

# find the correction and fix the naive solution
uph[close_pts] += correction[:close_trg.N]
vph[close_pts] += correction[close_trg.N:]

err_plot(uph, solution_func_u)
err_plot(vph, solution_func_v)

################################################################################
# correct with pair routines (preformed)

Up1 = Up.copy()
# to show how much easier the Pairing utility makes things
pair = Pairing(boundary, gridp, 'e', 1e-12)
code = pair.Setup_Close_Corrector(do_DLP=True, do_SLP=True, kernel='stokes', backend='preformed')
pair.Close_Correction(Up1.ravel(), tau, code)

err_plot(Up1[0], solution_func_u)
err_plot(Up1[1], solution_func_v)

################################################################################
# correct with close evaluation (on the fly)

uph = up.copy()
vph = vp.copy()
# generate close eval matrix
close_eval = Compensated_Stokes_Apply(boundary, close_trg, 'e', tau, do_DLP=True, do_SLP=True)
# generate naive matrix
naive_eval = Stokes_Layer_Apply(boundary, close_trg, forces=tau, dipstr=tau)
# construct close correction matrix
correction = close_eval - naive_eval

# find the correction and fix the naive solution
uph[close_pts] += correction[:close_trg.N]
vph[close_pts] += correction[close_trg.N:]

err_plot(uph, solution_func_u)
err_plot(vph, solution_func_v)

################################################################################
# correct with pair routines (on the fly)

Up1 = Up.copy()
# to show how much easier the Pairing utility makes things
code = pair.Setup_Close_Corrector(do_DLP=True, do_SLP=True, kernel='stokes')
pair.Close_Correction(Up1.ravel(), tau, code)

err_plot(Up1[0], solution_func_u)
err_plot(Up1[1], solution_func_v)


