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

N = 200

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Compensated_Stokes_Form = pybie2d.boundaries.global_smooth_boundary.stokes_close_quad.Compensated_Stokes_Form
Compensated_Stokes_DLP_Pressure_Form = pybie2d.boundaries.global_smooth_boundary.stokes_close_quad.Compensated_Stokes_DLP_Pressure_Form
Compensated_Stokes_DLP_Pressure_Apply = pybie2d.boundaries.global_smooth_boundary.stokes_close_quad.Compensated_Stokes_DLP_Pressure_Apply
Pairing = pybie2d.pairing.Pairing
Stokes_Pressure_Apply_FMM = pybie2d.kernels.low_level.stokes.Stokes_Pressure_Apply_FMM

################################################################################
# define problem

# boundary
boundary = GSB(c=squish(N,r=2,b=0.3,rot=np.pi/4.0))
boundary.add_module('Stokes_Close_Quad')
def solution_function(x, y):
	xc = -1.5
	yc = 1.5
	xd = x-xc
	yd = y-yc
	r = np.sqrt(xd**2 + yd**2)
	u = -np.log(r) + xd*xd/r**2 + xd*yd/r**2
	v = -np.log(r) + xd*yd/r**2 + yd*yd/r**2
	p = xd/r**2 + yd/r**2
	return u/(4*np.pi), v/(4*np.pi), p/(2*np.pi)
solution_func_u = lambda x, y: solution_function(x,y)[0]
solution_func_v = lambda x, y: solution_function(x,y)[1]
solution_func_p = lambda x, y: solution_function(x,y)[2]
bcu = solution_func_u(boundary.x, boundary.y)
bcv = solution_func_v(boundary.x, boundary.y)
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

full_grid = Grid([-2,2], N, [-2,2], N)
# this is hiding a lot of stuff!
phys, ext = boundary.find_interior_points(full_grid)
phys = full_grid.reshape(phys)
ext = full_grid.reshape(ext)

################################################################################
# solve for the density

DLP = Stokes_Layer_Singular_Form(boundary, ifdipole=True)
A = -0.5*np.eye(2*boundary.N) + DLP
# fix the nullspace
A[:,0] += np.concatenate([boundary.normal_x, boundary.normal_y])
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([-2,2], N, [-2,2], N, mask=phys)

# evaluate at the target points
u = np.zeros_like(gridp.xg)
v = np.zeros_like(gridp.xg)
Up = Stokes_Layer_Apply(boundary, gridp, dipstr=tau, backend='FMM',
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
close_distance = boundary.tolerance_to_distance(1e-12)
close_pts = gridp.find_near_points(boundary, close_distance)
close_trg = PointSet(gridp.x[close_pts], gridp.y[close_pts])
# generate close eval matrix
close_mat = Compensated_Stokes_Form(boundary, close_trg, 'i', do_DLP=True)
# generate naive matrix
naive_mat = Stokes_Layer_Form(boundary, close_trg, ifdipole=True)
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
# Compute the pressure

tau_stacked = tau.reshape([2, boundary.N])
p = Stokes_Pressure_Apply_FMM(boundary.get_stacked_boundary(), gridp.get_stacked_boundary(), dipstr=tau_stacked, dipvec=boundary.get_stacked_normal(), weights=boundary.weights)
close_mat = Compensated_Stokes_DLP_Pressure_Form(boundary, close_trg, 'i')
close_eval = close_mat.dot(tau)
# get constant adjustment
const_adjustment = close_eval[0] - solution_func_p(gridp.x[close_pts][0], gridp.y[close_pts][0])
p[close_pts] = close_eval
p -= const_adjustment
err_plot(p, solution_func_p)

tau_stacked = tau.reshape([2, boundary.N])
p = Stokes_Pressure_Apply_FMM(boundary.get_stacked_boundary(), gridp.get_stacked_boundary(), dipstr=tau_stacked, dipvec=boundary.get_stacked_normal(), weights=boundary.weights)
close_eval = Compensated_Stokes_DLP_Pressure_Apply(boundary, close_trg, 'i', tau)
# get constant adjustment
const_adjustment = close_eval[0] - solution_func_p(gridp.x[close_pts][0], gridp.y[close_pts][0])
p[close_pts] = close_eval
p -= const_adjustment
err_plot(p, solution_func_p)

