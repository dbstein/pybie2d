import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
plt.ion()
import pybie2d

"""
Demonstrate how to use the pybie2d package to solve an interior Laplace problem
On a complicated domain using a global quadrature

This example demonstrates how to do this entirely using low-level routines,
To demonstrate both how to use these low level routines
And to give you an idea what is going on under the hood in the
	higher level routines
"""

N = 1000

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Cauchy_Layer_Apply = pybie2d.kernels.high_level.cauchy.Cauchy_Layer_Apply
Compensated_Laplace_Full_Form = pybie2d.boundaries.global_smooth_boundary.Compensated_Laplace_Full_Form
Find_Near_Points = pybie2d.misc.near_points.find_near_points
Pairing = pybie2d.pairing.Pairing
Close_Corrector = pybie2d.close.Close_Corrector
Compensated_Laplace_Apply = pybie2d.boundaries.global_smooth_boundary.Compensated_Laplace_Apply

################################################################################
# define problem

# boundary
boundary = GSB(c=squish(N,r=2,b=0.3,rot=np.pi/4.0), 
											compute_differentiation_matrix=True)
# solution
solution_func = lambda x, y: 2*x + y
bc = solution_func(boundary.x, boundary.y)

def err_plot(up):
	# compute the error
	errorp = up - solution_func(full_grid.xg[phys], full_grid.yg[phys])
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
phys, ext = boundary.find_interior_points(full_grid)
phys = full_grid.reshape(phys)
ext = full_grid.reshape(ext)

################################################################################
# solve for the density

DLP = Laplace_Layer_Singular_Form(boundary, ifdipole=True)
A = -0.5*np.eye(boundary.N) + DLP
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

# evaluate at the target points
u = np.zeros_like(gridp.xg)
up = Laplace_Layer_Apply(boundary, gridp, dipstr=tau)
u[phys] = up

err_plot(up)

################################################################################
# correct with close evaluation (on the fly)

uph = up.copy()
# get distance to do close evaluation on from tolerance
close_distance = boundary.tolerance_to_distance(1e-12)
close_pts = gridp.find_near_points(boundary, close_distance)
close_trg = PointSet(gridp.x[close_pts], gridp.y[close_pts])
# generate close eval matrix
close_eval = Compensated_Laplace_Apply(boundary, close_trg, 'i', tau, \
				do_DLP=True, DLP_weight=None, do_SLP=False, SLP_weight=None)
# generate naive matrix
naive_eval = Laplace_Layer_Apply(boundary, close_trg, dipstr=tau)
# construct close correction matrix
correction = close_eval.real - naive_eval

# find the correction and fix the naive solution
uph[close_pts] += correction

err_plot(uph)

################################################################################
# correct with close evaluation (preformed)

if N <= 1000:
	uph = up.copy()
	# generate close eval matrix
	close_mat = Compensated_Laplace_Full_Form(boundary, close_trg, 'i', do_DLP=True, \
	                DLP_weight=None, do_SLP=False, SLP_weight=None, gradient=False)
	# generate naive matrix
	naive_mat = Laplace_Layer_Form(boundary, close_trg, ifdipole=True)
	# construct close correction matrix
	correction_mat = close_mat.real - naive_mat

	# find the correction and fix the naive solution
	correction = correction_mat.dot(tau)
	uph[close_pts] += correction

	err_plot(uph)

################################################################################
##### solve problem the easy way ###############################################
################################################################################

################################################################################
# find physical region

full_grid = Grid([-2,2], N, [-2,2], N, periodic=True)
phys, ext = boundary.find_interior_points(full_grid)
phys = full_grid.reshape(phys)
ext = full_grid.reshape(ext)

################################################################################
# solve for the density

DLP = Laplace_Layer_Singular_Form(boundary, ifdipole=True)
A = -0.5*np.eye(boundary.N) + DLP
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

# evaluate at the target points
up = Laplace_Layer_Apply(boundary, gridp, dipstr=tau)

################################################################################
# correct with pair routines (on the fly)

uph = up.copy()
# to show how much easier the Pairing utility makes things
pair = Pairing(boundary, gridp, 'i', 1e-12)
code2 = pair.Setup_Close_Corrector(do_DLP=True)
pair.Close_Correction(uph, tau, code2)

err_plot(uph)

################################################################################
# correct with pair routines (preformed)

if N <= 1000:
	uph = up.copy()
	# to show how much easier the Pairing utility makes things
	code1 = pair.Setup_Close_Corrector(do_DLP=True, backend='preformed')
	pair.Close_Correction(uph, tau, code1)

	err_plot(uph)

################################################################################
# generate a target that heads up to the boundary

px, py = boundary.x, boundary.y
nx, ny = boundary.normal_x, boundary.normal_y
adj = 1.0/10**np.arange(2,16)
tx = (px - nx*adj[:,None]).flatten()
ty = (py - ny*adj[:,None]).flatten()

approach_targ = PointSet(tx, ty)
sol = Compensated_Laplace_Apply(boundary, approach_targ, 'i', tau, do_DLP=True).real
true = solution_func(tx, ty)
err = np.abs(true-sol)
print('Error approaching boundary is: {:0.3e}'.format(err.max()))

