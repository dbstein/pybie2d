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

N = 100

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet
Laplace_Layer_Form = pybie2d.kernels.laplace.Laplace_Layer_Form
Laplace_Layer_Self_Form = pybie2d.kernels.laplace.Laplace_Layer_Self_Form
Laplace_Layer_Apply = pybie2d.kernels.laplace.Laplace_Layer_Apply
Cauchy_Layer_Apply = pybie2d.kernels.cauchy.Cauchy_Layer_Apply
Compensated_Laplace_Full_Form = pybie2d.boundaries.global_smooth_boundary.Compensated_Laplace_Full_Form
Find_Near_Points = pybie2d.misc.near_points.find_near_points
Pairing = pybie2d.pairing.Pairing
Close_Corrector = pybie2d.close.Close_Corrector
Compensated_Laplace_Apply = pybie2d.boundaries.global_smooth_boundary.Compensated_Laplace_Apply

################################################################################
# define problem

# boundary
boundary = GSB(c=squish(N,r=2,b=0.3,rot=np.pi/4.0), compute_kress_mats=False,
											compute_differentiation_matrix=True)
# solution
solution_func = lambda x, y: 2*x + y
# grid on which to test solutions
v = np.linspace(-2, 2, N, endpoint=True)
x, y = np.meshgrid(v, v, indexing='ij')
solution = solution_func(x, y)
# get boundary condition
bc = solution_func(boundary.x, boundary.y)

def err_plot(up):
	# compute the error
	errorp = up - solution_func(x[phys], y[phys])
	digitsp = -np.log10(np.abs(errorp)+1e-16)
	digits = np.zeros_like(x)
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

full_grid = PointSet(x, y)
# fast computation of winding number via cauchy sums
#	for smallish computations, accelerated via numba
#	for large computations, accelerated via FMM
wn = Cauchy_Layer_Apply(boundary, full_grid, dipstr=boundary.ones_vec)
winding_number = np.abs(wn).reshape(x.shape)
# find points where that sum was not accurate enough
# 	this uses trees built by the init functions for boundary, and full_grid
dist = boundary.tolerance_to_distance(1e-2)
q = Find_Near_Points(boundary, full_grid, dist).reshape(x.shape)
# set winding number in this close region to 0
winding_number[q] = 0.0
# generate phys array that's good except in near boundary region
phys = winding_number > 0.5
# brute force search for close region using matplotlib.path
# this should perhaps be changed to finding local coordinates via newtons method
poly = mpl.path.Path(boundary.stacked_boundary)
interior = poly.contains_points(np.column_stack([x[q], y[q]]))
# fix up close region
phys[q] = interior
# get ext
ext = np.logical_not(phys)

################################################################################
# solve for the density

DLP = Laplace_Layer_Self_Form(boundary, ifdipole=True, self_type='singular')
A = -0.5*np.eye(boundary.N) + DLP
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = PointSet(x[phys], y[phys])

# evaluate at the target points
u = np.zeros_like(x)
up = Laplace_Layer_Apply(boundary, gridp, dipstr=tau, backend='FMM')
u[phys] = up

err_plot(up)

################################################################################
# correct with close evaluation (preformed)

uph = up.copy()
# get distance to do close evaluation on from tolerance
close_distance = boundary.tolerance_to_distance(1e-12)
close_pts = Find_Near_Points(boundary, gridp, r=close_distance)
close_trg = PointSet(gridp.x[close_pts], gridp.y[close_pts])
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
# correct with close evaluation (on the fly)

uph = up.copy()
# get distance to do close evaluation on from tolerance
close_distance = boundary.tolerance_to_distance(1e-12)
close_pts = Find_Near_Points(boundary, gridp, r=close_distance)
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
##### solve problem the easy way ###############################################
################################################################################

################################################################################
# find physical region

full_grid = PointSet(x, y)
phys2, ext2 = boundary.find_interior_points(full_grid)
phys2 = full_grid.reshape(phys2)
ext2 = full_grid.reshape(ext2)

################################################################################
# solve for the density

DLP = Laplace_Layer_Self_Form(boundary, ifdipole=True, self_type='singular')
A = -0.5*np.eye(boundary.N) + DLP
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = PointSet(x[phys], y[phys])

# evaluate at the target points
up = Laplace_Layer_Apply(boundary, gridp, dipstr=tau)

################################################################################
# correct with pair routines (preformed)

uph = up.copy()
# to show how much easier the Pairing utility makes things
pair = Pairing(boundary, gridp, 'i', 1e-12)
code1 = pair.Setup_Close_Corrector(do_DLP=True, backend='preformed')
pair.Close_Correction(uph, tau, code1)

err_plot(uph)

################################################################################
# correct with pair routines (on the fly)

uph = up.copy()
# to show how much easier the Pairing utility makes things
code2 = pair.Setup_Close_Corrector(do_DLP=True, backend='numba')
pair.Close_Correction(uph, tau, code2)

err_plot(uph)
