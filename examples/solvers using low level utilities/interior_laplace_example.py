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

NB = 500
NG = 100

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Cauchy_Layer_Apply = pybie2d.kernels.high_level.cauchy.Cauchy_Layer_Apply
Compensated_Laplace_Form = pybie2d.boundaries.global_smooth_boundary.laplace_close_quad.Compensated_Laplace_Form
Find_Near_Points = pybie2d.misc.near_points.find_near_points
Pairing = pybie2d.pairing.Pairing
Compensated_Laplace_Apply = pybie2d.boundaries.global_smooth_boundary.laplace_close_quad.Compensated_Laplace_Apply

################################################################################
# define problem

# boundary
boundary = GSB(c=squish(NB,r=2,b=0.3,rot=np.pi/4.0))
boundary.add_module('Laplace_Close_Quad')
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

full_grid = Grid([-2,2], NG, [-2,2], NG)
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
gridp = Grid([-2,2], NG, [-2,2], NG, mask=phys)

# evaluate at the target points
u = np.zeros_like(gridp.xg)
up = Laplace_Layer_Apply(boundary, gridp, dipstr=tau)
u[phys] = up

err_plot(up)

################################################################################
# correct with close evaluation (on the fly)

uph = up.copy()
# get distance to do close evaluation on from tolerance
close_distance = boundary.tolerance_to_distance(1.0e-12)
# close_pts = gridp.find_near_points(boundary, close_distance)
close_pts = gridp.find_near_points(boundary, boundary.max_h*5.7)
close_trg = PointSet(gridp.x[close_pts], gridp.y[close_pts])
# generate close eval matrix
close_eval = Compensated_Laplace_Apply(boundary, close_trg, 'i', tau, do_DLP=True)
# generate naive matrix
naive_eval = Laplace_Layer_Apply(boundary, close_trg, dipstr=tau)
# construct close correction matrix
correction = close_eval.real - naive_eval
close_pts = gridp.find_near_points(boundary, boundary.max_h*5.7)
# find the correction and fix the naive solution
uph[close_pts] += correction

err_plot(uph)

################################################################################
# correct with close evaluation (preformed)

if NG <= 1000:
	uph = up.copy()
	# generate close eval matrix
	close_mat = Compensated_Laplace_Form(boundary, close_trg, 'i', do_DLP=True)
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

full_grid = Grid([-2,2], NG, [-2,2], NG)
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
gridp = Grid([-2,2], NG, [-2,2], NG, mask=phys)

# evaluate at the target points
up = Laplace_Layer_Apply(boundary, gridp, dipstr=tau)

################################################################################
# correct with pair routines (on the fly)

uph = up.copy()
# to show how much easier the Pairing utility makes things
pair = Pairing(boundary, gridp, 'i', 1e-12)
code2 = pair.Setup_Close_Corrector('laplace', do_DLP=True)
pair.Close_Correction(uph, tau, code2)

err_plot(uph)

################################################################################
# correct with pair routines (preformed)

if NG <= 1000:
	uph = up.copy()
	# to show how much easier the Pairing utility makes things
	code1 = pair.Setup_Close_Corrector('laplace', do_DLP=True, backend='preformed')
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
MAT = Compensated_Laplace_Form(boundary, approach_targ, 'i', do_DLP=True).real
sol = MAT.dot(tau)
true = solution_func(tx, ty)
err = np.abs(true-sol)
print('Error approaching boundary (form) is: {:0.3e}'.format(err.max()))

sol = Compensated_Laplace_Apply(boundary, approach_targ, 'i', tau, do_DLP=True).real
err = np.abs(true-sol)
print('Error approaching boundary (apply) is: {:0.3e}'.format(err.max()))

################################################################################
# check gradients

close_mat, dxm, dym = Compensated_Laplace_Form(boundary, close_trg, 'i',
													do_DLP=True, gradient=True)

uxt = 2.0*np.ones(close_trg.N)
uyt = np.ones(close_trg.N)
uxe = dxm.dot(tau)
uye = dym.dot(tau)

err_ux = np.abs(uxe-uxt)
err_uy = np.abs(uye-uyt)
print('Error in ux form is:  {:0.3e}'.format(err_ux.max()))
print('Error in uy form is:  {:0.3e}'.format(err_uy.max()))

ue2, uxe2, uye2 = Compensated_Laplace_Apply(boundary, close_trg, 'i', tau,
												do_DLP=True, gradient=True)

err_ux = np.abs(uxe2-uxt)
err_uy = np.abs(uye2-uyt)
print('Error in ux apply is: {:0.3e}'.format(err_ux.max()))
print('Error in uy apply is: {:0.3e}'.format(err_uy.max()))

################################################################################
# check gradients approaching the boundary

print('Error in gradients as you approach boundary:')

close_mat, dxm, dym = Compensated_Laplace_Form(boundary, approach_targ, 'i',
													do_DLP=True, gradient=True)

uxt = 2.0*np.ones(approach_targ.N)
uyt = np.ones(approach_targ.N)
uxe = dxm.dot(tau)
uye = dym.dot(tau)

err_ux = np.abs(uxe-uxt)
err_uy = np.abs(uye-uyt)
print('    Error in ux form is:  {:0.3e}'.format(err_ux.max()))
print('    Error in uy form is:  {:0.3e}'.format(err_uy.max()))

ue2, uxe2, uye2 = Compensated_Laplace_Apply(boundary, approach_targ, 'i', tau,
													do_DLP=True, gradient=True)

err_ux = np.abs(uxe2-uxt)
err_uy = np.abs(uye2-uyt)
print('    Error in ux apply is: {:0.3e}'.format(err_ux.max()))
print('    Error in uy apply is: {:0.3e}'.format(err_uy.max()))

# Error scaling as you approach the boundary
# from formations:
nb    = np.array([ 100,       200,       400,       800,       1600,      3200      ])
Eu_f  = np.array([ 2.487e-14, 6.750e-14, 1.181e-13, 2.389e-13, 4.947e-13, 1.148e-12 ])
Eux_f = np.array([ 7.176e-13, 4.718e-12, 1.547e-11, 6.872e-11, 2.224e-10, 1.043e-09 ])
# from applies:
Eu_a  = np.array([ 2.842e-14, 5.507e-14, 6.839e-14, 1.945e-13, 4.077e-13, 9.193e-13 ])
Eux_a = np.array([ 5.200e-13, 3.403e-12, 6.736e-12, 5.377e-11, 1.498e-10, 7.335e-10 ])

mpl.rc('text', usetex=True)
fig, ax = plt.subplots(1,1)
ax.plot(nb, Eu_f, color='black', linewidth=2, label=r'$u$, formed')
ax.plot(nb, Eu_a, color='blue', linewidth=2, label=r'$u$, applied')
ax.plot(nb, Eux_f, color='black', linestyle='--', linewidth=2, label=r'$\partial_x u$, formed')
ax.plot(nb, Eux_a, color='blue', linestyle='--', linewidth=2, label=r'$\partial_x u$, applied')
ax.plot(nb, nb*1.5e-16, color='gray', label=r'$\mathcal{O}(n_b)$')
ax.plot(nb, nb**2*3.5e-18, color='gray', linestyle='--', label=r'$\mathcal{O}(n_b^2)$')
ax.plot(nb, nb**2*3e-17, color='gray', linestyle='--')
ax.plot(nb, nb**3*1e-18, color='gray', linestyle=':', label=r'$\mathcal{O}(n_b^3)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Number of boundary points ($n_b$)')
ax.set_ylabel(r'$L^\infty$ error approaching boundary points')
ax.set_title(r'Interior Problem (DLP only)')
ax.legend()
