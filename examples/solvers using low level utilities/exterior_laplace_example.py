import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import numexpr as ne
plt.ion()
import pybie2d

"""
Demonstrate how to use the pybie2d package to solve an exterior Laplace problem
On a complicated domain using a global quadrature

This example demonstrates how to do this entirely using low-level routines,
To demonstrate both how to use these low level routines
And to give you an idea what is going on under the hood in the
	higher level routines
"""

N  = 100  # discretization for test grid
NB = 200  # discretization of boundary

# extract some functions for easy calling
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet
Grid = pybie2d.grid.Grid
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Cauchy_Layer_Apply = pybie2d.kernels.high_level.cauchy.Cauchy_Layer_Apply
Compensated_Laplace_Apply = pybie2d.boundaries.global_smooth_boundary.Compensated_Laplace_Apply
Compensated_Laplace_Full_Form = pybie2d.boundaries.global_smooth_boundary.Compensated_Laplace_Full_Form
Pairing = pybie2d.pairing.Pairing

################################################################################
# define problem

# boundary
boundary = GSB(c=star(NB,x=0,y=0,r=1.0,a=0.4,f=3,rot=np.pi/3.0))

# point at which to have a source
pt_source_location = 0.0 + 0.0j
pt_source_strength = 1.0

# solution
def solution_func(x, y):
	d2 = (x-pt_source_location.real)**2 + (y-pt_source_location.imag)**2
	return ne.evaluate('log(sqrt(d2))')
# grid on which to test solutions
v = np.linspace(-2, 2, N, endpoint=False)
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

full_grid = Grid([-2,2], N, [-2,2], N, periodic=True)
# fast computation of winding number via cauchy sums
#	for smallish computations, accelerated via numba
#	for large computations, accelerated via FMM
wn = Cauchy_Layer_Apply(boundary, full_grid, dipstr=np.ones(boundary.N))
bad = np.logical_or(np.isnan(wn), np.isinf(wn))
bad = np.logical_or(np.abs(wn) > 1e5, bad)
wn[bad] = 1.0
winding_number = np.abs(wn).reshape(x.shape)
# find points where that sum was not accurate enough
# 	this uses trees built by the init functions for boundary, and full_grid
dist = boundary.tolerance_to_distance(1e-2)
q = full_grid.find_near_points(boundary, dist)
# set winding number in this close region to 0
winding_number[q] = 0.0
# generate ext array that's good except in near boundary region
ext = winding_number > 0.5
# brute force search for close region using matplotlib.path
# this should perhaps be changed to finding local coordinates via newtons method
poly = mpl.path.Path(boundary.get_stacked_boundary(T=False))
interior = poly.contains_points(np.column_stack([x[q], y[q]]))
# fix up close region
ext[q] = interior
ext[bad.reshape(ext.shape)] = True
# get ext
phys = np.logical_not(ext)

################################################################################
# solve for the density

ALP = Laplace_Layer_Singular_Form(boundary, ifcharge=True, ifdipole=True)
A = 0.5*np.eye(boundary.N) + ALP
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

# evaluate at the target points
u = np.zeros_like(x)
up = Laplace_Layer_Apply(boundary, gridp, charge=tau, dipstr=tau)
u[phys] = up

err_plot(up)

################################################################################
# correct with close evaluation (preformed)

uph = up.copy()
# get distance to do close evaluation on from tolerance
close_distance = boundary.tolerance_to_distance(1e-12)
close_pts = gridp.find_near_points(boundary, close_distance).ravel()
close_trg = PointSet(gridp.x[close_pts], gridp.y[close_pts])
# generate close eval matrix
close_mat = Compensated_Laplace_Full_Form(boundary, close_trg, 'e',
								do_DLP=True, do_SLP=True, gradient=False)
# generate naive matrix
naive_mat = Laplace_Layer_Form(boundary, close_trg, ifdipole=True, ifcharge=True)
# construct close correction matrix
correction_mat = close_mat.real - naive_mat

# find the correction and fix the naive solution
correction = correction_mat.dot(tau)
uph[close_pts] += correction

err_plot(uph)

################################################################################
# correct with close evaluation (on the fly)

uph = up.copy()
# generate close eval matrix
close_eval = Compensated_Laplace_Apply(boundary, close_trg, 'e', tau, \
												do_DLP=True, do_SLP=True)
# generate naive matrix
naive_eval = Laplace_Layer_Apply(boundary, close_trg, charge=tau, dipstr=tau)
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

full_grid = Grid([-2,2], N, [-2,2], N, periodic=True)
phys2, ext2 = boundary.find_interior_points(full_grid)
phys2 = full_grid.reshape(phys2)
ext2 = full_grid.reshape(ext2)

################################################################################
# solve for the density

ALP = Laplace_Layer_Singular_Form(boundary, ifcharge=True, ifdipole=True)
A = 0.5*np.eye(boundary.N) + ALP
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

# evaluate at the target points
up = Laplace_Layer_Apply(boundary, gridp, charge=tau, dipstr=tau)

################################################################################
# correct with pair routines (full preformed)

uph = up.copy()
# to show how much easier the Pairing utility makes things
pair = Pairing(boundary, gridp, 'e', 1e-15)
code1 = pair.Setup_Close_Corrector(do_DLP=True, do_SLP=True, \
													backend='full preformed')
pair.Close_Correction(uph, tau, code1)

err_plot(uph)

################################################################################
# correct with pair routines (preformed)

# uph = up.copy()
# # to show how much easier the Pairing utility makes things
# code2 = pair.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='preformed')
# pair.Close_Correction(uph, tau, code2)

# err_plot(uph)

################################################################################
# correct with pair routines (on the fly)

uph = up.copy()
# to show how much easier the Pairing utility makes things
code3 = pair.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='fly')
pair.Close_Correction(uph, tau, code3)

err_plot(uph)

################################################################################
# generate a target that heads up to the boundary

px, py = boundary.x, boundary.y
nx, ny = boundary.normal_x, boundary.normal_y
adj = 1.0/10**np.arange(2,16)
tx = (px + nx*adj[:,None]).flatten()
ty = (py + ny*adj[:,None]).flatten()

approach_targ = PointSet(tx, ty)
mat = Compensated_Laplace_Full_Form(boundary, approach_targ, 'e', do_DLP=True, do_SLP=True)
# sol = Compensated_Laplace_Apply(boundary, approach_targ, 'e', tau, do_DLP=True, do_SLP=True)
sol = mat.dot(tau)
true = solution_func(tx, ty)
err = np.abs(true-sol)
print('Error approaching boundary (form) is: {:0.3e}'.format(err.max()))

sol = Compensated_Laplace_Apply(boundary, approach_targ, 'e', tau, do_DLP=True, do_SLP=True)
true = solution_func(tx, ty)
err = np.abs(true-sol)

print('Error approaching boundary (apply) is: {:0.3e}'.format(err.max()))

################################################################################
# check gradients

close_mat, dxm, dym = Compensated_Laplace_Full_Form(boundary, close_trg, 'e',
										do_DLP=True, do_SLP=True, gradient=True)

uxt = close_trg.x/(close_trg.x**2 + close_trg.y**2)
uyt = close_trg.y/(close_trg.x**2 + close_trg.y**2)
uxe = dxm.dot(tau)
uye = dym.dot(tau)

err_ux = np.abs(uxe-uxt)
err_uy = np.abs(uye-uyt)
print('Error in ux form is:  {:0.3e}'.format(err_ux.max()))
print('Error in uy form is:  {:0.3e}'.format(err_uy.max()))

ue2, uxe2, uye2 = Compensated_Laplace_Apply(boundary, close_trg, 'e', tau,
										do_DLP=True, do_SLP=True, gradient=True)

err_ux = np.abs(uxe2-uxt)
err_uy = np.abs(uye2-uyt)
print('Error in ux apply is: {:0.3e}'.format(err_ux.max()))
print('Error in uy apply is: {:0.3e}'.format(err_uy.max()))

################################################################################
# check gradients approaching the boundary

print('Error in gradients as you approach boundary:')

close_mat, dxm, dym = Compensated_Laplace_Full_Form(boundary, approach_targ, 'e',
										do_DLP=True, do_SLP=True, gradient=True)

uxt = approach_targ.x/(approach_targ.x**2 + approach_targ.y**2)
uyt = approach_targ.y/(approach_targ.x**2 + approach_targ.y**2)
uxe = dxm.dot(tau)
uye = dym.dot(tau)

err_ux = np.abs(uxe-uxt)
err_uy = np.abs(uye-uyt)
print('    Error in ux form is:  {:0.3e}'.format(err_ux.max()))
print('    Error in uy form is:  {:0.3e}'.format(err_uy.max()))

ue2, uxe2, uye2 = Compensated_Laplace_Apply(boundary, approach_targ, 'e', tau,
										do_DLP=True, do_SLP=True, gradient=True)

err_ux = np.abs(uxe2-uxt)
err_uy = np.abs(uye2-uyt)
print('    Error in ux apply is: {:0.3e}'.format(err_ux.max()))
print('    Error in uy apply is: {:0.3e}'.format(err_uy.max()))

# Error scaling as you approach the boundary
# from formations:
nb    = np.array([ 150,       300,       600,       1200,      2400      ])
Eu_f  = np.array([ 7.883e-15, 1.560e-14, 4.907e-14, 1.188e-13, 2.467e-13 ])
Eux_f = np.array([ 4.835e-13, 2.220e-12, 1.529e-11, 7.742e-11, 3.819e-10 ])
# from applies:
Eu_a  = np.array([ 7.383e-15, 1.243e-14, 4.630e-14, 1.076e-13, 2.488e-13 ])
Eux_a = np.array([ 6.115e-13, 1.540e-12, 1.738e-11, 5.748e-11, 3.926e-10 ])

mpl.rc('text', usetex=True)
fig, ax = plt.subplots(1,1)
ax.plot(nb, Eu_f, color='black', linewidth=2, label=r'$u$, formed')
ax.plot(nb, Eu_a, color='blue', linewidth=2, label=r'$u$, applied')
ax.plot(nb, Eux_f, color='black', linestyle='--', linewidth=2, label=r'$\partial_x u$, formed')
ax.plot(nb, Eux_a, color='blue', linestyle='--', linewidth=2, label=r'$\partial_x u$, applied')
ax.plot(nb, nb*3e-17, color='gray', label=r'$\mathcal{O}(n_b)$')
ax.plot(nb, nb**2*5e-19, color='gray', linestyle='--', label=r'$\mathcal{O}(n_b^2)$')
ax.plot(nb, nb**2*1.5e-17, color='gray', linestyle='--')
ax.plot(nb, nb**3*2e-19, color='gray', linestyle=':', label=r'$\mathcal{O}(n_b^3)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Number of boundary points ($n_b$)')
ax.set_ylabel(r'$L^\infty$ error approaching boundary points')
ax.set_title(r'Exterior Problem (SLP+DLP)')
ax.legend()


