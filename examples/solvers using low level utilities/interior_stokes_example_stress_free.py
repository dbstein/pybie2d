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

N = 100

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Compensated_Stokes_Form = pybie2d.boundaries.global_smooth_boundary.stokes_close_quad.Compensated_Stokes_Form
Pairing = pybie2d.pairing.Pairing

################################################################################
# define problem

# boundary
boundary = GSB(c=squish(N,r=2,b=0.3,rot=np.pi/4.0))
boundary.add_module('Stokes_SLP_Self_Traction')
boundary.add_module('Stokes_Close_Quad')
# solution
solution_func_u = lambda x, y: 2*y# + x**2
solution_func_v = lambda x, y: 0.5*x
solution_func_p = lambda x, y: 0.0
solution_stress_xx = lambda x, y: 2*x*0.0
solution_stress_xy = lambda x, y: -5.0/2.0*np.ones_like(x)
solution_stress_yy = lambda x, y: 0.0*np.ones_like(x)
bstress_xx = solution_stress_xx(boundary.x, boundary.y)
bstress_xy = solution_stress_xy(boundary.x, boundary.y)
bstress_yy = solution_stress_yy(boundary.x, boundary.y)
bcu = boundary.normal_x*bstress_xx + boundary.normal_y*bstress_xy
bcv = boundary.normal_x*bstress_xy + boundary.normal_y*bstress_yy
bc = -np.concatenate([bcu, bcv])

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

TRA = boundary.Stokes_SLP_Self_Traction.Form()
A = 0.5*np.eye(2*boundary.N) + TRA
# fix the nullspaces
A[:,0] = np.concatenate([boundary.normal_x, boundary.normal_y])
A[:boundary.N,1] = 1.0
A[boundary.N:,1] = 0.0
A[:boundary.N,2] = 0.0
A[boundary.N:,2] = 1.0
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([-2,2], N, [-2,2], N, mask=phys)

# evaluate at the target points
u = np.zeros_like(gridp.xg)
v = np.zeros_like(gridp.xg)
Up = Stokes_Layer_Apply(boundary, gridp, forces=tau, backend='FMM',
															out_type='stacked')
up = Up[0]
vp = Up[1]
u[phys] = up
v[phys] = vp

err_plot(up, solution_func_u)
err_plot(vp, solution_func_v)

ua = np.zeros_like(gridp.xg)
ua[phys] = solution_func_u(gridp.x, gridp.y)

fig, [ax1,ax2] = plt.subplots(1,2)
ax1.imshow(u)
ax2.imshow(ua)

################################################################################
# correct with close evaluation (preformed)

uph = up.copy()
vph = vp.copy()
# get distance to do close evaluation on from tolerance
close_distance = boundary.tolerance_to_distance(1e-12)
close_pts = gridp.find_near_points(boundary, close_distance)
close_trg = PointSet(gridp.x[close_pts], gridp.y[close_pts])
# generate close eval matrix
close_mat = Compensated_Stokes_Form(boundary, close_trg, 'i', do_SLP=True)
# generate naive matrix
naive_mat = Stokes_Layer_Form(boundary, close_trg, ifforces=True)
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
##### solve problem the easy way ###############################################
################################################################################

################################################################################
# find physical region

full_grid = Grid([-2,2], N, [-2,2], N)
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
Up = Stokes_Layer_Apply(boundary, gridp, dipstr=tau, backend='FMM',
															out_type='stacked')

################################################################################
# correct with pair routines (on the fly)

Up1 = Up.copy()
# to show how much easier the Pairing utility makes things
pair = Pairing(boundary, gridp, 'i', 1e-12)
code2 = pair.Setup_Close_Corrector(do_DLP=True, kernel='stokes')
pair.Close_Correction(Up1.ravel(), tau, code2)
up = Up1[0]
vp = Up1[1]

err_plot(up, solution_func_u)
err_plot(vp, solution_func_v)

################################################################################
# correct with pair routines (preformed)

Up1 = Up.copy()
# to show how much easier the Pairing utility makes things
code1 = pair.Setup_Close_Corrector(do_DLP=True, kernel='stokes', backend='preformed')
pair.Close_Correction(Up1.ravel(), tau, code1)

up = Up1[0]
vp = Up1[1]

err_plot(up, solution_func_u)
err_plot(vp, solution_func_v)

################################################################################
# generate a target that heads up to the boundary

px, py = boundary.x, boundary.y
nx, ny = boundary.normal_x, boundary.normal_y
adj = 1.0/10**np.arange(2,16)
tx = (px - nx*adj[:,None]).flatten()
ty = (py - ny*adj[:,None]).flatten()

approach_targ = PointSet(tx, ty)
mat = Compensated_Stokes_Form(boundary, approach_targ, 'i', do_DLP=True).real
sol = mat.dot(tau)
sol_u = sol[:approach_targ.N]
sol_v = sol[approach_targ.N:]
true_u = solution_func_u(tx, ty)
true_v = solution_func_v(tx, ty)
err_u = np.abs(true_u-sol_u)
err_v = np.abs(true_v-sol_v)
print('Error approaching boundary in u is: {:0.3e}'.format(err_u.max()))
print('Error approaching boundary in v is: {:0.3e}'.format(err_v.max()))

