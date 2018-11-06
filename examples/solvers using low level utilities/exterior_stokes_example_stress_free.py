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
squish = pybie2d.misc.curve_descriptions.squished_circle
star = pybie2d.misc.curve_descriptions.star
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
boundary = GSB(c=star(N,x=0,y=0,r=1.0,a=0.3,f=4,rot=np.pi/3.0))
boundary.add_module('Stokes_SLP_Self_Traction')
boundary.add_module('Stokes_Close_Quad')
# solution
def solution_function(x, y):
	r = np.sqrt(x**2 + y**2)
	u = (-np.log(r) + x*(x+y)/r**2)/(4*np.pi)
	v = (-np.log(r) + y*(x+y)/r**2)/(4*np.pi)
	p = (x+y)/(2*np.pi*r**2)
	return u, v, p
def solution_function_stress(x, y):
	r = np.sqrt(x**2 + y**2)
	ux = ((x+y)/r**2 - 2*(x**3 + x**2*y)/r**4) / (4*np.pi)
	uy = ((x-y)/r**2 - 2*(x**2*y + x*y**2)/r**4) / (4*np.pi)
	vx = ((y-x)/r**2 - 2*(x**2*y + x*y**2)/r**4) / (4*np.pi)
	vy = ((x+y)/r**2 - 2*(y**3 + y**2*x)/r**4) / (4*np.pi)
	p = (x+y)/(2*np.pi*r**2)
	sxx = -p + 2*ux
	sxy = uy + vx
	syy = -p + 2*vy
	return sxx, sxy, syy
def solution_function_u(x, y):
	return solution_function(x,y)[0]
def solution_function_v(x, y):
	return solution_function(x,y)[1]
bsxx, bsxy, bsyy = solution_function_stress(boundary.x, boundary.y)
bcu = boundary.normal_x*bsxx + boundary.normal_y*bsxy
bcv = boundary.normal_x*bsxy + boundary.normal_y*bsyy
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
mm = phys.copy()
phys = full_grid.reshape(ext)
ext = full_grid.reshape(mm)

################################################################################
# solve for the density

TRA = boundary.Stokes_SLP_Self_Traction.Form()
A = -0.5*np.eye(2*boundary.N) + TRA
A[:,0] += np.concatenate([boundary.normal_x, boundary.normal_y])
tau = np.linalg.solve(A, bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([-2,2], N, [-2,2], N, mask=phys)

# evaluate at the target points
u = np.zeros_like(gridp.xg)
v = np.zeros_like(gridp.xg)
Up = Stokes_Layer_Apply(boundary, gridp, forces=tau, backend='fly',
															out_type='stacked')
up = Up[0]
vp = Up[1]
u[phys] = up
v[phys] = vp

err_plot(up, solution_function_u)
err_plot(vp, solution_function_v)

################################################################################
# correct with close evaluation (preformed)

uph = up.copy()
vph = vp.copy()
# get distance to do close evaluation on from tolerance
close_distance = boundary.tolerance_to_distance(1e-12)
close_pts = gridp.find_near_points(boundary, close_distance)
close_trg = PointSet(gridp.x[close_pts], gridp.y[close_pts])
# generate close eval matrix
close_mat = Compensated_Stokes_Form(boundary, close_trg, 'e', do_SLP=True)
# generate naive matrix
naive_mat = Stokes_Layer_Form(boundary, close_trg, ifforce=True)
# construct close correction matrix
correction_mat = close_mat - naive_mat
# get correction
correction = correction_mat.dot(tau)

# find the correction and fix the naive solution
uph[close_pts] += correction[:close_trg.N]
vph[close_pts] += correction[close_trg.N:]

err_plot(uph, solution_function_u)
err_plot(vph, solution_function_v)

