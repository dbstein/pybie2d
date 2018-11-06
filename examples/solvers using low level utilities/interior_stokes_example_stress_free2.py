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
Pairing = pybie2d.pairing.Pairing

################################################################################
# define problem

# boundary
boundary = GSB(c=squish(N,x=0.0,y=0.0,r=2,b=0.3,rot=np.pi/6.0))
boundary.add_module('Stokes_SLP_Self_Traction')
boundary.add_module('Stokes_Close_Quad')
# solution
def solution_function(x, y):
	locx = -3
	locy = -3
	dx = x - locx
	dy = y - locy
	r = np.sqrt(dx**2 + dy**2)
	u = (-np.log(r) + dx*(dx+dy)/r**2)/(4*np.pi)
	v = (-np.log(r) + dy*(dx+dy)/r**2)/(4*np.pi)
	p = (dx+dy)/(2*np.pi*r**2)
	return u, v, p
def solution_function_stress(x, y):
	locx = -3
	locy = -3
	dx = x - locx
	dy = y - locy
	r = np.sqrt(dx**2 + dy**2)
	ux = ((dx+dy)/r**2 - 2*(dx**3 + dx**2*dy)/r**4) / (4*np.pi)
	uy = ((dx-dy)/r**2 - 2*(dx**2*dy + dx*dy**2)/r**4) / (4*np.pi)
	vx = ((dy-dx)/r**2 - 2*(dx**2*dy + dx*dy**2)/r**4) / (4*np.pi)
	vy = ((dx+dy)/r**2 - 2*(dy**3 + dy**2*dx)/r**4) / (4*np.pi)
	p = (dx+dy)/(2*np.pi*r**2)
	sxx = -p + 2*ux
	sxy = uy + vx
	syy = -p + 2*vy
	return sxx, sxy, syy
def solution_function_u(x, y):
	return solution_function(x,y)[0]
def solution_function_v(x, y):
	return solution_function(x,y)[1]
bu, bv, bp = solution_function(boundary.x, boundary.y)
bsxx, bsxy, bsyy = solution_function_stress(boundary.x, boundary.y)
bcu = boundary.normal_x*bsxx + boundary.normal_y*bsxy
bcv = boundary.normal_x*bsxy + boundary.normal_y*bsyy
bc1 = bcu*boundary.tangent_x + bcv*boundary.tangent_y
bc2 = bu*boundary.normal_x + bv*boundary.normal_y
bc = np.concatenate([bc1, bc2])

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

W0 = boundary.Stokes_SLP_Self_Traction.Form()
W1 = W0 + 0.5*np.eye(2*boundary.N)
W2 = np.zeros([boundary.N, 2*boundary.N])
np.fill_diagonal(W2[:N,:N], boundary.tangent_x)
np.fill_diagonal(W2[:N,N:], boundary.tangent_y)
W3 = W2.dot(W1)
W4 = Stokes_Layer_Singular_Form(boundary, ifforce=True)
W5 = np.zeros([boundary.N, 2*boundary.N])
np.fill_diagonal(W5[:N,:N], boundary.normal_x)
np.fill_diagonal(W5[:N,N:], boundary.normal_y)
W6 = W5.dot(W4)
A = np.row_stack([W3, W6])

iA = np.linalg.pinv(A)
tau = iA.dot(bc)

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

# fix the nullspace issues
if True:
	ztrg = PointSet(x=np.array([0.0,1.0]), y=np.array([0.0,0.0]))
	ue0, ve0, pe0 = solution_function(ztrg.x, ztrg.y)
	close_mat = Compensated_Stokes_Form(boundary, ztrg, 'i', do_SLP=True)
	U0 = close_mat.dot(tau)
	u0, v0 = U0[:ztrg.N], U0[ztrg.N:]
	u0 = u0 + 1j*v0
	ue0 = ue0 + 1j*ve0
	du0 = ue0-u0
	uconst = du0[0]
	urot = (du0[1] - du0[0]).imag
	uadj = 1j*urot*gridp.c
	up += uconst.real + uadj.real
	vp += uconst.imag + uadj.imag

u[phys] = up
v[phys] = vp

def destroy_ticks(ax):
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xticklabels([])
	ax.set_yticklabels([])

ua = solution_function_u(gridp.xg, gridp.yg)
fig, [ax1, ax2] = plt.subplots(1,2)
clf1 = ax1.pcolormesh(gridp.xg, gridp.yg, np.ma.array(u, mask=ext))
clf2 = ax2.pcolormesh(gridp.xg, gridp.yg, np.ma.array(ua, mask=ext))
plt.colorbar(clf1, ax=ax1)
plt.colorbar(clf2, ax=ax2)
destroy_ticks(ax1)
destroy_ticks(ax2)
ax1.set_aspect('equal')
ax2.set_aspect('equal')

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
close_mat = Compensated_Stokes_Form(boundary, close_trg, 'i', do_SLP=True)
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

