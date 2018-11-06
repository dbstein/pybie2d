import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import time
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

NG = 200
h_max = 1.0/NG

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
PPB = pybie2d.boundaries.panel_polygon_boundary.panel_polygon_boundary.Panel_Polygon_Boundary
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Boundary_Collection = pybie2d.boundaries.collection.BoundaryCollection
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply

################################################################################
# define problem

# boundary
boundary1 = PPB([0,1,1,0], [0,0,1,1], [h_max]*4, [True]*4, order=16, dyadic_levels=16, dyadic_base=4)
boundary2 = GSB(c=squish(2*int(1.5*NG/2),x=0.5,y=0.5,r=0.25,b=1.0,rot=0.0))
boundary = Boundary_Collection()
boundary.add([boundary1, boundary2], ['i', 'e'])
boundary.amass_information()
# get oversampled boundary for close eval and interpolation routines
upsample = lambda f: sp.signal.resample(f, 6*f.shape[0])
h = h_max
fbdy1, IMAT = boundary1.prepare_oversampling(h/6.0)
IMAT = sp.sparse.csr_matrix(IMAT)
fbdy2 = GSB(c=upsample(boundary2.c))
fbdy = Boundary_Collection()
fbdy.add([fbdy1, fbdy2], ['i', 'e'])
try:
	fbdy.amass_information()
except:
	pass
def upsample_scalar(f):
	f1 = f[:boundary1.N]
	f2 = f[boundary1.N:]
	return np.concatenate([ IMAT.dot(f1), upsample(f2) ])
def upsample_vector(f):
	fu = f[:boundary.N]
	fv = f[boundary.N:]
	return np.concatenate([ upsample_scalar(fu), upsample_scalar(fv) ])

# solution
solution_func_u = lambda x, y: 2*y + x
solution_func_v = lambda x, y: 0.5*x - y
bcu1 = solution_func_u(boundary1.x, boundary1.y)
bcu2 = solution_func_u(boundary2.x, boundary2.y)
bcv1 = solution_func_v(boundary1.x, boundary1.y)
bcv2 = solution_func_v(boundary2.x, boundary2.y)
bc1 = np.concatenate([bcu1, bcv1])
bc2 = np.concatenate([bcu2, bcv2])
bc = np.concatenate([bc1, bc2])
# bcu = solution_func_u(boundary.x, boundary.y)
# bcv = solution_func_v(boundary.x, boundary.y)
# bc = np.concatenate([bcu, bcv])

def err_plot(up, func):
	# compute the error
	errorp = up - func(full_grid.xg[phys], full_grid.yg[phys])
	digitsp = -np.log10(np.abs(errorp)+1e-16)
	digits = np.zeros_like(full_grid.xg)
	digits[phys] = digitsp
	mdigits = np.ma.array(digits, mask=ext)

	# plot the error as a function of space (only good in interior)
	fig, ax = plt.subplots(1,1)
	clf = ax.imshow(mdigits[:,::-1].T, extent=[0,1,0,1],
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

full_grid = Grid([0,1], NG, [0,1], NG, x_endpoints=[False,False], y_endpoints=[False,False])
x = full_grid.xg
y = full_grid.yg

phys = (x-0.5)**2 + (y-0.5)**2 > 0.25**2
ext = np.logical_not(phys)

good_evals = (x-0.5)**2 + (y-0.5)**2 > (0.25+boundary2.max_h)**2

################################################################################
# solve for the density

SS11 = -0.5*np.eye(2*boundary1.N) + Stokes_Layer_Singular_Form(boundary1, ifdipole=True)
SS12 = Stokes_Layer_Form(boundary1, boundary2, ifdipole=True)
SS21 = Stokes_Layer_Form(boundary2, boundary1, ifdipole=True, ifforce=True)
SS22 = 0.5*np.eye(2*boundary2.N) + Stokes_Layer_Singular_Form(boundary2, ifdipole=True, ifforce=True)
Nxx = boundary1.normal_x[:,None]*boundary1.normal_x*boundary1.weights
Nxy = boundary1.normal_x[:,None]*boundary1.normal_y*boundary1.weights
Nyx = boundary1.normal_y[:,None]*boundary1.normal_x*boundary1.weights
Nyy = boundary1.normal_y[:,None]*boundary1.normal_y*boundary1.weights
NN11 = np.bmat([[Nxx, Nxy], [Nyx, Nyy]])
SS11 += NN11
A = np.array(np.bmat([[SS11, SS21], [SS12, SS22]]))
AI = np.linalg.inv(A)
tau = AI.dot(bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([0,1], NG, [0,1], NG, mask=phys, x_endpoints=[False,False], y_endpoints=[False,False])

# evaluate at the target points
tau1 = tau[:2*boundary1.N].copy()
tau2 = tau[2*boundary1.N:].copy()
tau1x = tau1[:boundary1.N]
tau1y = tau1[boundary1.N:]
tau2x = tau2[:boundary2.N]
tau2y = tau2[boundary2.N:]
taux = np.concatenate([tau1x, tau2x])
tauy = np.concatenate([tau1y, tau2y])
tau = np.concatenate([taux, tauy])
forcx = np.concatenate([tau1x*0.0, tau2x])
forcy = np.concatenate([tau1y*0.0, tau2y])
forc = np.concatenate([forcx, forcy])
Up = Stokes_Layer_Apply(boundary, gridp, dipstr=tau, forces=forc)
up = Up[:gridp.N]
vp = Up[gridp.N:]
err_plot(up, solution_func_u)
err_plot(vp, solution_func_v)

u = np.zeros_like(x)
u[phys] = up
ua = np.zeros_like(x)
ua[phys] = solution_func_u(gridp.x, gridp.y)

################################################################################
# use the oversampling features

ftau = upsample_vector(tau)
fforc = upsample_vector(forc)
Up = Stokes_Layer_Apply(fbdy, gridp, dipstr=ftau, forces=fforc)
up = Up[:gridp.N]
vp = Up[gridp.N:]
err_plot(up, solution_func_u)
err_plot(vp, solution_func_v)

u = np.zeros_like(x)
u[phys] = up
u *= good_evals
v = np.zeros_like(x)
v[phys] = vp
v *= good_evals

