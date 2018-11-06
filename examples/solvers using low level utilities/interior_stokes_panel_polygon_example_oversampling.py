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

NG = 100
h_max = 0.05

# extract some functions for easy calling
PPB = pybie2d.boundaries.panel_polygon_boundary.panel_polygon_boundary.Panel_Polygon_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply

################################################################################
# define problem

# boundary
boundary = PPB([0,1,1,0], [0,0,1,1], [h_max]*4, [True]*4, dyadic_levels=24, dyadic_base=3)
# solution
solution_func_u = lambda x, y: 2*y + x
solution_func_v = lambda x, y: 0.5*x - y
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

full_grid = Grid([0,1], NG, [0,1], NG, x_endpoints=[False,False], y_endpoints=[False,False])
# this is hiding a lot of stuff!
phys, ext = boundary.find_interior_points(full_grid)
phys = full_grid.reshape(phys)
ext = full_grid.reshape(ext)

################################################################################
# solve for the density

DLP = Stokes_Layer_Singular_Form(boundary, ifdipole=True)
A = -0.5*np.eye(2*boundary.N) + DLP
A[:,0] += np.concatenate([boundary.normal_x, boundary.normal_y])
AI = np.linalg.inv(A)
tau = AI.dot(bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([0,1], NG, [0,1], NG, mask=phys, x_endpoints=[False,False], y_endpoints=[False,False])

# evaluate at the target points
Up = Stokes_Layer_Apply(boundary, gridp, dipstr=tau)
up = Up[:gridp.N]
vp = Up[gridp.N:]
err_plot(up, solution_func_u)
err_plot(vp, solution_func_v)

################################################################################
# use the oversampling features

hmax = gridp.xg[1,0] - gridp.xg[0,0]
fbdy, IMAT = boundary.prepare_oversampling(hmax/6.0)
IMAT = sp.sparse.csr_matrix(IMAT)
taux = tau[:boundary.N]
tauy = tau[boundary.N:]
ftaux = IMAT.dot(taux)
ftauy = IMAT.dot(tauy)
ftau = np.concatenate([ftaux, ftauy])
Up = Stokes_Layer_Apply(fbdy, gridp, dipstr=ftau)
up = Up[:gridp.N]
vp = Up[gridp.N:]
err_plot(up, solution_func_u)
err_plot(vp, solution_func_v)

