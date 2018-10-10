import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import time
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

NG = 10
h_max = 1.0/NG

# extract some functions for easy calling
PPB = pybie2d.boundaries.panel_polygon_boundary.panel_polygon_boundary.Panel_Polygon_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Cauchy_Layer_Apply = pybie2d.kernels.high_level.cauchy.Cauchy_Layer_Apply
Find_Near_Points = pybie2d.misc.near_points.find_near_points
Pairing = pybie2d.pairing.Pairing

################################################################################
# define problem

# boundary
boundary = PPB([0,1,1,0], [0,0,1,1], [h_max]*4, [True]*4)
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
# this is hiding a lot of stuff!
phys, ext = boundary.find_interior_points(full_grid)
phys = full_grid.reshape(phys)
ext = full_grid.reshape(ext)

################################################################################
# solve for the density

DLP = Laplace_Layer_Singular_Form(boundary, ifdipole=True)
A = -0.5*np.eye(boundary.N) + DLP
AI = np.linalg.inv(A)
tau = AI.dot(bc)

################################################################################
# naive evaluation

# generate a target for the physical grid
gridp = Grid([0,1], NG, [0,1], NG, mask=phys, x_endpoints=[False,False], y_endpoints=[False,False])

# evaluate at the target points
u = np.zeros_like(gridp.xg)
up = Laplace_Layer_Apply(boundary, gridp, dipstr=tau)
u[phys] = up

err_plot(up)

################################################################################
# correct with close evaluation (preformed)

uph = up.copy()
# generate close eval matrix
boundary.Laplace_Correct_Close_Setup(gridp)
boundary.Laplace_Correct_Close(tau, uph)
err_plot(uph)

################################################################################
# now redo (in simulation style) to demonstrate speed

st = time.time()
tau = AI.dot(bc)
uph = Laplace_Layer_Apply(boundary, gridp, dipstr=tau)
boundary.Laplace_Correct_Close(tau, uph)
err_plot(uph)
et = time.time()
print('Solution took: ', 1000*(et-st), 'ms')


