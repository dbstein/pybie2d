import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
plt.ion()
import pybie2d

"""o solve an interior Modified Helmholtz problem
On a complicated domain using a global quadr
Demonstrate how to use the pybie2d package tature

This example demonstrates how to do this entirely using low-level routines,
To demonstrate both how to use these low level routines
And to give you an idea what is going on under the hood in the
	higher level routines
"""

NG = 1000
h_max = 0.01

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
PPB = pybie2d.boundaries.panel_polygon_boundary.panel_polygon_boundary.Panel_Polygon_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply

################################################################################
# define problem

# boundary
boundary = PPB([0,1,1,0], [0,0,1,1], [h_max]*4, [True]*4, dyadic_levels=20, dyadic_base=3)
# solution
solution_func = lambda x, y: 2*x + y
bc = solution_func(boundary.x, boundary.y)
bcx = lambda x, y: 2.0*np.ones_like(x)
bcy = lambda x, y: 1.0*np.ones_like(x)
bcn = lambda x, y, nx, ny: bcx(x, y)*nx + bcy(x, y)*ny

def err_plot(u):
	# compute the error
	error = u - solution_func(gridp.xg, gridp.yg)
	digits = -np.log10(np.abs(error)+1e-16)
	mdigits = np.ma.array(digits)

	# plot the error as a function of space (only good in interior)
	fig, ax = plt.subplots(1,1)
	clf = ax.imshow(mdigits[:,::-1].T, extent=[0,1,0,1],
												cmap=mpl.cm.viridis_r)
	ax.set_aspect('equal')
	fig.colorbar(clf)

	print('Error: {:0.2e}'.format(np.abs(error).max()))

################################################################################
##### solve problem the hard way ###############################################
################################################################################

################################################################################
# find physical region
# (this implements a fast way to tell if points are in or out of the boundary)
# (and of course, for the squish boundary, we could easily figure out something
#      faster, but this illustrates a general purpose routine)

gridp = Grid([0,1], NG, [0,1], NG, x_endpoints=[False,False], y_endpoints=[False,False])

################################################################################
# solve for the density

DLP = Laplace_Layer_Singular_Form(boundary, ifdipole=True)
SLPp = (DLP/boundary.weights).T*boundary.weights
A = 0.5*np.eye(boundary.N) + SLPp
tau = np.linalg.solve(A, bcn(boundary.x, boundary.y, boundary.normal_x, boundary.normal_y))

# fix the mean
target = PointSet(x=np.array((0.5)),y=np.array((0.5)))
good_eval = Laplace_Layer_Apply(boundary, target=target, charge=tau)
correction = (2*0.5 + 0.5) - good_eval

################################################################################
# naive evaluation

u = Laplace_Layer_Apply(boundary, gridp, charge=tau)
u = gridp.reshape(u)
u += correction
err_plot(u)

################################################################################
# oversampled

hmax = gridp.xg[1,0] - gridp.xg[0,0]
fbdy, IMAT = boundary.prepare_oversampling(hmax/6.0)
IMAT = sp.sparse.csr_matrix(IMAT)
ftau = IMAT.dot(tau)
u = Laplace_Layer_Apply(fbdy, gridp, charge=ftau)
u = gridp.reshape(u)

u += correction
err_plot(u)

ua = 2*gridp.xg + gridp.yg

