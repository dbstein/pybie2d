import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import numexpr as ne
import scipy as sp
import scipy.sparse
plt.ion()
import pybie2d

"""
Demonstrate how to use the pybie2d package to solve an interior/exterior
Laplace problem on a complicated domain using a global quadrature
And boundary collections
"""

N = 1000
NB1 = 500
NB2 = 600

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Laplace_Layer_Singular_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Apply
Cauchy_Layer_Apply = pybie2d.kernels.high_level.cauchy.Cauchy_Layer_Apply
Find_Near_Points = pybie2d.misc.near_points.find_near_points
Pairing = pybie2d.pairing.Pairing
Close_Corrector = pybie2d.close.Close_Corrector
Compensated_Laplace_Apply = pybie2d.boundaries.global_smooth_boundary.Compensated_Laplace_Apply
Boundary_Collection = pybie2d.boundaries.collection.BoundaryCollection
Evaluate_Tau = pybie2d.solvers.laplace_dirichlet.Evaluate_Tau
LaplaceDirichletSolver = pybie2d.solvers.laplace_dirichlet.LaplaceDirichletSolver

boundary1 = GSB(c=squish(NB1,r=2,b=0.3,rot=np.pi/4.0))
boundary2 = GSB(c=star(NB2,x=0.75,y=0.75,r=0.3,a=0.4,f=7,rot=np.pi/3.0))

boundary = Boundary_Collection()
boundary.add([boundary1, boundary2], ['i', 'e'])
boundary.amass_information()

def solution_func(x, y):
	d2 = (x-0.75)**2 + (y-0.75)**2
	return ne.evaluate('log(sqrt(d2)) + 2*x + y')

bc1 = solution_func(boundary1.x, boundary1.y)
bc2 = solution_func(boundary2.x, boundary2.y)
bc = np.concatenate([bc1, bc2])

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
# find physical region

full_grid = Grid([-2,2], N, [-2,2], N, periodic=True)
# this is hiding a lot of stuff!
phys1, ext1 = boundary1.find_interior_points(full_grid)
phys2, ext2 = boundary2.find_interior_points(full_grid)
phys = full_grid.reshape(np.logical_and(phys1, ext2))
ext = np.logical_not(phys)

################################################################################
# iteratively solve for the density

solver = LaplaceDirichletSolver(boundary, solve_type='iterative', check_close=False)
tau = solver.solve(bc, disp=True, restart=100, tol=1e-14)

################################################################################
# evaluate solution (no close corrections)

gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

u = np.zeros_like(gridp.xg)
up = Evaluate_Tau(boundary, gridp, tau)
u[phys] = up

err_plot(up)

################################################################################
# make on-the-fly close corrections

