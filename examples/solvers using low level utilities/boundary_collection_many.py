import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import numexpr as ne
import scipy as sp
import scipy.sparse
import time
plt.ion()
import pybie2d

"""
Demonstrate how to use the pybie2d package to solve an interior/exterior
Laplace problem on a complicated domain using a global quadrature
"""

N = 1000
NBS = [600]*22
NBS[1] = 500
NBS[2] = 500
NBS[-1] = 3000

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Laplace_Layer_Singular_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Apply
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Cauchy_Layer_Apply = pybie2d.kernels.high_level.cauchy.Cauchy_Layer_Apply
Find_Near_Points = pybie2d.misc.near_points.find_near_points
Pairing = pybie2d.pairing.Pairing
CollectionPairing = pybie2d.pairing.CollectionPairing
Close_Corrector = pybie2d.close.Close_Corrector
Compensated_Laplace_Apply = pybie2d.boundaries.global_smooth_boundary.Compensated_Laplace_Apply
Boundary_Collection = pybie2d.boundaries.collection.BoundaryCollection
LaplaceDirichletSolver = pybie2d.solvers.laplace_dirichlet.LaplaceDirichletSolver
Evaluate_Tau = pybie2d.solvers.laplace_dirichlet.Evaluate_Tau

boundaries = []
shapes = np.concatenate([ [squish,]*1, [star,]*17, [squish,]*3, [star,]*1 ])
xs = np.array([ 0.0,0.3142,1.18,0.5,0.1,-0.12,-0.34,-0.56,0.1,-0.14,-0.38,-0.62,-0.86,0.1,-0.25,-0.6,-0.95,-1.3,0.9,0.6,0.3,-0.8 ]) + np.e/1e5
ys = np.array([ 0.0,0.75,0.8,0.1,0.1,0.1,0.1,0.1,-0.131,-0.131,-0.131,-0.131,-0.131,-0.51,-0.51,-0.51,-0.51,-0.51,1.5,1.25,1.3,-1.2 ]) + np.e/1e5
rs = np.concatenate([ [2,0.3,0.475,0.2,], [0.1,]*9, [0.15,]*5, [0.29,0.2,0.15,0.37] ])
ab_s = np.concatenate([ [0.3,0.4,0.2,], [0.5,]*10, [0.2,]*5, [0.1,]*2, [0.5,0.5] ])
fs = np.concatenate([ [None,7,11,], [3,]*5, [5,]*5, [7,]*5, [None,]*3, [13,] ])
rots = np.pi*np.concatenate([ [0.25,1.0/3,1/1.9,], [1.0,]*15, [0.0,3.5/5,0.5,1.0,] ])

# sel = [0,1,2,3,4,5,]#6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,]
sel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,]
shapes = shapes[sel]
xs = xs[sel]
ys = ys[sel]
rs = rs[sel]
ab_s = ab_s[sel]
fs = fs[sel]
rots = rots[sel]
NBS = np.array(NBS)[sel]

for i in range(len(sel)):
	if shapes[i] == squish:
		boundaries.append(GSB(c=squish(NBS[i],x=xs[i],y=ys[i],r=rs[i],b=ab_s[i],rot=rots[i])))
	else:
		boundaries.append(GSB(c=star(NBS[i],x=xs[i],y=ys[i],r=rs[i],a=ab_s[i],f=fs[i],rot=rots[i])))

sides = np.concatenate([['i'],['e']*(len(boundaries)-1)])

boundary = Boundary_Collection()
boundary.add(boundaries, sides)
boundary.amass_information()

def solution_func(x, y):
	u = 2*x + y
	for i in range(boundary.n_boundaries):
		if i != 0:
			d2 = (x-xs[i])**2 + (y-ys[i])**2
			u += np.log(np.sqrt(d2))
	return u

bc = np.array([])
for i in range(boundary.n_boundaries):
	bc = np.concatenate([bc, solution_func(boundary.boundaries[i].x, boundary.boundaries[i].y)])

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
# iteratively solve for the density

print('\n\n----- Iterative Solver -----\n\n')
print('Generating Solver')
solver = LaplaceDirichletSolver(boundary, solve_type='iterative', check_close=True)

print('Running iterative solver')

tau = solver.solve(bc, disp=True, restart=100, tol=1e-14)

################################################################################
# find physical region

full_grid = Grid([-2,2], N, [-2,2], N, periodic=True)
phys, ext = boundary.compute_physical_region(target=full_grid)

################################################################################
# evaluate solution (no close corrections)

print('Evaluating solution on grid')
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

up = Evaluate_Tau(boundary, gridp, tau)
err_plot(up)

################################################################################
# make on-the-fly close corrections (the easy way)

print('Setting up close corrections')
pair = CollectionPairing(boundary, gridp, 1e-12)
code = pair.Setup_Close_Corrector(
	e_args={'do_DLP' : True, 'do_SLP' : True, },
	i_args={'do_DLP' : True, }
)
print('Evaluating close corrections')
st = time.time()
pair.Close_Correction(up, tau, code)
et = time.time()
err_plot(up)
