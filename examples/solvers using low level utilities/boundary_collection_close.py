import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import numexpr as ne
import scipy as sp
import scipy.sparse
plt.ion()
import pybie2d
import time

"""
Demonstrate how to use the pybie2d package to solve an interior/exterior
Laplace problem on a complicated domain using a global quadrature
"""

N = 1000
NB1 = 300
NB2 = 600
NB3 = 800

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
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
Compensated_Laplace_Apply = pybie2d.boundaries.global_smooth_boundary.laplace_close_quad.Compensated_Laplace_Apply
Boundary_Collection = pybie2d.boundaries.collection.BoundaryCollection
LaplaceDirichletSolver = pybie2d.solvers.laplace_dirichlet.LaplaceDirichletSolver
Evaluate_Tau = pybie2d.solvers.laplace_dirichlet.Evaluate_Tau
Compensated_Laplace_Form = pybie2d.boundaries.global_smooth_boundary.laplace_close_quad.Compensated_Laplace_Form

boundary1 = GSB(c=squish(NB1,r=2,b=0.3,rot=np.pi/4.0))
star_center = (0.3142, 0.75)
star_center2 = (1.18, 0.8)
boundary2 = GSB(c=star(NB2,x=star_center[0],y=star_center[1],r=0.3,a=0.4,f=7,rot=np.pi/3.0))
boundary3 = GSB(c=star(NB3,x=star_center2[0],y=star_center2[1],r=0.475,a=0.2,f=11,rot=np.pi/1.9))

boundary = Boundary_Collection()
boundary.add([boundary1, boundary2, boundary3], ['i', 'e', 'e'])
boundary.amass_information()

def solution_func(x, y):
	d2 = (x-star_center[0])**2 + (y-star_center[1])**2
	d3 = (x-star_center2[0])**2 + (y-star_center2[1])**2
	return ne.evaluate('log(sqrt(d2)) + log(sqrt(d3)) + 2*x + y')
def solution_func_x(x, y):
	d2 =  (x-star_center[0])/((x-star_center[0])**2 + (y-star_center[1])**2)
	d3 = (x-star_center2[0])/((x-star_center2[0])**2 + (y-star_center2[1])**2)
	return ne.evaluate('d2 + d3 + 2')
def solution_func_y(x, y):
	d2 =  (y-star_center[1])/((x-star_center[0])**2 + (y-star_center[1])**2)
	d3 = (y-star_center2[1])/((x-star_center2[0])**2 + (y-star_center2[1])**2)
	return ne.evaluate('d2 + d3 + 1')

bc1 = solution_func(boundary1.x, boundary1.y)
bc2 = solution_func(boundary2.x, boundary2.y)
bc3 = solution_func(boundary3.x, boundary3.y)
bc = np.concatenate([bc1, bc2, bc3])

def err_plot(up, func=solution_func):
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

def dumb_plot(up):
	thing = np.zeros_like(full_grid.xg)
	thing[phys] = up
	fig, ax = plt.subplots(1,1)
	clf = ax.imshow(thing[:,::-1].T, extent=[-2,2,-2,2],
												cmap=mpl.cm.viridis_r)
	ax.set_aspect('equal')
	fig.colorbar(clf)

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
print('Computing interior/exteriors')
# phys1, ext1 = boundary1.find_interior_points(full_grid)
# phys2, ext2 = boundary2.find_interior_points(full_grid)
# phys3, ext3 = boundary3.find_interior_points(full_grid)
# exter = np.logical_and(ext2, ext3)
# phys = full_grid.reshape(np.logical_and(phys1, exter))
# ext = np.logical_not(phys)

# find physical region the easy way
# st = time.time()
phys, ext = boundary.compute_physical_region(target=full_grid)
# et = time.time()

# st = time.time()
# out = Cauchy_Layer_Apply(boundary, full_grid, dipstr=np.ones(boundary.N)).real
# et = time.time()
# out = out.reshape(full_grid.shape)
# out[out > 2] = 2.0
# out[out < -2] = -2.0

################################################################################
# evaluate solution (no close corrections)

print('Evaluating solution on grid')
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

up = Evaluate_Tau(boundary, gridp, tau)
err_plot(up)

################################################################################
# make on-the-fly close corrections (the hard way)

print('Applying close corrections')
pair1 = Pairing(boundary1, gridp, 'i', 1e-12)
pair2 = Pairing(boundary2, gridp, 'e', 1e-12)
pair3 = Pairing(boundary3, gridp, 'e', 1e-12)
code1 = pair1.Setup_Close_Corrector(do_DLP=True, backend='fly')
code2 = pair2.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='fly')
code3 = pair3.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='fly')

pair1.Close_Correction(up, tau[:NB1],        code1)
pair2.Close_Correction(up, tau[NB1:NB1+NB2], code2)
pair3.Close_Correction(up, tau[NB1+NB2:],    code3)
err_plot(up)

################################################################################
# fast form operator (the hard way)

print('\n\n----- Non-Iterative Solver -----\n\n')

boundary1 = GSB(c=squish(NB1,r=2,b=0.3,rot=np.pi/4.0))
star_center = (0.3142, 0.75)
star_center2 = (1.18, 0.8)
boundary2 = GSB(c=star(NB2,x=star_center[0],y=star_center[1],r=0.3,a=0.4,f=7,rot=np.pi/3.0))
boundary3 = GSB(c=star(NB3,x=star_center2[0],y=star_center2[1],r=0.475,a=0.2,f=11,rot=np.pi/1.9))
boundary = Boundary_Collection()
boundary.add([boundary1, boundary2, boundary3], ['i', 'e', 'e'])
boundary.amass_information()

print('Forming matrix')

# without close corrections
OP = np.empty([boundary.N, boundary.N], dtype=float)
OP[       :NB1    ,        :NB1    ] = Laplace_Layer_Singular_Form(boundary1, ifdipole=True)
OP[NB1    :NB1+NB2,        :NB1    ] = Laplace_Layer_Form(boundary1, boundary2, ifdipole=True)
OP[NB1+NB2:       ,        :NB1    ] = Laplace_Layer_Form(boundary1, boundary3, ifdipole=True)
OP[       :NB1    , NB1    :NB1+NB2] = Laplace_Layer_Form(boundary2, boundary1, ifdipole=True, ifcharge=True)
OP[NB1    :NB1+NB2, NB1    :NB1+NB2] = Laplace_Layer_Singular_Form(boundary2, ifdipole=True, ifcharge=True)
OP[NB1+NB2:       , NB1    :NB1+NB2] = Laplace_Layer_Form(boundary2, boundary3, ifdipole=True, ifcharge=True)
OP[       :NB1    , NB1+NB2:       ] = Laplace_Layer_Form(boundary3, boundary1, ifdipole=True, ifcharge=True)
OP[NB1    :NB1+NB2, NB1+NB2:       ] = Laplace_Layer_Form(boundary3, boundary2, ifdipole=True, ifcharge=True)
OP[NB1+NB2:       , NB1+NB2:       ] = Laplace_Layer_Singular_Form(boundary3, ifdipole=True, ifcharge=True)
# fully form close corrections
pair12 = Pairing(boundary1, boundary2, 'i', 1e-12)
code12 = pair12.Setup_Close_Corrector(do_DLP=True, backend='preformed')
pair13 = Pairing(boundary1, boundary3, 'i', 1e-12)
code13 = pair13.Setup_Close_Corrector(do_DLP=True, backend='preformed')
pair21 = Pairing(boundary2, boundary1, 'e', 1e-12)
code21 = pair21.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='preformed')
pair23 = Pairing(boundary2, boundary3, 'e', 1e-12)
code23 = pair23.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='preformed')
pair31 = Pairing(boundary3, boundary1, 'e', 1e-12)
code31 = pair31.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='preformed')
pair32 = Pairing(boundary3, boundary2, 'e', 1e-12)
code32 = pair32.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='preformed')
# insert into operator
OP[NB1    :NB1+NB2,        :NB1    ][pair12.close_points, :] += pair12.close_correctors[code12].preparations['correction_mat']
OP[NB1+NB2:       ,        :NB1    ][pair13.close_points, :] += pair13.close_correctors[code13].preparations['correction_mat']
OP[       :NB1    , NB1    :NB1+NB2][pair21.close_points, :] += pair21.close_correctors[code21].preparations['correction_mat']
OP[NB1+NB2:       , NB1    :NB1+NB2][pair23.close_points, :] += pair23.close_correctors[code23].preparations['correction_mat']
OP[       :NB1    , NB1+NB2:       ][pair31.close_points, :] += pair31.close_correctors[code31].preparations['correction_mat']
OP[NB1    :NB1+NB2, NB1+NB2:       ][pair32.close_points, :] += pair32.close_correctors[code32].preparations['correction_mat']
# add in 0.5I terms
OP[       :NB1    ,        :NB1    ] -= 0.5*np.eye(boundary1.N)
OP[NB1    :NB1+NB2, NB1    :NB1+NB2] += 0.5*np.eye(boundary2.N)
OP[NB1+NB2:       , NB1+NB2:       ] += 0.5*np.eye(boundary3.N)

print('Solving for density')
tau2 = np.linalg.solve(OP, bc)

################################################################################
# fast form operator (easy way)

solver = LaplaceDirichletSolver(boundary, solve_type='formed', check_close=True)
tau3 = solver.solve(bc)

################################################################################
# evaluate solution (no close corrections)

print('Evaluating solution on grid')
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

up = Evaluate_Tau(boundary, gridp, tau3)
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
pair.Close_Correction(up, tau3, code)
err_plot(up)

################################################################################
# eigendecomposition of matrix

print('Computing eigendecomposition of operator')
E = np.linalg.eig(OP)
fig, ax = plt.subplots(1,1)
ax.scatter(E[0].real, E[0].imag)

################################################################################
# compute derivatives

uu, uux, uuy = Laplace_Layer_Apply(boundary, gridp, charge=tau*boundary.SLP_vector, dipstr=tau, gradient=True)

# correct u up close
pair1 = Pairing(boundary1, gridp, 'i', 1e-15)
code1 = pair1.Setup_Close_Corrector(do_DLP=True, backend='fly')
tauh = tau[:boundary1.N]
pair1.Close_Correction(uu, tauh, code1)
pair2 = Pairing(boundary2, gridp, 'e', 1e-15)
code2 = pair2.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='fly')
tauh = tau[boundary1.N:boundary1.N+boundary2.N]
pair2.Close_Correction(uu, tauh, code2)
pair3 = Pairing(boundary3, gridp, 'e', 1e-15)
code3 = pair3.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='fly')
tauh = tau[boundary1.N+boundary2.N:]
pair3.Close_Correction(uu, tauh, code3)

# correct ux up close
close_mat1a, dxm1a, dym1a = Compensated_Laplace_Form(boundary1, pair1.close_targ, 'i', do_DLP=True, gradient=True)
close_mat1b, dxm1b, dym1b = Laplace_Layer_Form(boundary1, pair1.close_targ, ifdipole=True, gradient=True)
close_mat2a, dxm2a, dym2a = Compensated_Laplace_Form(boundary2, pair2.close_targ, 'e', do_DLP=True, do_SLP=True, gradient=True)
close_mat2b, dxm2b, dym2b = Laplace_Layer_Form(boundary2, pair2.close_targ, ifdipole=True, ifcharge=True, gradient=True)
close_mat3a, dxm3a, dym3a = Compensated_Laplace_Form(boundary3, pair3.close_targ, 'e', do_DLP=True, do_SLP=True, gradient=True)
close_mat3b, dxm3b, dym3b = Laplace_Layer_Form(boundary3, pair3.close_targ, ifdipole=True, ifcharge=True, gradient=True)

tauh = tau[:boundary1.N]
uux[pair1.close_points] += dxm1a.dot(tauh) - dxm1b.dot(tauh)
uuy[pair1.close_points] += dym1a.dot(tauh) - dym1b.dot(tauh)
tauh = tau[boundary1.N:boundary1.N+boundary2.N]
uux[pair2.close_points] += dxm2a.dot(tauh) - dxm2b.dot(tauh)
uuy[pair2.close_points] += dym2a.dot(tauh) - dym2b.dot(tauh)
tauh = tau[boundary1.N+boundary2.N:]
uux[pair3.close_points] += dxm3a.dot(tauh) - dxm3b.dot(tauh)
uuy[pair3.close_points] += dym3a.dot(tauh) - dym3b.dot(tauh)

err_plot(uu)
err_plot(uux, solution_func_x)
err_plot(uuy, solution_func_y)

