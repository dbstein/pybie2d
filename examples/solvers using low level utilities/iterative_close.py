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
"""

N = 1000
NB1 = 200
NB2 = 500
NB3 = 600

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
Close_Corrector = pybie2d.close.Close_Corrector
Compensated_Laplace_Apply = pybie2d.boundaries.global_smooth_boundary.Compensated_Laplace_Apply

boundary1 = GSB(c=squish(NB1,r=2,b=0.3,rot=np.pi/4.0), 
											compute_differentiation_matrix=False)
star_center = (0.3142, 0.75)
star_center2 = (1.18, 0.8)
boundary2 = GSB(c=star(NB2,x=star_center[0],y=star_center[1],r=0.3,a=0.4,f=7,
							rot=np.pi/3.0),compute_differentiation_matrix=False)
boundary3 = GSB(c=star(NB3,x=star_center2[0],y=star_center2[1],r=0.475,a=0.2,f=11,
							rot=np.pi/1.9),compute_differentiation_matrix=False)
# need a container for this, but hack for now
boundary = PointSet(c=np.concatenate([boundary1.c, boundary2.c, boundary3.c]),
														compute_tree=False)
boundary.stacked_normal_T = np.column_stack([boundary1.stacked_normal_T, boundary2.stacked_normal_T, boundary3.stacked_normal_T])
boundary.weights = np.concatenate([boundary1.weights, boundary2.weights, boundary3.weights])

def solution_func(x, y):
	d2 = (x-star_center[0])**2 + (y-star_center[1])**2
	d3 = (x-star_center2[0])**2 + (y-star_center2[1])**2
	return ne.evaluate('log(sqrt(d2)) + log(sqrt(d3)) + 2*x + y')

bc1 = solution_func(boundary1.x, boundary1.y)
bc2 = solution_func(boundary2.x, boundary2.y)
bc3 = solution_func(boundary3.x, boundary3.y)
bc = np.concatenate([bc1, bc2, bc3])

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
print('Generating Close Correctors')

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
slp = np.zeros_like(boundary.x)
slp[NB1:] = 1.0

print('Generating Singular Quadratures')

SU2 = Laplace_Layer_Singular_Form(boundary2, ifcharge=True)
SN2 = Laplace_Layer_Form(boundary2, ifcharge=True)
D2 = SU2 - SN2
SU3 = Laplace_Layer_Singular_Form(boundary3, ifcharge=True)
SN3 = Laplace_Layer_Form(boundary3, ifcharge=True)
D3 = SU3 - SN3

def full_apply(tau):
	# first apply naive quad
	u = Laplace_Layer_Apply(boundary, charge=slp*tau, dipstr=tau)
	# sweep through and correct the DLP self quads
	u[:NB1]        -= 0.25*boundary1.curvature*boundary1.weights/np.pi*tau[:NB1]
	u[NB1:NB1+NB2] -= 0.25*boundary2.curvature*boundary2.weights/np.pi*tau[NB1:NB1+NB2]
	u[NB1+NB2:]    -= 0.25*boundary3.curvature*boundary3.weights/np.pi*tau[NB1+NB2:]
	# sweep through and correct the SLP self quads
	u[NB1:NB1+NB2] += D2.dot(tau[NB1:NB1+NB2])
	u[NB1+NB2:]    += D3.dot(tau[NB1+NB2:])
	# do the close corrections
	pair12.Close_Correction(u[NB1:NB1+NB2], tau[:NB1],        code12)
	pair13.Close_Correction(u[NB1+NB2:],    tau[:NB1],        code13)
	pair21.Close_Correction(u[:NB1],        tau[NB1:NB1+NB2], code21)
	pair23.Close_Correction(u[NB1+NB2:],    tau[NB1:NB1+NB2], code23)
	pair31.Close_Correction(u[:NB1],        tau[NB1+NB2:],    code31)
	pair32.Close_Correction(u[NB1:NB1+NB2], tau[NB1+NB2:],    code32)
	# now add the 0.5I term
	u[:NB1]        -= 0.5*tau[:NB1]
	u[NB1:NB1+NB2] += 0.5*tau[NB1:NB1+NB2]
	u[NB1+NB2:]    += 0.5*tau[NB1+NB2:]
	return u

Full_Apply_Operator = sp.sparse.linalg.LinearOperator(
							shape  = [boundary.N, boundary.N],
							matvec = full_apply)

# setup a block preconditioner
PC1 = Laplace_Layer_Singular_Form(boundary1, ifdipole=True)
PC1 -= 0.5*np.eye(boundary1.N)
PC2 = Laplace_Layer_Singular_Form(boundary2, ifcharge=True, ifdipole=True)
PC2 += 0.5*np.eye(boundary2.N)
PC3 = Laplace_Layer_Singular_Form(boundary3, ifcharge=True, ifdipole=True)
PC3 += 0.5*np.eye(boundary3.N)
IPC1 = np.linalg.inv(PC1)
IPC2 = np.linalg.inv(PC2)
IPC3 = np.linalg.inv(PC3)
def prec(tau):
	return np.concatenate([
				IPC1.dot(tau[:NB1]),
				IPC2.dot(tau[NB1:NB1+NB2]),
				IPC3.dot(tau[NB1+NB2:])
			])
Prec_Apply_Operator = sp.sparse.linalg.LinearOperator(
							shape  = [boundary.N, boundary.N],
							matvec = prec)

class Gmres_Counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

print('Running iterative solver with preconditioner')
counter = Gmres_Counter(True)
out = sp.sparse.linalg.gmres(Full_Apply_Operator, bc, callback=counter, restart=100, tol=1e-14, M=Prec_Apply_Operator)
tau = out[0]
print('Running iterative solver without preconditioner')
counter = Gmres_Counter(True)
out = sp.sparse.linalg.gmres(Full_Apply_Operator, bc, callback=counter, restart=100, tol=1e-14)

################################################################################
# find physical region

full_grid = Grid([-2,2], N, [-2,2], N, periodic=True)
print('Computing interior/exteriors')
phys1, ext1 = boundary1.find_interior_points(full_grid)
phys2, ext2 = boundary2.find_interior_points(full_grid)
phys3, ext3 = boundary3.find_interior_points(full_grid)
exter = np.logical_and(ext2, ext3)
phys = full_grid.reshape(np.logical_and(phys1, exter))
ext = np.logical_not(phys)

################################################################################
# evaluate solution (no close corrections)

print('Evaluating solution on grid')
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

up = Laplace_Layer_Apply(boundary, gridp, charge=tau*slp, dipstr=tau)
err_plot(up)

################################################################################
# make on-the-fly close corrections

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
# fast form operator

print('\n\n----- Non-Iterative Solver -----\n\n')

boundary1 = GSB(c=squish(NB1,r=2,b=0.3,rot=np.pi/4.0), 
											compute_differentiation_matrix=True)
star_center = (0.3142, 0.75)
star_center2 = (1.18, 0.8)
boundary2 = GSB(c=star(NB2,x=star_center[0],y=star_center[1],r=0.3,a=0.4,f=7,
							rot=np.pi/3.0),compute_differentiation_matrix=True)
boundary3 = GSB(c=star(NB3,x=star_center2[0],y=star_center2[1],r=0.475,a=0.2,f=11,
							rot=np.pi/1.9),compute_differentiation_matrix=True)

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
code12 = pair12.Setup_Close_Corrector(do_DLP=True, backend='full preformed')
pair13 = Pairing(boundary1, boundary3, 'i', 1e-12)
code13 = pair13.Setup_Close_Corrector(do_DLP=True, backend='full preformed')
pair21 = Pairing(boundary2, boundary1, 'e', 1e-12)
code21 = pair21.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='full preformed')
pair23 = Pairing(boundary2, boundary3, 'e', 1e-12)
code23 = pair23.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='full preformed')
pair31 = Pairing(boundary3, boundary1, 'e', 1e-12)
code31 = pair31.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='full preformed')
pair32 = Pairing(boundary3, boundary2, 'e', 1e-12)
code32 = pair32.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='full preformed')
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
# evaluate solution (no close corrections)

print('Evaluating solution on grid')
gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

up = Laplace_Layer_Apply(boundary, gridp, charge=tau*slp, dipstr=tau2)
err_plot(up)

################################################################################
# make on-the-fly close corrections

print('Applying close corrections')
pair1 = Pairing(boundary1, gridp, 'i', 1e-12)
pair2 = Pairing(boundary2, gridp, 'e', 1e-12)
pair3 = Pairing(boundary3, gridp, 'e', 1e-12)
code1 = pair1.Setup_Close_Corrector(do_DLP=True, backend='fly')
code2 = pair2.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='fly')
code3 = pair3.Setup_Close_Corrector(do_DLP=True, do_SLP=True, backend='fly')

pair1.Close_Correction(up, tau2[:NB1],        code1)
pair2.Close_Correction(up, tau2[NB1:NB1+NB2], code2)
pair3.Close_Correction(up, tau2[NB1+NB2:],    code3)
err_plot(up)

################################################################################
# eigendecomposition of matrix

print('Computing eigendecomposition of operator')
E = np.linalg.eig(OP)
fig, ax = plt.subplots(1,1)
ax.scatter(E[0].real, E[0].imag)


