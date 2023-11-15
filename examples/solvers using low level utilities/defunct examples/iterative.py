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
NB1 = 500
NB2 = 600

# extract some functions for easy calling
squish = pybie2d.misc.curve_descriptions.squished_circle
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
PointSet = pybie2d.point_set.PointSet
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Laplace_Layer_Singular_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Apply
Cauchy_Layer_Apply = pybie2d.kernels.high_level.cauchy.Cauchy_Layer_Apply
Find_Near_Points = pybie2d.misc.near_points.find_near_points
Pairing = pybie2d.pairing.Pairing
Close_Corrector = pybie2d.close.Close_Corrector

boundary1 = GSB(c=squish(NB1,r=2,b=0.3,rot=np.pi/4.0))
boundary2 = GSB(c=star(NB2,x=0.75,y=0.75,r=0.3,a=0.4,f=7,rot=np.pi/3.0))
# need a container for this, but hack for now
boundary = PointSet(c=np.concatenate([boundary1.c, boundary2.c]))
boundary.stacked_normal_T = np.column_stack([boundary1.get_stacked_normal(), boundary2.get_stacked_normal()])
boundary.weights = np.concatenate([boundary1.weights, boundary2.weights])
def temp(s,T=True):
	return s.stacked_normal_T
boundary.get_stacked_normal = temp

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

def full_apply(tau):
	# first apply naive quad
	u = Laplace_Layer_Apply(boundary, charge=tau, dipstr=tau)
	# now sweep through and correct the DLP quads
	u[:NB1] -= 0.25*boundary1.curvature*boundary1.weights/np.pi*tau[:NB1]
	u[NB1:] -= 0.25*boundary2.curvature*boundary2.weights/np.pi*tau[NB1:]
	# now sweep through and correct the SLP quads
	su1 = Laplace_Layer_Singular_Apply(boundary1, charge=tau[:NB1])
	nu1 = Laplace_Layer_Apply(boundary1, charge=tau[:NB1])
	u[:NB1] += su1 - nu1
	su2 = Laplace_Layer_Singular_Apply(boundary2, charge=tau[NB1:])
	nu2 = Laplace_Layer_Apply(boundary2, charge=tau[NB1:])
	u[NB1:] += su2 - nu2
	# now add the 0.5I term
	u[:NB1] -= 0.5*tau[:NB1]
	u[NB1:] += 0.5*tau[NB1:]
	return u

Full_Apply_Operator = sp.sparse.linalg.LinearOperator(
							shape  = [boundary.N, boundary.N],
							matvec = full_apply)
class Gmres_Counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
counter = Gmres_Counter(True)
out = sp.sparse.linalg.gmres(Full_Apply_Operator, bc, callback=counter, restart=100, tol=1e-12)
tau = out[0]

################################################################################
# evaluate solution (no close corrections)

gridp = Grid([-2,2], N, [-2,2], N, mask=phys, periodic=True)

u = np.zeros_like(gridp.xg)
up = Laplace_Layer_Apply(boundary, gridp, charge=tau, dipstr=tau)
u[phys] = up

err_plot(up)

################################################################################
# make on-the-fly close corrections

