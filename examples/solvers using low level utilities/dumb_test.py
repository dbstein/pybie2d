import numpy as np
import scipy as sp
import scipy.sparse
import scipy.special
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path
import time
plt.ion()
import pybie2d

NB = 5000
helmholtz_k = 100.0

squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet
Grid = pybie2d.grid.Grid
Modified_Helmholtz_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply
Modified_Helmholtz_Kernel_Apply = pybie2d.kernels.low_level.modified_helmholtz.Modified_Helmholtz_Kernel_Apply
Modified_Helmholtz_Kernel_Form = pybie2d.kernels.low_level.modified_helmholtz.Modified_Helmholtz_Kernel_Form
k0 = scipy.special.k0
k1 = scipy.special.k1

boundary = GSB(c=squish(2*int(NB/2),r=10.0,b=1.0))
bx1 = boundary.x + boundary.max_h*boundary.normal_x
by1 = boundary.y + boundary.max_h*boundary.normal_y
targ = PointSet(bx1, by1)
tau = np.ones(boundary.N)
st = time.time()
up = Modified_Helmholtz_Layer_Apply(boundary, targ, k=helmholtz_k, charge=tau, backend='FMM')
et = time.time()
print('...Maximum value, FMM:    {:0.12e}'.format(up.max()))
print('...Time:                  {:0.1f}'.format((et-st)*1000))
st = time.time()
up = Modified_Helmholtz_Layer_Apply(boundary, targ, k=helmholtz_k, charge=tau, backend='numba')
et = time.time()
print('...Maximum value, Numba: {:0.12e}'.format(up.max()))
print('...Time:                  {:0.1f}'.format((et-st)*1000))
true = up.copy()

print('\nTest evaluation using trees')
boundary = GSB(c=squish(2*int(NB/2),r=10.0,b=1.0))
bx1 = boundary.x + boundary.max_h*boundary.normal_x
by1 = boundary.y + boundary.max_h*boundary.normal_y
targ = PointSet(bx1, by1)
# find when k0 < 1e-12
func = lambda x: k0(helmholtz_k*x) - 1e-12
dfunc = lambda x: -helmholtz_k*k1(helmholtz_k*x)
dist = sp.optimize.newton(func, 1e-5, fprime=dfunc, tol=1e-12, maxiter=500)
close_enough = targ.find_near_points(boundary, dist)

st = time.time()

# slice domain into regions of size dist
boundary_x_min = boundary.x.min()
boundary_y_min = boundary.y.min()
target_x_min = targ.x.min()
target_y_min = targ.y.min()
boundary_x_max = boundary.x.max()
boundary_y_max = boundary.y.max()
target_x_max = targ.x.max()
target_y_max = targ.y.max()
x_min = np.min([boundary_x_min, target_x_min])
y_min = np.min([boundary_y_min, target_y_min])
x_max = np.max([boundary_x_max, target_x_max])
y_max = np.max([boundary_y_max, target_y_max])
x_ran = x_max-x_min
y_ran = y_max-y_min
Nx = int(np.ceil(x_ran/dist))
Ny = int(np.ceil(y_ran/dist))
x_cuts = np.linspace(x_min, x_max, Nx+1, endpoint=True)
y_cuts = np.linspace(y_min, y_max, Ny+1, endpoint=True)
# loop through this matrix and apply
def compare(x, cl, cu, right_end):
	if right_end:
		return np.logical_and(x >= cl, x <= cu)
	else:
		return np.logical_and(x >= cl, x < cu)
all_source_points = boundary.get_stacked_boundary()
all_target_points = targ.get_stacked_boundary()
tau = np.ones(boundary.N)
all_weighted_charge = tau*boundary.weights
# first get source boxes
source_inx_inds = np.empty(Nx, dtype=object)
for i in range(Nx):
	source_inx_inds[i] = np.where(compare(boundary.x, x_cuts[i], x_cuts[i+1], i==Nx-1))[0]
source_iny_inds = np.empty(Ny, dtype=object)
for j in range(Ny):
	source_iny_inds[j] = np.where(compare(boundary.y, y_cuts[j], y_cuts[j+1], j==Ny-1))[0]
source_in_inds = np.empty([Nx, Ny], dtype=object)
box_has_source = np.empty([Nx, Ny], dtype=bool)
for i in range(Nx):
	for j in range(Ny):
		source_in_inds[i,j] = np.intersect1d(source_inx_inds[i], source_iny_inds[j], assume_unique=True)
		box_has_source[i,j] = source_in_inds[i,j].shape[0] > 0
# now target boxes
target_inx_inds = np.empty(Nx, dtype=object)
for i in range(Nx):
	target_inx_inds[i] = np.where(compare(targ.x, x_cuts[i], x_cuts[i+1], i==Nx-1))[0]
target_iny_inds = np.empty(Ny, dtype=object)
for j in range(Ny):
	target_iny_inds[j] = np.where(compare(targ.y, y_cuts[j], y_cuts[j+1], j==Ny-1))[0]
target_in_inds = np.empty([Nx, Ny], dtype=object)
box_needed = np.empty([Nx, Ny], dtype=bool)
for i in range(Nx):
	for j in range(Ny):
		box_needed_here = False
		for k1 in (-1,0,1):
			for k2 in (-1,0,1):
				try:
					box_needed_here = box_needed_here or box_has_source[i+k1,j+k2]
				except:
					pass
		box_needed[i,j] = box_needed_here
		if box_needed_here:
			target_in_inds[i,j] = np.intersect1d(target_inx_inds[i], target_iny_inds[j], assume_unique=True)
		else:
			target_in_inds[i,j] = np.array([])
# loop through the sources and add up
matrix = np.zeros([targ.N, boundary.N])
out = np.zeros(targ.N)
# guess how much memory we'll need
currentmax = 1000
xinds = np.zeros(currentmax, dtype=int)
yinds = np.zeros(currentmax, dtype=int)
data = np.zeros(currentmax)
totaln = 0
for i in range(Nx):
	for j in range(Ny):
		source_inds = source_in_inds[i,j]
		if source_inds.shape[0] > 0:
			source_points = all_source_points[:,source_inds]
			weighted_charge = all_weighted_charge[source_inds]
			for k1 in (-1,0,1):
				for k2 in (-1,0,1):
					ii = i + k1
					jj = j + k2
					if ii >= 0 and ii < Nx and jj >= 0 and jj < Ny:
						target_inds = target_in_inds[i+k1,j+k2]
						if target_inds.shape[0] > 0:
							target_points = all_target_points[:,target_inds]
							out[target_inds] += Modified_Helmholtz_Kernel_Apply(source_points, target_points, helmholtz_k, charge=weighted_charge, backend='numba')
							MAT = Modified_Helmholtz_Kernel_Form(source_points, target_points, helmholtz_k, ifcharge=True)
							heren = np.prod(MAT.shape)
							totaln += heren
							while totaln > currentmax:
								oldxinds = xinds.copy()
								oldyinds = yinds.copy()
								olddata = data.copy()
								oldcurrentmax = currentmax
								currentmax *= 2
								xinds = np.zeros(currentmax, dtype=int)
								yinds = np.zeros(currentmax, dtype=int)
								data = np.zeros(currentmax)
								xinds[:oldcurrentmax] = oldxinds
								yinds[:oldcurrentmax] = oldyinds
								data[:oldcurrentmax] = olddata
							indsx, indsy = np.meshgrid(target_inds, source_inds, indexing='ij')
							xinds[totaln-heren:totaln] = indsx.flatten()
							yinds[totaln-heren:totaln] = indsy.flatten()
							data[totaln-heren:totaln] = MAT.flatten()
et = time.time()

xinds = xinds[:totaln]
yinds = yinds[:totaln]
data = data[:totaln]
matrix = sp.sparse.coo_matrix((data, (xinds, yinds)))
matrix = matrix.tocsr()

up = out
err = np.abs(up - true).max()
print('...Maximum value, treecode:    {:0.12e}'.format(up.max()))
print('...Error, treecode:            {:0.12e}'.format(err))
print('...Time:                       {:0.1f}'.format((et-st)*1000))
st = time.time()
up = matrix.dot(tau*boundary.weights)
err = np.abs(up - true).max()
et = time.time()
print('...Maximum value, treemat:     {:0.12e}'.format(up.max()))
print('...Error, treecode:            {:0.12e}'.format(err))
print('...Time:                       {:0.1f}'.format((et-st)*1000))


