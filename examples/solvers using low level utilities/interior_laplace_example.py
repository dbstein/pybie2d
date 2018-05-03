import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import pybie2d

"""
Demonstrate how to use the pybie2d package to solve an interior Laplace problem
On a complicated domain using a global quadrature

This example demonstrates how to do this entirely using low-level routines
"""

N = 500

squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.Global_Smooth_Boundary
target = pybie2d.targets.target.Target
laplace_layer = pybie2d.kernels.laplace.Laplace_Layer_Form
laplace_kernel = pybie2d.kernels.laplace.Laplace_Layer_Apply

boundary = GSB(c=squish(N,r=2,b=0.3,rot=np.pi/4.0))

# solution
solution_func = lambda x, y: 2*x + y
# grid on which to test solutions
v = np.linspace(-2, 2, N, endpoint=True)
x, y = np.meshgrid(v, v, indexing='ij')
solution = solution_func(x, y)
# get boundary condition
bc = solution_func(boundary.x, boundary.y)

# generate a target
gridp = target(x, y)

# solve for tau
DLP = laplace_layer(boundary, boundary, ifdipole=True)
A = -0.5*np.eye(boundary.N) + DLP
tau = np.linalg.solve(A, bc)

# evaluate at the target points
u = laplace_kernel(boundary, gridp, dipstr=tau)
u = gridp.reshape(u)

# compute the error
error = u - solution_func(x, y)
digits = np.log10(np.abs(error)+1e-16)

# plot the error as a function of space (only good in interior)
fig, ax = plt.subplots(1,1)
clf = ax.pcolormesh(x, y, digits)
bx = np.pad(boundary.x, (0,1), mode='wrap')
by = np.pad(boundary.y, (0,1), mode='wrap')
ax.plot(bx, by, color='white')
ax.set_aspect('equal')
fig.colorbar(clf)
