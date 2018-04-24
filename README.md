# pyBIE2D: Python Tools for Boundary Integral Equations in 2D
 ---
The goal of this project is to provide tools in python for solving the Laplace and Stokes problems in two dimensions, modeled after Alex Barnett's package BIE2D for MATLAB available at: https://github.com/ahbarnett/BIE2D.

This project is in a very early state. For now, the only things implemented are:

1. The readme
2. Global Smooth Quadratures with Kress self eval and Cauchy compensated close eval (now including close-eval matrix formation)
3. Laplace SLP and DLP Kernels
4. Reasonably smart 'Boundary Collections'
5. Integrated FMM Routines for Laplace SLP, DLP
6. A few examples for solving Laplace problems 

Most of the code is quite messy and internal consistency between classes is pretty weak, and will have to be cleaned up as further development makes clearer what classes/modules should own what routines. Expect significant code reorganization in the near future.

##Installation
Install by navigating to the package route and typing:
```bash
pip install .
```
The FMM routines require pyfmmlib2d. The installation of pybie2d and the non-FMM portions of the code should work without installing that package, but FMM routinies will not run and will probably produce errors with completely useless error messages. The package requires numpy, numba, numexpr, and matplotlib, but will not check that these are installed (in order to protect conda-based installations from pip upgrades).

## A simple example

The goal of this project is to build a set of easy to use routines to define boundaries and integrate this with optimized, efficient code for solving homogeneous PDE on the domains bounded by these boundaries. The following code solves a problem inside of *boundary1* and outside of *boundary2* and *boundary3*, and provides the solution on a provided grid, using close evaluation routines where necessary to provide high accuracy. Evaluation where close eval is not needed is done using an FMM for speed.

```python
import pyBIE2D
import numpy as np

def solution_func(x, y):
	sol = 2*x + y
	r1 = np.sqrt((x-1)**2 + (y-1)**2)
	r2 = np.sqrt((x+1)**2 + (y+1)**2)
	sol += np.log(r1)
	sol += 2*np.log(r2)
	return sol

# setup grid
v = np.linspace(-4, 4, 200, endpoint=True)
x, y = np.meshgrid(v, v, indexing='ij')

# setup boundaries
squish = pyBIE2D.curve_descriptions.squished_circle
star = pyBIE2D.curve_descriptions.star
boundary1 = pyBIE2D.Global_Smooth_Boundary(c=squish(400,r=4,b=0.3,rot=np.pi/4.0))
boundary2 = pyBIE2D.Global_Smooth_Boundary(c=star(300,x=1,y=1,r=0.5,a=0.6,f=3,rot=np.pi/3.0))
boundary3 = pyBIE2D.Global_Smooth_Boundary(c=star(400,x=-1,y=-1,r=0.6,a=0.06,f=7,rot=0.0))

# construct Boundary_Collection object and add boundaries
boundary = pyBIE2D.Boundary_Collection()
boundary.add([boundary1, boundary2, boundary3],['i','e','e'])
boundary.amass_information()

# get the physical region of the grid
phys, gridp = boundary.compute_physical_region(x, y, return_target=True)

# construct Direct_Laplace_Solver object
LS = pyBIE2D.Direct_Laplace_Solver(boundary)

# set targets
LS.set_target(gridp)

# get the boundary condition
bc = solution_func(boundary.x, boundary.y)
LS.set_boundary_condition(bc)

# solve for density
LS.solve()

# evaluate density at points (FMM + close evaluations)
up = LS.FMM_evaluation()

# construct solution and plot errors
u = np.zeros_like(x)
u[phys] = up
```

This produces a domain and solution to a known problem with correct digits as shown below:

![Solution](digits.png?raw=true "Title")
