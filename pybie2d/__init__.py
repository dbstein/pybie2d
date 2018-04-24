try:
	from pyfmmlib2d import FMM
	have_fmm = True
except:
	have_fmm = False
try:
	import numexpr
	have_numexpr = True
except:
	have_numexpr = False
try:
	import numba
	have_numba = True
except:
	have_numba = False
# from . import curve_descriptions
# from ._laplace_kernels import Laplace_DLP
# from ._laplace_kernels import Laplace_SLP
# from ._laplace_kernels import Laplace_Far_Apply
# from ._stokes_kernels import Stokes_DLP
# from ._stokes_kernels import Stokes_SLP
# from ._stokes_kernels import Stokes_Far_Apply
# from ._target import Target
# from ._boundary import Boundary
# from ._global_smooth_boundary import Global_Smooth_Boundary
# from ._panel_polygon_boundary import Panel_Polygon_Boundary
# from ._boundary_collection import Boundary_Collection
# from ._equation_solver import Equation_Solver
# from ._laplace_solver import Direct_Laplace_Solver
# from ._laplace_solver import Iterative_Laplace_Solver
# from ._stokes_solver import Direct_Stokes_Solver
