try:
	from pyfmmlib2d import FMM
	have_fmm = True
except:
	have_fmm = False

from . import backend_defaults
from . import misc
from . import kernels
from . import boundaries
from . import point_set
from . import close
from . import pairing
from . import grid
from . import solvers

