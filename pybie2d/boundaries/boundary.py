import numpy as np
import scipy as sp
import scipy.spatial
from ..point_set import PointSet
from ..misc.interior_points import find_interior_points

class Boundary(PointSet):
    """
    Parent class for all "Boundaries"
    This class should never be used directly
    Always instantiate Boundaries through the child classes

    Methods:
        __init__:
            pre-initialization routine
            must be provided by child class
        __init2__:
            post-initialization routine to be called at end of child init
        compute_quadrature:
            the workhorse for this class - computes quadrature nodes and weights
            must be provided by child class
        find_interior_points:
            computes which points of target are interior to target or not
    """
    def __init__(self, x=None, y=None, c=None):
        super(Boundary, self).__init__(x, y, c)
    # end __init__ function definition

    def find_interior_points(self, target):
        """
        Computes interior/exterior points via cauchy sum +
        brute force search near to boundary, using matplotlib.path
        and treating the boundary as a discrete polygon
        for the brute force search

        if the boundary type has a simple way to deal with this,
        this method should be overwritten by the child class
        """
        return find_interior_points(self, target)

    def decorate(self, decoration_name, *args, **kwargs):
        """
        Function for calling boundary decoraters
        (Not sure this is necessary?)
        """
        getattr(self, decoration_name)(args, kwargs)

    # Decorations shared across boundary classes
    def stack_normal(self):
        if not hasattr(self, 'normal_stacked'):
            self.stacked_normal = np.column_stack([self.normal_x, self.normal_y])
            self.stacked_normal_T = self.stacked_normal.T
        self.normal_stacked = True
    def get_stacked_normal(self, T=True):
        self.stack_normal()
        return self.stacked_normal_T if T else self.stacked_normal

    def FMM_preparations(self):
        if not hasattr(self, 'prepared_for_FMM'):
            self.stack_boundary()
            self.stack_normal()
        self.prepared_for_FMM = True
