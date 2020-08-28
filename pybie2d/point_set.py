import numpy as np
import scipy as sp
import scipy.spatial

from .misc.near_points import find_near_points

class PointSet(object):
    """
    This class impelements a "Point Set"

    Instantiation: see documentation to self.__init__()
        note that compute_tree method is called by default in the instantiation
    Methods:
        compute_tree:
            computes a kdtree for the nodes of x
    """
    def __init__(self, x=None, y=None, c=None):
        """
        Initialize a Point Set.

        x (optional): real vector of x-coordinates
        y (optional): real vector of y-coordinates
        c (optional): complex vector with c.real giving x-coordinates
            and c.imag giving y-coordinates

        The user must provide at least one of the following sets of inputs:
            (1) x and y            
                (x and y positions, as real vectors)
            (2) c
                (x and y positions as a complex vector, x=c.real, y=cimag)
        If both are provided the real vectors will be used
        """
        if x is not None and y is not None:
            self.shape = x.shape
            self.x = x.ravel()
            self.y = y.ravel()
            self.c = self.x + 1j*self.y
        elif c is not None:
            self.shape = c.shape
            self.c = c.ravel()
            self.x = self.c.real
            self.y = self.c.imag
        else:
            raise Exception('Not enough parameters provided to define Point Set.')
        self.N = self.x.shape[0]
    # end __init__ function definition

    def get_stacked_boundary(self, T=True):
        self.stack_boundary()
        return self.stacked_boundary_T if T else self.stacked_boundary

    def stack_boundary(self):
        if not hasattr(self, 'boundary_stacked'):
            self.stacked_boundary = np.column_stack([self.x, self.y])
            self.stacked_boundary_T = self.stacked_boundary.T
            self.boundary_stacked = True

    def compute_tree(self):
        """
        Compute a kd-tree based on the coordinate values
        """
        if not hasattr(self, 'tree'):
            self.tree = sp.spatial.cKDTree(self.get_stacked_boundary(T=False))
    # end compute_tree function definition

    def reshape(self, f):
        """
        reshape a result to the original shape of the Point Set
        """
        return f.reshape(self.shape)
    # end reshape function definition

    def find_near_points(self, boundary, dist):
        """
        Finds all points in self that are within some distance of boundary
        """
        self.compute_tree()
        close_pts = find_near_points(boundary, self, dist)
        return self.reshape(close_pts)

