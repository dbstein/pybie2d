import numpy as np
import scipy as sp
import scipy.spatial

class PointSet(object):
    """
    This class impelements a "Point Set"

    Instantiation: see documentation to self.__init__()
        note that compute_tree method is called by default in the instantiation
    Methods:
        compute_tree:
            computes a kdtree for the nodes of x
    """
    def __init__(   self, x=None, y=None, c=None, compute_tree=True):
        """
        Initialize a Point Set.
        A kd-tree is computed by default.

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
            self.x = x.flatten()
            self.y = y.flatten()
            self.c = self.x + 1j*self.y
        elif c is not None:
            self.shape = c.shape
            self.c = c.flatten()
            self.x = self.c.real
            self.y = self.c.imag
        else:
            raise StandardError('Not enough parameters provided to define \
                Point Set.')
        self.N = self.x.shape[0]
        self.stacked_boundary = np.column_stack([self.x, self.y])
        self.stacked_boundary_T = self.stacked_boundary.T
        self.tree_computed = False
        if compute_tree:
            self.compute_tree()
    # end __init__ function definition

    def compute_tree(self):
        """
        Compute a kd-tree based on the coordinate values
        """
        if not self.tree_computed:
            self.tree = sp.spatial.cKDTree(self.stacked_boundary)
            self.tree_computed = True
    # end compute_tree function definition

    def reshape(self, f):
        """
        reshape a result to the original shape of the Point Set
        """
        return f.reshape(self.shape)
    # end reshape function definition
