import numpy as np
import scipy as sp
import scipy.spatial

class Boundary(object):
    """
    Parent class for all "Boundaries"
    This class should never be used directly
    Always instantiate Boundaries through the child classes

    Methods:
        compute_quadrature:
            the workhorse for this class - computes quadrature nodes and weights
            must be provided by child class!
        compute_tree:
            computes a kdtree for the physical nodes of the boundary
    """
    def __init__(self):
        self.target_set = False
    # end __init__ function definition

    def __init2__(self, compute_quadrature, compute_tree):
        """
        Secondary initilization done by all Boundary child types after child
        specific initialization is done
        """
        if compute_quadrature:
            self.compute_quadrature()
            self.quadrature_computed = True
        else:
            self.quadrature_computed = False

        if compute_tree:
            self.compute_tree()
            self.tree_computed = True
        else:
            self.tree_computed = False
    # end __init__2 function definition

    def compute_quadrature(self):
        raise StandardError("'compute_quadrature' function must be provided \
                by the child subclass of type Boundary")
    # end compute_quadrature function

    def compute_tree(self):
        """
        Compute a kd-tree based on the coordinate values
        """
        bdy = np.column_stack([self.x, self.y])
        self.tree = sp.spatial.cKDTree(bdy)
        self.tree_computed = True
    # end compute_tree function definition
