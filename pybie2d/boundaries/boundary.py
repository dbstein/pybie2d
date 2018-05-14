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
        compute_quadrature:
            the workhorse for this class - computes quadrature nodes and weights
            must be provided by child class!
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

        self.tree_computed = False
        if compute_tree:
            self.compute_tree()
    # end __init__2 function definition

    def compute_quadrature(self):
        raise StandardError("'compute_quadrature' function must be provided \
                by the child subclass of type Boundary")
    # end compute_quadrature function

    def find_interior_points(self, target):
        """
        Computes interior/exterior points via cauchy sum +
        brute force search near to boundary, using matplotlib.path
        and treating the boundary as a discrete polygon
        for the brute force search

        if the boundary type has a simple way to deal with this,
        this method should probably be overwritten by the child class!
        """
        return find_interior_points(self, target)
