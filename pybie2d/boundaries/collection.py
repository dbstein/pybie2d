import numpy as np
import scipy as sp
import scipy.spatial
from ..point_set import PointSet

class BoundaryCollection(PointSet):
    """
    Boundary Collection Class
    """
    def __init__(self):
        self.n_boundaries = 0
        self.boundaries = []
        self.sides = []
        self.Ns = []
        self.N = 0
        self.amassed = False
    # end __init__ function definition

    def add(self, b, side):
        """
        Add boundaries to the Boundary_Collection
        Can add single boundaries or multiple boundaries
        Inputs:
            b:    (Boundary or list(Boundary))
            side: (string or list(string)), 'i' or 'e'
        Example usage:
            BC = Boundary_Collection()
            # add single boundary
            BC.add(b, 'i')
            # add multiple boundaries
            BC.add([b1, b2, b3], ['i','e','e'])
        """
        if type(b) is list:
            self._add_list(b, side)
        else:
            self._add_single(b, side)
        self.amassed = False
    # end add function definition

    def _add_single(self, b, side):
        self.n_boundaries += 1
        self.boundaries.append(b)
        self.sides.append(side)
        self.Ns.append(b.N)
        self.N += b.N
        self.NSUM = np.concatenate([ (0,), np.cumsum(np.array(self.Ns)) ])
    # end _add_single function definition

    def _add_list(self, blist, slist):
        for b, side in zip(blist, slist):
            self._add_single(b, side)
    # end _add_list function definition

    def amass_information(self):
        """
        Amass the information required for bulk calls-
        e.g. to FMM and standard SLP and DLP functions
        """
        fields = ('x', 'y', 'c', 'normal_x', 'normal_y', 'normal_c', 'weights',
                  'scaled_speed', 'scaled_cp', 'curvature')
        for field in fields:
            code = 'self.' + field + \
                    ' = np.array([b.' + field + ' for b in self.boundaries])'
            exec(code)
            code = 'self.' + field + ' = np.concatenate(self.' + field + ')'
            exec(code)
        self.stacked_boundary = np.column_stack([self.x, self.y])
        self.stacked_boundary_T = self.stacked_boundary.T
        self.stacked_normal = np.column_stack([self.normal_x, self.normal_y])
        self.stacked_normal_T = self.stacked_normal.T
        self.amassed = True
        self.SLP_vector = \
                np.repeat((np.array(self.sides) == 'e').astype(int), self.Ns)
    # end amass_information function definition

    def get_inds(self, i):
        return self.NSUM[i], self.NSUM[i+1]

    def has_exterior(self):
        return 'e' in self.sides
    # end has_exterior function definition

    def has_interior(self):
        return 'i' in self.sides
    # end has_interior function definition

    def compute_physical_region(self, x, y, return_target=False):
        """
        Compute the physical region. This assumes there is only one boundary
        that is labeled 'i'
        Inputs:
            x ndarray(float): x coordinates for points which are being queried
            y ndarray(float): y coordinates for points which are being queried
        """
        has_interior = self.has_interior()
        if has_interior:
            interior_loc = np.where(np.array(self.sides)=='i')[0][0]
            bh = self.boundaries[interior_loc]
            phys = bh.compute_physical_region(x, y)
        else:
            phys = np.ones(x.shape, dtype=bool)
        for b_ind, b in enumerate(self.boundaries):
            if self.sides[b_ind] == 'e':
                bh = self.boundaries[b_ind]
                phys_here = bh.compute_physical_region(x, y, 'e')
                phys = np.logical_and(phys, phys_here)
        if return_target:
            target = Target(x[phys].flatten(), y[phys].flatten())
            return phys, target
        else:
            return phys
    # end compute_physical_region function definition    
