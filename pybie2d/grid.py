import numpy as np
import scipy as sp
import scipy.spatial
import numba

from .point_set import PointSet

class Grid(PointSet):
    """
    This class impelements a "Grid"
    A subclass of PointSet that has routines optimized for grid structures

    Instantiation: see documentation to self.__init__()
    Methods:
    """
    def __init__(   self, x_bounds, Nx, y_bounds, Ny, mask=None, periodic=True):
        """
        Initialize a Grid.

        x_bounds (required): tuple(float), (x_lower, x_upper)
        Nx       (required): int,          number of points in x grid
        y_bounds (required): tuple(float), (y_lower, y_upper)
        Ny       (required): int,          number of points in y grid
        mask     (optional): bool(Nx, Ny), mask giving points to use/not use
        """
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.Nx = Nx
        self.Ny = Ny
        self.mask = mask
        self.periodic = periodic
        self.endpoint = not self.periodic
        self.xv, self.xh = np.linspace(self.x_bounds[0], self.x_bounds[1],
                            self.Nx, retstep=True, endpoint=self.endpoint)
        self.yv, self.yh = np.linspace(self.x_bounds[0], self.x_bounds[1],
                            self.Nx, retstep=True, endpoint=self.endpoint)
        self.xg, self.yg = np.meshgrid(self.xv, self.yv, indexing='ij')
        if mask is None:
            self.x = self.xg.ravel()
            self.y = self.yg.ravel()
        else:
            self.x = self.xg[mask].ravel()
            self.y = self.yg[mask].ravel()
        super(Grid, self).__init__(self.x, self.y)
        self.shape = (self.Nx, self.Ny)
    # end __init__ function definition

    def find_near_points_numpy(self, boundary, dist):
        """
        Finds all points in self that are within some distance of boundary
        This overwrites the parents class method that depends on trees
        Since it is possible to do much better for a grid...
        """
        search_dist_x = int(np.ceil(dist/self.xh))
        search_dist_y = int(np.ceil(dist/self.yh))
        dist2 = dist**2
        close_pts = np.zeros(self.shape, dtype=bool)
        for i in range(boundary.N):
            x = boundary.x[i]
            y = boundary.y[i]
            x_loc = int(np.floor((x - self.x_bounds[0])/self.xh))
            y_loc = int(np.floor((y - self.y_bounds[0])/self.yh))
            x_lower = x_loc - search_dist_x
            x_upper = x_loc + search_dist_x + 1
            y_lower = y_loc - search_dist_y
            y_upper = y_loc + search_dist_y + 1
            xd = self.xv[x_lower:x_upper] - x
            yd = self.yv[y_lower:y_upper] - y
            dists2 = xd[:,None]**2 + yd**2
            close_pts[x_lower:x_upper, y_lower:y_upper] += dists2 < dist2
        out = close_pts if self.mask is None else close_pts[self.mask]
        return out

    def find_near_points(self, boundary, dist):
        """
        Finds all points in self that are within some distance of boundary
        This overwrites the parents class method that depends on trees
        Since it is possible to do much better for a grid...
        """
        search_dist_x = int(np.ceil(dist/self.xh))
        search_dist_y = int(np.ceil(dist/self.yh))
        dist2 = dist**2
        close_pts = np.zeros(self.shape, dtype=int)
        _find_near_numba(boundary.x, boundary.y, self.xv, self.yv, self.xh,
            self.yh, self.x_bounds[0], self.y_bounds[0], search_dist_x,
            search_dist_y, dist2, close_pts)
        out = close_pts > 0
        out = out if self.mask is None else out[self.mask]
        return out

@numba.njit
def _find_near_numba(x, y, xv, yv, xh, yh, xlb, ylb, xsd, ysd, d2, close):
    N = x.shape[0]
    Nx = xv.shape[0]
    Ny = yv.shape[0]
    for i in numba.prange(N):
        x_loc = int((x[i] - xlb)/xh)
        y_loc = int((y[i] - ylb)/yh)
        x_lower = max(x_loc - xsd, 0)
        x_upper = min(x_loc + xsd + 1, Nx)
        y_lower = max(y_loc - ysd, 0)
        y_upper = min(y_loc + ysd + 1, Ny)
        for j in range(x_lower, x_upper):
            for k in range(y_lower, y_upper):
                xd = xv[j] - x[i]
                yd = yv[k] - y[i]
                dist2 = xd**2 + yd**2
                close[j, k] += int(dist2 < d2)


