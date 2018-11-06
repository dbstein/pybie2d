import numpy as np
import scipy as sp
import matplotlib as mpl
import warnings
import os

from ..boundary import Boundary
from ...point_set import PointSet
from ...misc.mkl_sparse import SpMV_viaMKL
from ...kernels.high_level.laplace import Laplace_Layer_Form
from ...kernels.high_level.stokes import Stokes_Layer_Form

class Panel_Polygon_Boundary(Boundary):
    """
    Panel Polygon
    (really a container for Panel_Line_Boundaries)

    THIS HAS YET TO BE REWRITTEN FOR THE NEW STYLE PACKAGE

    Currently can only be used for 'i' boundaries
    """
    def __init__(self, xs, ys, hs, rs, order=16, dyadic_levels=16, dyadic_base=2):
        self.xs = np.array(xs) # x coords of nodes
        self.ys = np.array(ys) # y coords of nodes
        self.hs = np.array(hs) # max h between nodes
        self.rs = np.array(rs) # whether to refine around nodes
        self.xsw = np.pad(self.xs,(0,1),mode='wrap')
        self.ysw = np.pad(self.ys,(0,1),mode='wrap')
        self.rsw = np.pad(self.rs,(0,1),mode='wrap')
        self.Nlines = self.xs.shape[0]
        self.order = order
        lines = []
        for i in range(self.Nlines):
            lines.append(_Panel_Line_Boundary( self.xsw[i], self.xsw[i+1],
                            self.ysw[i], self.ysw[i+1], self.hs[i], order,
                            self.rsw[i], dyadic_levels, dyadic_base,
                            self.rsw[i+1], dyadic_levels, dyadic_base
                        ))
        self.lines = lines
        fields = ('x', 'y', 'c', 'normal_x', 'normal_y', 'normal_c',
                  'tangent_x', 'tangent_y', 'tangent_c', 'weights',
                  'scaled_speed', 'scaled_cp', 'curvature')
        for field in fields:
            code = 'self.' + field + \
                    ' = np.array([l.' + field + ' for l in self.lines])'
            exec(code)
            code = 'self.' + field + ' = np.concatenate(self.' + field + ')'
            exec(code)
        self.N = self.x.shape[0]
        super(Panel_Polygon_Boundary, self).__init__(c = self.c)
        self.panels = []
        self.panel_inds = []
        bottom_ind = 0
        for l in self.lines:
            for k in l.panels:
                self.panels.append(k)
                self.panel_inds.append([bottom_ind, bottom_ind + self.order])
                bottom_ind += self.order
        self.complex_weights = self.scaled_cp
    def prepare_oversampling(self, max_h):
        """
        Prepares an oversampled boudnary, and an oversampling routine
        Every panel on the oversampled grid will be at least as resolved as
        specified by the input variable max_h
        """
        # construct fine panels
        structures = []
        for panel in self.panels:
            if panel.h_max <= max_h:
                structures.append(panel)
            else:
                line = _Panel_Line_Boundary( panel.x1, panel.x2, panel.y1,
                                                    panel.y2, max_h, self.order)
                structures.append(line)
        # construct quadrature information from fine panels
        fine_xs = []
        fine_ys = []
        fine_normal_xs = []
        fine_normal_ys = []
        fine_weights = []
        for panel in structures:
            fine_xs = np.concatenate([fine_xs, panel.x])
            fine_ys = np.concatenate([fine_ys, panel.y])
            fine_normal_xs = np.concatenate([fine_normal_xs, panel.normal_x])
            fine_normal_ys = np.concatenate([fine_normal_ys, panel.normal_y])
            fine_weights = np.concatenate([fine_weights, panel.weights])
        # construct a new PointSet with the relevant objects
        fine_boundary = BasicPanel(x=fine_xs, y=fine_ys, nx=fine_normal_xs, ny=fine_normal_ys, w=fine_weights)
        # construct an interpolation matrix
        IMAT = np.zeros([fine_boundary.N, self.N])
        n_panels = len(self.panels)
        tracker = 0
        for i in range(n_panels):
            if type(structures[i]) == _panel:
                TEMP = IMAT[tracker*self.order:(tracker+1)*self.order,i*self.order:(i+1)*self.order]
                np.fill_diagonal(TEMP, 1.0)
                tracker += 1
            if type(structures[i]) == _Panel_Line_Boundary:
                line = structures[i]
                usex = np.abs(line.x2 - line.x1) > 0
                if usex:
                    lx1 = line.x1
                    lx2 = line.x2
                else:
                    lx1 = line.y1
                    lx2 = line.y2
                lxd = lx2 - lx1
                for j in range(line.n_panels):
                    panel = line.panels[j]
                    if usex:
                        px1 = panel.x1
                        px2 = panel.x2
                    else:
                        px1 = panel.y1
                        px2 = panel.y2
                    t1 = (px1 - lx1)/lxd
                    t2 = (px2 - lx1)/lxd
                    # transform t1, t2 to -1, 1 interval
                    t1 = 2*t1 - 1
                    t2 = 2*t2 - 1
                    # get t values to interpolate to
                    tv = (line.gauss_node+1)*(t2-t1)/2.0 + t1
                    C_mat = np.polynomial.legendre.legvander(line.gauss_node, self.order-1)
                    E_mat = np.polynomial.legendre.legvander(tv, self.order-1)
                    I_mat = E_mat.dot(np.linalg.inv(C_mat))
                    IMAT[tracker*self.order:(tracker+1)*self.order,i*self.order:(i+1)*self.order] = I_mat
                    tracker += 1
        return fine_boundary, IMAT

    def Laplace_Correct_Close_Setup(self, target, side=None, stokes=False):
        """
        u: uncorrected evaluation u
        tau: density to evaluate
        target: Target (this target should only contain points on correct side)
        M: correct points that are within M*boundary.max_h of boundary
        """
        t = target
        t.compute_tree()
        self.Close_As = []
        self.Close_Inds = []
        if stokes:
            self.Close_A1s = []
            self.Close_A2s = []
            self.Close_A12s = []
            self.Stokes_DLP_MATS = []
        for panel in self.panels:
            a = panel.x1 + 1j*panel.y1
            b = panel.x2 + 1j*panel.y2
            zsc = (b-a)/2.0
            zmid = (b+a)/2.0
            # get close points
            groups = t.tree.query_ball_point((zmid.real,zmid.imag),
                                                    3.0*np.abs(zsc))
            ind = np.unique(groups)
            if len(ind) > 0:
                pc = panel.c
                tc = t.c[ind]
                Tt = (tc-zmid)/zsc
                Tp = (pc-zmid)/zsc
                Nt = Tt.shape[0]
                ap = np.arange(1,panel.order+1)
                c = (1.0-(-1)**ap)/ap
                V = np.ones((panel.order,panel.order), dtype=complex)
                for k in np.arange(1,panel.order):
                    V[:,k] = V[:,k-1]*Tp
                P = np.zeros((panel.order, Nt), dtype=complex)
                d = 1.5
                inr = np.abs(Tt) <= d       # near
                ifr = np.logical_not(inr)   # far
                # only implemented for side == 'i' right now
                P[0,:] = 1j*np.pi/2.0 + np.log((1-Tt)/(1j*(-1-Tt)))
                if np.sum(inr)>0:
                    for k in np.arange(1, panel.order):
                        P[k,inr] = Tt[inr]*P[k-1,inr] + c[k-1]
                wxp = panel.scaled_cp/zsc
                if np.sum(ifr)>0:
                    for k in np.arange(2,panel.order+1):
                        here1 = wxp*Tp**(k-1)
                        here2 = Tp-Tt[ifr,None]
                        here = here1/here2
                        P[k-1,ifr] = np.sum(here, axis=1)
                A = np.linalg.solve(V.T, P).T * (1j/(2.0*np.pi))
                close_targ = PointSet(tc.real, tc.imag)
                if not stokes:
                    A -= Laplace_Layer_Form(panel, close_targ, ifdipole=True)
                    A = A.real
                else:
                    AS = Stokes_Layer_Form(panel, close_targ, ifdipole=True)
                    # AS = Stokes_DLP(panel, close_targ)
                    self.Stokes_DLP_MATS.append(AS)
                self.Close_As.append(A)
                self.Close_Inds.append(ind)
                if stokes:
                    R1a = 1.0/(1.0-Tt)
                    R1b = 1.0/(1.0+Tt)
                    pp = np.arange(panel.order)[:,None]
                    R2 = pp*np.row_stack(   [np.zeros(Nt,dtype=complex),
                                            P[np.arange(panel.order-1)]])
                    R = -(R1a + (-1)**pp*R1b) + R2
                    A = np.linalg.solve(V.T, R).T * (1j/(2.0*np.pi*zsc))
                    self.Close_A1s.append(A.real)
                    self.Close_A2s.append(-A.imag)
                    self.Close_A12s.append(A.real-1j*A.imag)
            else:
                self.Close_As.append(None)
                self.Close_Inds.append(None)
                if stokes:
                    self.Close_A1s.append(None)
                    self.Close_A2s.append(None)
                    self.Close_A12s.append(None)
                    self.Stokes_DLP_MATS.append(None)
        if not stokes:
            cols = np.array([], dtype=int)
            rows = np.array([], dtype=int)
            dat = np.array([], dtype=float)
            for p_ind, panel in enumerate(self.panels):
                A = self.Close_As[p_ind]
                if A is not None:
                    C = self.Close_Inds[p_ind]
                    panel_inds = self.panel_inds[p_ind]
                    pi = np.arange(panel_inds[0],panel_inds[1])
                    pis = np.tile(pi, A.shape[0])
                    Cs = np.repeat(C, self.order)
                    rows = np.concatenate([rows, Cs])
                    cols = np.concatenate([cols, pis])
                    dat = np.concatenate([dat, A.flatten()])
            M = sp.sparse.coo_matrix((dat, (rows,cols)), \
                    shape=(target.N,self.N), dtype=float)
            self.Sparse_Close_Correction_Mat = M.tocsr()
    # end Laplace_Correct_Close_Setup function definition
    def Stokes_Correct_Close_Setup(self, target, side=None, input_mat=None):
        if input_mat is None:
            self.Laplace_Correct_Close_Setup(target, side, stokes=True)
            cols = np.array([], dtype=int)
            rows = np.array([], dtype=int)
            dat = np.array([], dtype=float)
            snc = np.abs(self.normal_c)
            snx = self.normal_x/self.normal_c
            sny = self.normal_y/self.normal_c
            for p_ind, panel in enumerate(self.panels):
                A = self.Close_As[p_ind]
                if A is not None:
                    C = self.Close_Inds[p_ind]
                    CAI = self.Close_A12s[p_ind]
                    DLP = self.Stokes_DLP_MATS[p_ind]
                    panel_inds = self.panel_inds[p_ind]
                    pi = np.arange(panel_inds[0],panel_inds[1])
                    xp = target.x[self.Close_Inds[p_ind]]
                    yp = target.y[self.Close_Inds[p_ind]]
                    I1 = self.panel_inds[p_ind][0]
                    I2 = self.panel_inds[p_ind][1]
                    snxp = snx[I1:I2]
                    snyp = sny[I1:I2]
                    xh = self.x[I1:I2]
                    yh = self.y[I1:I2]
                    # deal with tau_u
                    fh = np.zeros(self.order, dtype=float)
                    for i in range(self.order):
                        fh *= 0.0
                        fh[i] = 1.0
                        f1 = snxp*fh
                        f2 = snyp*fh
                        f3 = xh*fh
                        corr = A.dot(f1).real + 0j
                        corr += 1j*A.dot(f2).real
                        corr += CAI.dot(f3)
                        corr -= xp*CAI.dot(fh)
                        corru = corr.real
                        corrv = corr.imag
                        corrs = DLP.dot(np.concatenate([fh, fh*0.0]))
                        corru -= corrs[:A.shape[0]]
                        corrv -= corrs[A.shape[0]:]
                        pis = np.repeat(pi[i], A.shape[0])
                        cols = np.concatenate([cols, pis, pis])
                        rows = np.concatenate([rows, C, C+target.N])
                        dat = np.concatenate([dat, corru, corrv])
                    # deal with tau_v
                    for i in range(self.order):
                        fh *= 0.0
                        fh[i] = 1.0
                        f1 = snxp*1j*fh
                        f2 = snyp*1j*fh
                        f3 = yh*fh
                        corr = A.dot(f1).real + 0j
                        corr += 1j*A.dot(f2).real
                        corr += CAI.dot(f3)
                        corr -= yp*CAI.dot(fh)
                        corru = corr.real
                        corrv = corr.imag
                        corrs = DLP.dot(np.concatenate([fh*0.0, fh]))
                        corru -= corrs[:A.shape[0]]
                        corrv -= corrs[A.shape[0]:]
                        pis = np.repeat(pi[i], A.shape[0])
                        cols = np.concatenate([cols, pis+self.N, pis+self.N])
                        rows = np.concatenate([rows, C, C+target.N])
                        dat = np.concatenate([dat, corru, corrv])
            M = sp.sparse.coo_matrix((dat, (rows,cols)), \
                    shape=(2*target.N,2*self.N), dtype=float)
            self.Sparse_Close_Correction_Mat = M.tocsr()
        else:
            self.Sparse_Close_Correction_Mat = input_mat
    def Stokes_Correct_Close(self, tau, u, v, target=None, mu=1.0, side=None):
        out = SpMV_viaMKL(self.Sparse_Close_Correction_Mat, tau)
        N2 = int(out.shape[0]/2)
        u += out[:N2]
        v += out[N2:]
    def Laplace_Correct_Close( self, tau, u, target=None, side=None):
        u += SpMV_viaMKL(self.Sparse_Close_Correction_Mat, tau)
    def find_interior_points(self, target):
        poly = mpl.path.Path(np.column_stack([self.xs, self.ys]))
        phys = poly.contains_points(np.column_stack([target.x, target.y]))
        ext = np.logical_not(phys)
        return phys, ext

class _Panel_Line_Boundary(object):
    """
    This class impelements a "line boundary element" for use in
    Boundary Integral methods

    The quadrature is made using a set of panels

    Instantiation: see documentation to self.__init__()
        note that both compute_quadrature and compute_tree methods
        are called by default in the instantiation
        if you have fairly simple needs, you may be able to set one
        or both of these computations off to save time
    Methods:
        THIS CLASS HAS NO SELF_SLP OR CLOSE_EVALUATION SCHEMES YET!
    """
    def __init__(   self, x1, x2, y1, y2, h, order=16,
                    refine_1=False, dyadic_levels_1=16, dyadic_base_1=4,
                    refine_2=False, dyadic_levels_2=16, dyadic_base_2=4,
                ):
        """
        This function initializes the boundary element. It also computes
        quadrature elements and a kd-tree, by default. Computation of the
        quadrature is fast. Computation of the kd-tree can be turned off if
        it is not needed

        x1: x-coordinate for the endpoint 1
        x2: x-coordinate for the endpoint 2
        y1: y-coordinate for the endpoint 1
        y2: y-coordinate for the endpoint 2
        h:  maximum allowable node spacing
        order: quadrature order
        refine_1: whether to locally refine around endpoint 1
        dyadic_levels_1: number of 'dyadic' levels to be used around endpoint 1
        dyadic_base_1: base for the dyadic refinement around endpoint 1
        refine_2: whether to locally refine around endpoint 2
        dyadic_levels_2: number of 'dyadic' levels to be used around endpoint 2
        dyadic_base_2: base for the dyadic refinement around endpoint 2
        compute_tree: whether or not to compute a tree
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.h = h
        self.order = order
        self.refine_1 = refine_1
        self.dyadic_levels_1 = dyadic_levels_1
        self.dyadic_base_1 = dyadic_base_1
        self.refine_2 = refine_2
        self.dyadic_levels_2 = dyadic_levels_2
        self.dyadic_base_2 = dyadic_base_2
        self.xrange = self.x2 - self.x1
        self.yrange = self.y2 - self.y1
        self.L = np.sqrt(self.xrange**2 + self.yrange**2)
        # set up base gauss quadrature
        self.gauss_node, self.gauss_weight = \
                np.polynomial.legendre.leggauss(self.order)
        self.t_base = (self.gauss_node + 1)/2.0
        # figure out how many panels we need to satisfy the h requirement
        self.t_base_h_max = np.abs(self.t_base[1:] - self.t_base[:-1]).max()
        self.n_main_panels = int(np.ceil(self.L*(self.t_base_h_max/self.h)))
        self.t_cuts = np.linspace(0, 1.0, self.n_main_panels+1)
        # refine around the endpoints
        if self.refine_1:
            self.t_cuts_base = self.t_cuts.copy()
            self.t_cuts = np.empty(self.t_cuts_base.shape[0]
                             + self.dyadic_levels_1, dtype=float)
            self.t_cuts[0] = self.t_cuts_base[0]
            self.t_cuts[self.dyadic_levels_1+1:] = self.t_cuts_base[1:]
            cutter = 1.0/self.dyadic_base_1**np.arange(1,self.dyadic_levels_1+1)
            edge = self.t_cuts_base[1]
            self.t_cuts[1:self.dyadic_levels_1+1] = edge*cutter[::-1]
        if self.refine_2:
            self.t_cuts_base = self.t_cuts.copy()
            self.t_cuts = np.empty(self.t_cuts_base.shape[0]
                             + self.dyadic_levels_2, dtype=float)
            self.t_cuts[:] = -100
            self.t_cuts[-1] = self.t_cuts_base[-1]
            self.t_cuts[:-self.dyadic_levels_2-1] = self.t_cuts_base[:-1]
            cutter = 1.0/self.dyadic_base_2**np.arange(1,self.dyadic_levels_2+1)
            edge = self.t_cuts_base[-2]
            self.t_cuts[-self.dyadic_levels_2-1:-1] = edge + (1-cutter)*(1-edge)
        self.t_cuts_h = self.t_cuts[1:] - self.t_cuts[:-1]
        self.t = (self.t_cuts[:-1][:,None] + self.t_base*self.t_cuts_h[:,None]).flatten()
        self.x = self.x1 + self.xrange*self.t
        self.y = self.y1 + self.yrange*self.t
        self.c = self.x + 1j*self.y
        self.N = self.x.shape[0]
        self.compute_quadrature()
        # get the total number of quadrature nodes
        self.n_panels = self.t_cuts_h.shape[0]
        panels = []
        for i in range(self.n_panels):
            panels.append(_panel(
                    self.x1+self.xrange*self.t_cuts[i],
                    self.x1+self.xrange*self.t_cuts[i+1],
                    self.y1+self.yrange*self.t_cuts[i],
                    self.y1+self.yrange*self.t_cuts[i+1],
                    self.x[self.order*i:self.order*(i+1)],
                    self.y[self.order*i:self.order*(i+1)],
                    self.w[self.order*i:self.order*(i+1)],
                    self.tangent_x[self.order*i:self.order*(i+1)],
                    self.tangent_y[self.order*i:self.order*(i+1)],
                    self.normal_x[self.order*i:self.order*(i+1)],
                    self.normal_y[self.order*i:self.order*(i+1)],
                    self.scaled_cp[self.order*i:self.order*(i+1)],
                    self.scaled_speed[self.order*i:self.order*(i+1)],
                    self.order
                ))
        self.panels = panels

    # end __init__ function definition

    def compute_quadrature(self):
        """
        Compute various parameters related to the boundary parameterization
        That will be needed throughout these computations
        This code is based off of quadr.m ((c) Alex Barnett 10/8/14)
        Names have been made to be more readable. Old names are also included.
        """
        # get the weights
        self.weights = (self.gauss_weight*self.L * (self.t_cuts_h[:,None]/2.0)).flatten()
        self.stacked_boundary = np.column_stack((self.x, self.y))
        self.stacked_boundary_T = self.stacked_boundary.T
        self.curvature = np.zeros(self.N, dtype=float)
        self.tangent_x = np.repeat(self.xrange/self.L, self.N)
        self.tangent_y = np.repeat(self.yrange/self.L, self.N)
        self.tangent_c = self.tangent_x + 1j*self.tangent_y
        self.normal_c = -1j*self.tangent_c
        self.normal_x = self.normal_c.real
        self.normal_y = self.normal_c.imag
        # for taking dot products
        self.ones_vec = np.ones(self.N, dtype=float)
        # old names for compatability
        self.cnx = self.normal_c
        self.nx = self.normal_x
        self.ny = self.normal_y
        self.w = self.weights
        self.scp = self.tangent_c*self.weights
        self.scaled_cp = self.scp
        self.scaled_speed = np.zeros_like(self.scp)
        xd = self.x[1:] - self.x[:-1]
        yd = self.y[1:] - self.y[:-1]
        rd = np.sqrt(xd**2 + yd**2)
        self.h_max = rd.max()
    # end compute_quadrature function definition

class _panel(object):
    def __init__(self, x1, x2, y1, y2, x, y, w, tx, ty, nx, ny, scp, ssp, order):
        self.order = order
        self.N = self.order
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.x = x
        self.y = y
        self.c = self.x + 1j*self.y
        self.w = w
        self.tangent_x = tx
        self.tangent_y = ty
        self.tangent_c = tx + 1j*ty
        self.nx = nx
        self.ny = ny
        self.cnx = self.nx + 1j*self.ny
        self.scp = scp
        self.scaled_speed = ssp
        xd = self.x[1:] - self.x[:-1]
        yd = self.y[1:] - self.y[:-1]
        rd = np.sqrt(xd**2 + yd**2)
        self.h_max = rd.max()
        self.weights = self.w
        self.curvature = np.zeros(order)
        self.normal_x = self.nx
        self.normal_y = self.ny
        self.normal_c = self.cnx
        self.scaled_cp = self.scp
    def stack_normal(self):
        if not hasattr(self, 'normal_stacked'):
            self.stacked_normal = np.column_stack([self.normal_x, self.normal_y])
            self.stacked_normal_T = self.stacked_normal.T
        self.normal_stacked = True
    def get_stacked_normal(self, T=True):
        self.stack_normal()
        return self.stacked_normal_T if T else self.stacked_normal
    def get_stacked_boundary(self, T=True):
        self.stack_boundary()
        return self.stacked_boundary_T if T else self.stacked_boundary
    def stack_boundary(self):
        if not hasattr(self, 'boundary_stacked'):
            self.stacked_boundary = np.column_stack([self.x, self.y])
            self.stacked_boundary_T = self.stacked_boundary.T
            self.boundary_stacked = True

class BasicPanel(object):
    def __init__(self, x, y, nx, ny, w):
        self.x = x
        self.y = y
        self.c = self.x + 1j*self.y
        self.weights = w
        self.normal_x = nx
        self.normal_y = ny
        self.normal_c = self.normal_x + 1j*self.normal_y
        self.N = self.x.shape[0]
    def stack_normal(self):
        if not hasattr(self, 'normal_stacked'):
            self.stacked_normal = np.column_stack([self.normal_x, self.normal_y])
            self.stacked_normal_T = self.stacked_normal.T
        self.normal_stacked = True
    def get_stacked_normal(self, T=True):
        self.stack_normal()
        return self.stacked_normal_T if T else self.stacked_normal
    def get_stacked_boundary(self, T=True):
        self.stack_boundary()
        return self.stacked_boundary_T if T else self.stacked_boundary
    def stack_boundary(self):
        if not hasattr(self, 'boundary_stacked'):
            self.stacked_boundary = np.column_stack([self.x, self.y])
            self.stacked_boundary_T = self.stacked_boundary.T
            self.boundary_stacked = True


