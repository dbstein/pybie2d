import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os

from ..misc.basic_functions import interpolate_to_p, differentiate, differentiation_matrix
from .boundary import Boundary

class Global_Smooth_Boundary(Boundary):
    """
    This class impelements a "global smooth boundary" for use in
    Boundary Integral methods

    Instantiation: see documentation to self.__init__()
        note that both compute_quadrature and compute_tree methods
        are called by default in the instantiation
        if you have fairly simple needs, you may be able to set one
        or both of these computations off to save time
    Public Methods:
        all parent methods (see documentation for Boundary class)
        Laplace_Correct_Close
        Stokes_Correct_Close
    """
    def __init__(   self, x=None, y=None, c=None, x_func=None, y_func=None,
                    c_func=None, xp_func=None, yp_func=None, cp_func=None,
                    xpp_func=None, ypp_func=None, cpp_func=None, N=None,
                    inside_point=None, compute_quadrature=True,
                    compute_tree=True, self_type='kress'):
        """
        This function initializes the boundary element. It also computes
        quadrature elements and a kd-tree, by default. These computations can
        be turned off if speed is desired and they aren't needed.

        x (optional): real vector of x-coordinates
        y (optional): real vector of y-coordinates
        c (optional): complex vector with c.real giving x-coordinates
            and c.imag giving y-coordinates
        # the following functions must take a real parameter vector defined
          on [0, 2pi) and return an array of the appropriate type
          * denotes x, y, or c; if x the function returns x values,
          y returns y values, and c returns complex values with x = c.real and
          y = c.imag
        *_func (optional): coordinates
        *p_func (optional): first parameter derivative of coordinates
        *pp_func (optional): second parameter derivative of coordinates
        inside_point (optional): either tuple/list/array (real, real)
            or complex scalar giving coordinates for a point inside s,
            preferably far from the boundary if this is not provided, will be
            computed as (x.mean(), y.mean()), which may not be inside!
        self_type (optional): 'kress'; (only kress implemented for now)
        The user must provide at least one of the following sets of inputs:
            (1) x and y            
                (x and y positions, as real vectors)
            (2) c
                (x and y positions as a complex vector, x=c.real, y=cimag)
            (3) x_func, y_func, N
                (functions returning x, y positions as real vectors given a
                parameter vector on [0,2*pi])
                the parameterization will be constructed to have N elements
            (4) c_func, N
                (function returning x, y positions as a complex vector given a
                paremeter vector on [0,2*pi])
                the parameterization will be constructed to have N elements

        If real and complex functions or positions are provided real values
            will be used
        If values and functions are provided, functions will be used
        Functions for derivatives are only used if functions for coordinates are
            provided and if the functions for derivatives use the same
            (real/complex) type as the functions for coordinates
        If derivative functions are not provided spectral differentiation is
            used to compute the derivatives
        If inside_point is not provided, it will be computed as the mean
            this may not actually be inside!

        As of now, its not clear to me that the Alpert quadrature scheme will
        work if n is odd. For now, I am throwing an error if N is odd
        """
        super(Global_Smooth_Boundary, self).__init__()

        real_functions_available = x_func is not None and y_func is not None
        complex_function_available = c_func is not None
        functions_available = real_functions_available or \
                                complex_function_available
        N_available = N is not None
        use_functions = N_available and functions_available

        self.extras =   {   'compute_quadrature' : compute_quadrature,
                            'compute_tree' : compute_tree,
                        }

        if use_functions:
            self.N = N
            self.t, self.h = np.linspace(0.0, 2.0*np.pi, self.N,
                                    endpoint=False, retstep=True)
            if real_functions_available:
                self.x_func = x_func
                self.y_func = y_func
                self.x = self.x_func(self.t)
                self.y = self.y_func(self.t)
                self.c = self.x + 1j*self.y
                self.position_functions = 'Real'
                if xp_func is not None and yp_func is not None:
                    self.xp_func = xp_func
                    self.yp_func = yp_func
                    self.first_derivative_functions = 'Real'
                else:
                    self.first_derivative_functions = 'None'
                if xpp_func is not None and ypp_func is not None:
                    self.xpp_func = xpp_func
                    self.ypp_func = ypp_func
                    self.second_derivative_functions = 'Real'
                else:
                    self.second_derivative_functions = 'None'
            else:
                self.c_func = c_func
                self.c = self.c_func(self.t)
                self.x = self.c.real
                self.y = self.c.imag
                self.position_functions = 'Complex'
                if cp_func is not None:
                    self.cp_func = cp_func
                    self.first_derivative_functions = 'Complex'
                else:
                    self.first_derivative_functions = 'None'
                if cpp_func is not None:
                    self.cpp_func = cpp_func
                    self.second_derivative_functions = 'Complex'
                else:
                    self.second_derivative_functions = 'None'
        elif x is not None and y is not None:
            self.x = x
            self.y = y
            self.c = self.x + 1j*self.y
            self.N = self.x.shape[0]
            self.t, self.h = np.linspace(0.0, 2*np.pi, self.N,
                                endpoint=False, retstep=True)
            self.position_functions = 'None'
            self.first_derivative_functions = 'None'
            self.second_derivative_functions = 'None'
        elif c is not None:
            self.c = c
            self.x = c.real
            self.y = c.imag
            self.N = self.c.shape[0]
            self.t, self.h = np.linspace(0.0, 2*np.pi, self.N,
                                endpoint=False, retstep=True)
            self.first_derivative_functions = 'None'
            self.second_derivative_functions = 'None'
        else:
            raise StandardError('Unacceptable set of parameters provided.')
        if inside_point is not None:
            if type(inside_point) == complex:
                self.inside_point_c = inside_point
                self.inside_point_x = self.inside_point_c.real
                self.inside_point_y = self.inside_point_c.imag
            else:
                self.inside_point_x = inside_point[0]
                self.inside_point_y = inside_point[1]
                self.inside_point_c = self.inside_point_x + \
                                        1j*self.inside_point_y
        else:
            self.inside_point_c = np.mean(self.c)
            self.inside_point_x = self.inside_point_c.real
            self.inside_point_y = self.inside_point_c.imag

        if self.N % 2 != 0:
            raise StandardError('The closed_boundary_element class currently \
                only accepts N being even.')

        self.self_type = self_type

        self.__init2__(compute_quadrature, compute_tree)
    # end __init__ function definition

    #########################
    #### Public Methods ####
    #########################

    def compute_quadrature(self):
        """
        Compute various parameters related to the boundary parameterization
        That will be needed throughout these computations
        This code is based off of quadr.m ((c) Alex Barnett 10/8/14)
        Names have been made to be more readable. Old names are also included.

        Note that 
        """
        self.stacked_boundary = np.column_stack([self.x, self.y])
        self.stacked_boundary_T = self.stacked_boundary.T
        self.k = np.fft.rfftfreq(self.N, self.h/(2.0*np.pi)) # fourier modes
        self.ik = 1j*self.k
        self.ik2 = self.ik**2
        self.xhat = np.fft.rfft(self.x)
        self.yhat = np.fft.rfft(self.y)
        if self.first_derivative_functions == 'Real':
            self.xp = self.xp_func(self.t)
            self.yp = self.yp_func(self.t)
            self.cp = self.xp + 1j*self.yp
        elif self.first_derivative_functions == 'Complex':
            self.cp = self.cp_func(self.t)
            self.xp = self.cp.real
            self.yp = self.cp.imag
        else:
            self.xp = np.fft.irfft(self.xhat*self.ik)
            self.yp = np.fft.irfft(self.yhat*self.ik)
            self.cp = self.xp + 1j*self.yp
        if self.second_derivative_functions == 'Real':
            self.xpp = self.xpp_func(self.t)
            self.ypp = self.ypp_func(self.t)
            self.cpp = self.xpp + 1j*self.ypp
        elif self.second_derivative_functions == 'Complex':
            self.cpp = self.cpp_func(self.t)
            self.xpp = self.cpp.real
            self.ypp = self.cpp.imag
        else:
            self.xpp = np.fft.irfft(self.xhat*self.ik2)
            self.ypp = np.fft.irfft(self.yhat*self.ik2)
            self.cpp = self.xpp + 1j*self.ypp
        self.speed = np.abs(self.cp)
        self.tangent_c = self.cp/self.speed
        self.tangent_x = self.tangent_c.real
        self.tangent_y = self.tangent_c.imag
        self.normal_c = -1.0j*self.tangent_c
        self.normal_x = self.normal_c.real
        self.normal_y = self.normal_c.imag
        self.stacked_normal = np.column_stack([self.normal_x, self.normal_y])
        self.stacked_normal_T = self.stacked_normal.T
        self.tangent_c = 1.0j*self.normal_c
        self.tangent_x = self.tangent_c.real
        self.tangent_y = self.tangent_c.imag
        self.curvature = -(np.conj(self.cpp)*self.normal_c).real/self.speed**2
        self.weights = self.h*self.speed
        self.complex_weights = self.h*self.cp
        self.scaled_cp = self.cp/self.N
        self.scaled_speed = self.speed/self.N
        self.max_h = 2.0*np.pi*np.max(self.scaled_speed)
        self.quadrature_computed = True
        self.area = 2.0*np.pi*np.sum(self.x*self.yp)/self.N
        self.perimeter = 2.0*np.pi*np.sum(self.speed)/self.N
        # for taking dot products
        self.ones_vec = np.ones(self.N, dtype=float)
        # Alex Barnett's old names
        self.sp = self.speed
        self.tang = self.tangent_c
        self.cur = self.curvature
        self.w = self.weights
        self.cw = self.complex_weights
        self.a = self.inside_point_c
        # my old names (to be removed once the code is changed)
        self.ctang = self.tang
        self.xtang = self.tangent_x
        self.ytang = self.tangent_y
        self.cnx = self.normal_c
        self.nx = self.normal_x
        self.ny = self.normal_y
        self.cpn = self.scaled_cp
        self.spn = self.scaled_speed
    # end compute_quadrature function definition

    # self quadrature (apply) for Laplace SLP
    def Laplace_SLP_Self_Apply(self, tau):
        """
        Apply Laplace SLP self-interaction kernel

        Inputs:
            tau,    required, dtype(ns): density (dtype can be float or complex)
        """
        return _Laplace_SLP_Self_Kress_Apply(self.x, self.y, self.speed, tau)
    # end Laplace_SLP_Self_Apply function definition

    # self quadrature (form) for Laplace SLP
    def Laplace_SLP_Self_Form(self):
        """
        Form Laplace SLP self-interaction matrix
            (and normal derivative matrix, if requested)
        """
        return _Laplace_SLP_Self_Kress_Form(self.x, self.y, self.speed)
    # end Laplace_SLP_Self_Apply function definition

    #########################
    #### Private Methods ####
    #########################

    def _test_inside_point(self, eps=1e-10):
        """
        Test whether the provided or generated inside point is acceptable
        returns True if the point is okay, False if its not
        """
        if not self.quadrature_computed:
            self.compute_quadrature()
        test_value = np.sum(self.complex_weights/(self.c-self.inside_point_c))
        return np.abs(test_value - 2.0j*np.pi) < eps
    # end _test_inside_point function
##### End of Global_Smooth_Boundary class definition ###########################

################################################################################
#### Kress Self-Evaluation routines for Laplace SLP ############################
################################################################################

def _Laplace_SLP_Self_Kress_Form(sx, sy, speed):
    """
    sx,    required, float(ns), x-coordinates
    sy,    required, float(ns), y-coordinates
    speed, required, float(ns), speed of discretization
    """
    N = sx.shape[0]
    sxt = sx[:,None]
    syt = sy[:,None]
    dx = ne.evaluate('sxt - sx')
    dy = ne.evaluate('syt - sy')
    d2 = dx**2 + dy**2
    sw = -speed/N
    C = sp.linalg.circulant(0.5*np.log(4.0*np.sin(np.pi*np.arange(N)/N)**2))
    A = ne.evaluate('-0.5*log(d2) + C')
    np.fill_diagonal(A, -np.log(speed))
    V1 = 1.0/np.arange(1,int(N/2)+1)
    V = np.concatenate([ (0.0,), V1, V1[:-1][::-1] ])
    IV = 0.5*np.fft.ifft(V).real
    C = sp.linalg.circulant(IV)
    return ne.evaluate('(A/N + C)*speed')

def _Laplace_SLP_Self_Kress_Apply(sx, sy, speed, tau):
    """
    sx,    required, float(ns), x-coordinates
    sy,    required, float(ns), y-coordinates
    speed, required, float(ns), speed of discretization
    tau,   required, dtype(ns), density

    for now, just form the matrix and apply it
    this should be recoded using numba
    """
    A = _Laplace_SLP_Self_Kress_Form
    return A.dot(tau)

