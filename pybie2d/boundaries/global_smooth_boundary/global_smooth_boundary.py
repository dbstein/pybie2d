import numpy as np
import scipy as sp
import scipy.signal
import warnings
import os

from ..boundary import Boundary
from .laplace_slp_self_kress import Laplace_SLP_Self_Kress
from .stokes_slp_self_kress import Stokes_SLP_Self_Kress
from .laplace_cslp_self_kress import Laplace_CSLP_Self_Kress
from .laplace_close_quad import Laplace_Close_Quad
from .stokes_close_quad import Stokes_Close_Quad
from .modified_helmholtz_dlp_self import Modified_Helmholtz_DLP_Self
from .modified_helmholtz_slp_self import Modified_Helmholtz_SLP_Self
from .modified_helmholtz_close_quad import Modified_Helmholtz_Close_Quad
from .stokes_slp_self_traction import Stokes_SLP_Self_Traction

def my_resample(f, n):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sp.signal.resample(f, n)

class Global_Smooth_Boundary(Boundary):
    """
    This class impelements a "global smooth boundary" for use in
    Boundary Integral methods

    Instantiation: see documentation to self.__init__()
    """
    def __init__(self, x=None, y=None, c=None):
        """
        This function initializes the boundary element.

        x (optional): real vector of x-coordinates
        y (optional): real vector of y-coordinates
        c (optional): complex vector with c.real giving x-coordinates
            and c.imag giving y-coordinates
        The user must provide at least one of the following sets of inputs:
            (1) x and y            
                (x and y positions, as real vectors)
            (2) c
                (x and y positions as a complex vector, x=c.real, y=cimag)
        If inside_point is not provided, it will be computed as the mean
            this may not actually be inside!

        As of now, its not clear to me that everything will
        work if n is odd. For now, I will throw an error if x/y/c have an
        odd number of elements in them
        """
        super(Global_Smooth_Boundary, self).__init__(x, y, c)
        if self.N % 2 != 0:
            raise Exception('The Global_Smooth_Boundary class only accepts \
                                                                    even N.')
        self.t, self.dt = np.linspace(0, 2*np.pi, self.N, \
                                                endpoint=False, retstep=True)
        self.k = np.fft.fftfreq(self.N, self.dt/(2.0*np.pi)) # fourier modes
        self.k[int(self.N/2)] = 0.0 # wipe out nyquist frequency
        self.ik = 1j*self.k
        self.chat = np.fft.fft(self.c)
        self.cp = np.fft.ifft(self.chat*self.ik)
        self.cpp = np.fft.ifft(self.chat*self.ik**2)
        self.speed = np.abs(self.cp)
        self.tangent_c = self.cp/self.speed
        self.tangent_x = self.tangent_c.real
        self.tangent_y = self.tangent_c.imag
        self.normal_c = -1.0j*self.tangent_c
        self.normal_x = self.normal_c.real
        self.normal_y = self.normal_c.imag
        self.curvature = -(np.conj(self.cpp)*self.normal_c).real/self.speed**2
        self.weights = self.dt*self.speed
        self.complex_weights = self.dt*self.cp
        self.scaled_cp = self.cp/self.N
        self.scaled_speed = self.speed/self.N
        self.max_h = np.max(self.weights)
        self.area = self.dt*np.sum(self.x*self.cp.imag)
        self.perimeter = self.dt*np.sum(self.speed)

        self.defined_modules =  [   'Laplace_SLP_Self_Kress',
                                    'Stokes_SLP_Self_Kress',
                                    'Laplace_CSLP_Self_Kress',
                                    'Laplace_Close_Quad',
                                    'Stokes_Close_Quad',
                                    'Modified_Helmholtz_DLP_Self',
                                    'Modified_Helmholtz_SLP_Self',
                                    'Modified_Helmholtz_Close_Quad',
                                    'Stokes_SLP_Self_Traction',
                                ]
    # end __init__ function definition

    def add_module(self, name):
        """
        Add a module to boundary to obtain specific functionality
        """
        if not name in self.defined_modules:
            msg = "Module '" + name + "' is not a known module for Global_Smooth_Boundary class."
            raise Exception(msg)
        if not hasattr(self, name):
            setattr(self, name, eval(name + '(self)'))
    # end add_module function definition

    def set_inside_point(self, c):
        """
        Set an inside point, used in close evaluation schemes
        If self-eval schemes are called before this is set, this will be
        computed as the average of the boundary nodes, which may not be inside
        c should be an imaginary float, with c.real=x, c.imag=y
        """
        good = self._test_inside_point(c)
        if not good:
            warnings.warn('Inside point failed basic test, is it actually inside?')
        self.inside_point_c = c
    # end set_inside_point function definition

    def get_inside_point(self):
        if not hasattr(self, 'inside_point_c'):
            candidate = np.sum(self.c)/self.N
            good = self._test_inside_point(candidate)
            if not good:
                warnings.warn('Inside point computed as mean, failed basic test, is it actually inside?')
            self.inside_point_c = candidate
        return self.inside_point_c
    # end get_inside_point function definition

    def generate_resampled_boundary(self, new_N):
        sfc = my_resample(self.c, new_N)
        # sfc = sp.signal.resample(self.c, new_N)
        return Global_Smooth_Boundary(c=sfc)
    # end generate_resampled_boundary definition

    #########################
    #### Public Methods #####
    #########################

    ##### These provide interfaces so that high level funcitons
    ##### Can easily extract the functions that they need

    # self quadrature (apply) for Laplace SLP
    def Laplace_SLP_Self_Apply(self, *args, **kwargs):
        self.add_module('Laplace_SLP_Self_Kress')
        return self.Laplace_SLP_Self_Kress.Apply(*args, **kwargs)

    # self quadrature (form) for Laplace SLP
    def Laplace_SLP_Self_Form(self, *args, **kwargs):
        self.add_module('Laplace_SLP_Self_Kress')
        return self.Laplace_SLP_Self_Kress.Form(*args, **kwargs)

    # self quadrature (apply) for Stokes SLP
    def Stokes_SLP_Self_Apply(self, *args, **kwargs):
        self.add_module('Stokes_SLP_Self_Kress')
        return self.Stokes_SLP_Self_Kress.Apply(*args, **kwargs)

    # self quadrature (form) for Stokes SLP
    def Stokes_SLP_Self_Form(self, *args, **kwargs):
        self.add_module('Stokes_SLP_Self_Kress')
        return self.Stokes_SLP_Self_Kress.Form(*args, **kwargs)

    # self quadrature (form) for Modified Helmholtz DLP
    def Modified_Helmholtz_DLP_Self_Form(self, *args, **kwargs):
        self.add_module('Modified_Helmholtz_DLP_Self')
        return self.Modified_Helmholtz_DLP_Self.Form(*args, **kwargs)

    # self quadrature (form) for Modified Helmholtz SLP
    def Modified_Helmholtz_SLP_Self_Form(self, *args, **kwargs):
        self.add_module('Modified_Helmholtz_SLP_Self')
        return self.Modified_Helmholtz_SLP_Self.Form(*args, **kwargs)

    # generate Minimum Global Smooth Boundary based on first point
    # this is very useful for Circulant problems
    def Generate_1pt_Circulant_Boundary(self):
        return Minimal_Global_Smooth_Boundary(self.c[0], self.normal_c[0], 
                    self.weights[0], self.curvature[0])

    ###### Method for generating close corrections
    def tolerance_to_distance(self, tol):
        """
        Given error tolerance, finds distance where close evaluation is needed
        """
        return -np.log(tol)*self.max_h/4.5

    def Get_Close_Corrector(self, kernel, *args, **kwargs):
        if kernel == 'laplace':
            self.add_module('Laplace_Close_Quad')
            return self.Laplace_Close_Quad.Get_Close_Corrector(*args, **kwargs)
        elif kernel == 'stokes':
            self.add_module('Stokes_Close_Quad')
            return self.Stokes_Close_Quad.Get_Close_Corrector(*args, **kwargs)
        elif kernel == 'modified_helmholtz':
            self.add_module('Modified_Helmholtz_Close_Quad')
            return self.Modified_Helmholtz_Close_Quad.Get_Close_Corrector(*args, **kwargs)
        else:
            raise Exception("Specified kernel: '" + kernel + "' not recognized.")

    #########################
    #### Private Methods ####
    #########################

    def _test_inside_point(self, candidate, eps=1e-10):
        """
        Test whether the provided or generated inside point is acceptable
        returns True if the point is okay, False if its not
        """
        test_value = np.sum(self.complex_weights/(self.c-candidate))
        return np.abs(test_value - 2.0j*np.pi) < eps
    # end _test_inside_point function

def arr_check(x):
    if type(x) != np.ndarray:
        x = np.array([x,])
    return x
class Minimal_Global_Smooth_Boundary(Global_Smooth_Boundary):
    def __init__(self, c, normal_c, weights, curvature):
        self.c = arr_check(c)
        self.N = self.c.size
        self.normal_c = arr_check(normal_c)
        self.weights = arr_check(weights)
        self.curvature = arr_check(curvature)
        self.x = self.c.real
        self.y = self.c.imag
        self.normal_x = self.normal_c.real
        self.normal_y = self.normal_c.imag
