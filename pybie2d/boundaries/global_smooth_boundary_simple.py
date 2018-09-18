import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os
import numba

from .boundary import Boundary

class Global_Smooth_Boundary(Boundary):
    """
    This class impelements a "global smooth boundary" for use in
    Boundary Integral methods

    Instantiation: see documentation to self.__init__()
        the compute_quadrature method is called by default
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
        super(Global_Smooth_Boundary, self).__init__()
        if self.N % 2 != 0:
            raise Exception('The Global_Smooth_Boundary class only accepts \
                                                                    even N.')
        self.t, self.dt = np.linspace(0, 2*np.pi, self.N, \
                                                endpoint=False, retstep=True)
        self.k = np.fft.fftfreq(self.N, self.dt/(2.0*np.pi)) # fourier modes
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
        self.area = self.dt*np.sum(self.x*self.yp)
        self.perimeter = self.dt*np.sum(self.speed)
    # end __init__ function definition

    """
    Define decorations
    """

    def Kress_Perparations(self):
        """
        Constructs vectors used to speed up Kress/CSLP Routines
        """
        v1 = 4.0*np.sin(np.pi*np.arange(self.N)/self.N)**2
        v1[0] = 1.0
        self.Kress_V1 = 0.5*np.log(v1)/self.dt
        self.Kress_V1_hat = np.fft.fft(self.Kress_V1)
        v2 = np.abs(self.k)
        v2[0] = np.Inf
        IV = 1.0/v2
        self.Kress_V2 = 0.5*np.fft.ifft(IV).real/self.dt
        self.Kress_V2_hat = 0.5*IV/self.dt
        self.Kress_V_hat = self.Kress_V1_hat/self.N + self.Kress_V2_hat
        self.Kress_V = self.Kress_V1/self.N + self.Kress_V2
        RVe = IV.copy()
        RVe[int(self.N/2)+1:] = 0.0
        RVe[int(self.N/2)] /= 2
        RVi = IV.copy()
        RVi[:int(self.N/2)] = 0.0
        self.Kress_VRe = np.fft.ifft(RVi)
        self.Kress_VRi = np.fft.ifft(RVe)
        self.Kress_VRe_hat = RVi
        self.Kress_VRi_hat = RVe
        self.Kress_x = (np.cos(self.t) + 1j*np.sin(self.t))
        self.CSLP_limit =  np.log(1j*self.Kress_x/self.cp) * \
                                            self.weights/(2*np.pi)
        self.sawlog = -1j*self.t + np.log(self.inside_point_c - self.c)
        self.sawlog.imag = np.unwrap(self.sawlog.imag)
        self.inf_scale = self.complex_weights/(self.c-self.inside_point_c) \
                                                                / (2.0j*np.pi)




