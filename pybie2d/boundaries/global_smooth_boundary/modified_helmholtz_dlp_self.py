import numpy as np
import scipy as sp
import scipy.spatial
import scipy.signal
import numexpr as ne
import warnings
import os

from ...kernels.high_level.laplace import Laplace_Layer_Singular_Form
from ...misc.numba_special_functions import numba_k0, numba_k1
from ...misc.basic_functions import interpolate_to_p

alpert_order = 16

if alpert_order == 2:
    alpert_x = np.array ([  1.591549430918953e-01 ])
    alpert_w = np.array ([  5.0e-01 ])
    alpert_a = 1
elif alpert_order == 3:
    alpert_x = np.array ([  1.150395811972836e-01, 9.365464527949632e-01 ])
    alpert_w = np.array ([  3.913373788753340e-01, 1.108662621124666e+00 ])
    alpert_a = 2
elif alpert_order == 4:
    alpert_x = np.array ([  2.379647284118974e-02, 2.935370741501914e-01, 1.023715124251890e+00 ])
    alpert_w = np.array ([  8.795942675593887e-02, 4.989017152913699e-01, 9.131388579526912e-01 ])
    alpert_a = 2
elif alpert_order == 6:
    alpert_x = np.array ([  4.004884194926570e-03, 7.745655373336686e-02, 3.972849993523248e-01, 1.075673352915104e+00, 2.003796927111872e+00 ])
    alpert_w = np.array ([  1.671879691147102e-02, 1.636958371447360e-01, 4.981856569770637e-01, 8.372266245578912e-01, 9.841730844088381e-01 ])
    alpert_a = 3
elif alpert_order == 8:
    alpert_x = np.array ([  6.531815708567918e-03, 9.086744584657729e-02, 3.967966533375878e-01, 1.027856640525646e+00, 1.945288592909266e+00, 2.980147933889640e+00, 3.998861349951123e+00 ])
    alpert_w = np.array ([  2.462194198995203e-02, 1.701315866854178e-01, 4.609256358650077e-01, 7.947291148621894e-01, 1.008710414337933e+00, 1.036093649726216e+00, 1.004787656533285e+00 ])
    alpert_a = 5
elif alpert_order == 16:
    alpert_x = np.array([ 8.371529832014113E-04, 1.239382725542637E-02, 6.009290785739468E-02, 1.805991249601928E-01, 4.142832599028031E-01, 7.964747731112430E-01, 1.348993882467059E+00, 2.073471660264395E+00, 2.947904939031494E+00, 3.928129252248612E+00, 4.957203086563112E+00, 5.986360113977494E+00, 6.997957704791519E+00, 7.999888757524622E+00, 8.999998754306120E+00 ])
    alpert_w = np.array([ 3.190919086626234E-03, 2.423621380426338E-02, 7.740135521653088E-02, 1.704889420286369E-01, 3.029123478511309E-01, 4.652220834914617E-01, 6.401489637096768E-01, 8.051212946181061E-01, 9.362411945698647E-01, 1.014359775369075E+00, 1.035167721053657E+00, 1.020308624984610E+00, 1.004798397441514E+00, 1.000395017352309E+00, 1.000007149422537E+00 ])
    alpert_a = 10
else:
    raise Exception('Specified Alpert Order not defined.')
alpert_g = 2*alpert_x.shape[0]
alpert_W =  np.concatenate([ alpert_w[::-1], alpert_w ])

class Modified_Helmholtz_DLP_Self(object):
    """
    Module providing Modified Helmholtz DLP Self-Evaluation
    """
    def __init__(self, GSB):
        """
        Initializes the Modified Helmholtz DLP Self-Evaluation
        """
        self.boundary = GSB
        self.alpert_X = np.concatenate([ 2.0*np.pi - alpert_x[::-1]*self.boundary.dt, alpert_x*self.boundary.dt ])
        self.alpert_I = interpolate_to_p(np.eye(self.boundary.N), self.alpert_X)
        self.mats = {}

    def Form(self, k):
        if k not in self.mats.keys():
            self.mats[k] = Modified_Helmholtz_DLP_Self_Form(self.boundary, k, self.alpert_I)
        return self.mats[k]

    def Apply(self, k, tau, backend='fly'):
        self.Form(k)
        return self.mats[k].dot(tau)

def Modified_Helmholtz_DLP_Self_Form(source, k, aI):
    DL = Laplace_Layer_Singular_Form(source, ifdipole=True)
    C = Alpert_Correction(source.x, source.y, source.normal_x, source.normal_y, source.weights, k, aI)
    C /= source.weights
    C = C.T*source.weights
    return DL-C

def Alpert_Correction(x, y, nx, ny, w, k, aI):
    N = x.shape[0]
    sel1 = np.arange(N)
    sel2 = sel1[:,None]
    sel = np.mod(sel1 + sel2, N)
    Yx = x[sel]
    Yy = y[sel]
    IYx = aI.dot(Yx)
    IYy = aI.dot(Yy)
    FYx = np.row_stack(( IYx, Yx[alpert_a:(-alpert_a+1)] ))
    FYy = np.row_stack(( IYy, Yy[alpert_a:(-alpert_a+1)] ))
    W = w[sel]
    IW = aI.dot(W)*alpert_W[:,None]
    FW = np.row_stack(( IW, W[alpert_a:(-alpert_a+1)] ))
    # evaluate greens function
    dx = x - FYx
    dy = y - FYy
    r = np.sqrt(dx**2 + dy**2)
    kr = k*r
    inv_twopi = 0.5/np.pi
    k1kr_m_kir2 = inv_twopi*(k*numba_k1(kr)/r - (1.0/r)**2)
    GFx = dx*k1kr_m_kir2
    GFy = dy*k1kr_m_kir2
    A = ((nx*GFx+ny*GFy)*FW).T
    # reconstruct
    Ag = A[:,:alpert_g]
    IAg = Ag.dot(aI)
    IAg[:,alpert_a:(-alpert_a+1)] += A[:,alpert_g:]
    # reorganize
    inv_sel = np.mod((sel1 + sel2[::-1] + 1), N)
    return IAg[sel1, inv_sel.T].T
