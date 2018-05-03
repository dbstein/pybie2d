import numpy as np
import numexpr as ne
import scipy as sp
import scipy.linalg

def interpolate_to_p(x, p):
	"""
	Interpolate a periodic function to the points p
	Multiple functions can be interpolated at a time by maxing x a matrix
	where each column corresponds to a periodic function. All functions must be
	on the interval [0, 2*pi) where only the left endpoint is included.
	To generate an interpolation matrix, input the identity matrix for x.
	"""
	N = x.shape[0]
	k = np.fft.fftfreq(N, 1.0/N)
	xh = np.fft.fft(x, axis=0)
	pn = p[:,None]
	pk = ne.evaluate('pn*k')
	# this is surprisingly a lot faster than a complex exponential
	I = ne.evaluate('(cos(pk) + 1j*sin(pk)) / N')
	interp = I.dot(xh)
	if x.dtype == float:
		interp = interp.real
	return interp

def differentiate(x, d=1):
	n = x.shape[0]
	xh = np.fft.fft(x, axis=0)
	k = np.fft.fftfreq(n, 1.0/n)
	if d == -1:
		k[0] = np.Inf
	if len(x.shape) == 1:
		out = np.fft.ifft(xh*(1j*k)**d)
	else:
		out = np.fft.ifft(xh*(1j*k[:,None])**d, axis=0)
	if x.dtype == float:
		out = out.real
	return out

def differentiation_matrix(N):
    dft = sp.linalg.dft(N)
    idft = dft.conjugate()/N
    k = 1j*np.fft.fftfreq(N, 1.0/N)
    return idft.dot(k[:,None]*dft)
