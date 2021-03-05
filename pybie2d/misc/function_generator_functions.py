from function_generator import FunctionGenerator
from function_generator.error_models import relative_error_model
from scipy.special import k0, k1
import numpy as np
import numba
import time

"""
Generate function generator based versions of k0(x), k1(x).
These are good to 1e-14, in relative error out to x=200. Past x=200, they return 0.
The absolute value for these values is < 2e-88.
"""

fk0 = FunctionGenerator(k0, 0, 200, tol=1e-14, n=8, mw=1e-15, error_model=relative_error_model)
fk1 = FunctionGenerator(k1, 0, 200, tol=1e-14, n=8, mw=1e-15, error_model=relative_error_model)

_fk0 = fk0.get_base_function()
_fk1 = fk1.get_base_function()

@numba.njit()
def _fg_k0(x):
	if x > 200:
		return 0.0
	else:
		return _fk0(x)

@numba.njit()
def _fg_k1(x):
	if x > 200:
		return 0.0
	else:
		return _fk1(x)

@numba.njit(parallel=True)
def fg_k0(x):
	sh = x.shape
	x = x.ravel()
	out = np.zeros(x.size, dtype=np.float64)
	for i in numba.prange(x.size):
		if x[i] <= 200:
			out[i] = _fk0(x[i])
	return out.reshape(sh)

@numba.njit(parallel=True)
def fg_k1(x):
	sh = x.shape
	x = x.ravel()
	out = np.zeros(x.size, dtype=np.float64)
	for i in numba.prange(x.size):
		if x[i] <= 200:
			out[i] = _fk1(x[i])
	return out.reshape(sh)

# test these
if False:
	x = np.linspace(0, 100, 1000*1000)[1:-1]
	_ = fg_k0(x)
	_ = fg_k1(x)

	st=time.time(); ak0 = k0(x);    scipy_k0_time=time.time()-st
	st=time.time(); ek0 = fg_k0(x); funcg_k0_time=time.time()-st
	st=time.time(); ak1 = k1(x);    scipy_k1_time=time.time()-st
	st=time.time(); ek1 = fg_k1(x); funcg_k1_time=time.time()-st
	st=time.time(); ak1 = k1(x);    scipy_k1_time=time.time()-st

	abs_err0 = np.abs(ak0-ek0)/np.abs(ak0)
	abs_err1 = np.abs(ak1-ek1)/np.abs(ak1)

	print('Adjusted relative error, k0: {:0.2e}'.format(abs_err0.max()))
	print('Adjusted relative error, k1: {:0.2e}'.format(abs_err1.max()))
	print('Time (ms), scipy, k0:        {:0.5f}'.format(scipy_k0_time*1000))
	print('Time (ms), scipy, k1:        {:0.5f}'.format(scipy_k1_time*1000))
	print('Time (ms), funcg, k0:        {:0.5f}'.format(funcg_k0_time*1000))
	print('Time (ms), funcg, k1:        {:0.5f}'.format(funcg_k1_time*1000))