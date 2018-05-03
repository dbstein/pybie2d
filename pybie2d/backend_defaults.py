"""
Configures the values at which the computational backend shifts between 
different methods

Note that for matrix formations, the only backend used is numexpr
For matrix applies, the program selects between numba and numexpr based
on the value set in backend_parameters['numba_max']
"""

from . import have_fmm

backend_parameters = {}
backend_parameters['numba_max'] = 10000*10000

def configure_backend_options(numba_max=None):
    """
    Sets backend parameters

    Parameters:
        numba_max: sets the transition point for kernel applies. if the size of
            the total apply (N*M) < numba_max, the computation is done directly
            using the numba backend.  if N*M > numba_max, the computation is
            done using an FMM. If FMM routines were not found (i.e. if 
            have_fmm == False), then direct computation with numba will always
            be used  
    """
    if numba_max is not None:
        backend_parameters['numba_max'] = numba_max

def get_backend(n_source, n_target, backend=None):
    """
    Return which computational backend to use, depending on the size of the
    kernel computation, what resources are available, and the backend requested

    If backend is provided, will check if backend is one of the available
        backends, and return the backend if available, otherwise raise an error
    If backend is not provided:
        will return 'FMM' if n_source*n_target > backend_parameters['numba_max']
            and FMM routines are available (have_fmm==True),
        otherwise will return 'numba'
    """
    size = n_source*n_target
    if backend is None:
        backend = get_backend_internal(size)
    else:
        if backend not in ('numba', 'FMM'):
            raise Exception("Requested backend '" + backend + "' is not an \
                                                        implemented backend.")
        if backend == 'FMM' and not have_fmm:
            raise Exception('FMM backend requested but not found.')
    return backend

def get_backend_internal(size):
    if size > backend_parameters['numba_max'] and have_fmm:
        backend = 'FMM'
    else:
        backend = 'numba'
    return backend
