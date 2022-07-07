import numpy as np
import scipy.sparse as sparse
from ctypes import POINTER,c_void_p,c_int,c_char,c_double,byref,cdll
import platform
try:
    if platform.system() == 'Darwin':
        mkl = cdll.LoadLibrary("libmkl_rt.dylib")
    else:
        mkl = cdll.LoadLibrary("libmkl_rt.so")
    mkl_is_here = True
except:
    mkl_is_here = False

if mkl_is_here:
    def SpMV_viaMKL( A, x ):
        """
        Wrapper to Intel's SpMV
        (Sparse Matrix-Vector multiply)
        For medium-sized matrices, this is 4x faster
        than scipy's default implementation
        Stephen Becker, April 24 2014
        stephen.beckr@gmail.com
        """

        # import numpy as np
        # import scipy.sparse as sparse
        # from ctypes import POINTER,c_void_p,c_int,c_char,c_double,byref,cdll
        # mkl = cdll.LoadLibrary("libmkl_rt.so")

        SpMV = mkl.mkl_cspblas_dcsrgemv
        # Dissecting the "cspblas_dcsrgemv" name:
        # "c" - for "c-blas" like interface (as opposed to fortran)
        #    Also means expects sparse arrays to use 0-based indexing, which python does
        # "sp"  for sparse
        # "d"   for double-precision
        # "csr" for compressed row format
        # "ge"  for "general", e.g., the matrix has no special structure such as symmetry
        # "mv"  for "matrix-vector" multiply

        if not sparse.isspmatrix_csr(A):
            raise Exception("Matrix must be in csr format")
        (m,n) = A.shape

        # The data of the matrix
        data    = A.data.ctypes.data_as(POINTER(c_double))
        indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
        indices = A.indices.ctypes.data_as(POINTER(c_int))

        # Allocate output, using same conventions as input
        nVectors = 1
        if x.ndim is 1:
           y = np.empty(m,dtype=np.double,order='F')
           if x.size != n:
               raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
        elif x.shape[1] is 1:
           y = np.empty((m,1),dtype=np.double,order='F')
           if x.shape[0] != n:
               raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
        else:
           nVectors = x.shape[1]
           y = np.empty((m,nVectors),dtype=np.double,order='F')
           if x.shape[0] != n:
               raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))

        # Check input
        if x.dtype.type is not np.double:
           x = x.astype(np.double,copy=True)
        # Put it in column-major order, otherwise for nVectors > 1 this FAILS completely
        if x.flags['F_CONTIGUOUS'] is not True:
           x = x.copy(order='F')

        if nVectors == 1:
           np_x = x.ctypes.data_as(POINTER(c_double))
           np_y = y.ctypes.data_as(POINTER(c_double))
           # now call MKL. This returns the answer in np_y, which links to y
           SpMV(byref(c_char(b"N")), byref(c_int(m)),data ,indptr, indices, np_x, np_y ) 
        else:
           for columns in range(nVectors):
               xx = x[:,columns]
               yy = y[:,columns]
               np_x = xx.ctypes.data_as(POINTER(c_double))
               np_y = yy.ctypes.data_as(POINTER(c_double))
               SpMV(byref(c_char(b"N")), byref(c_int(m)),data,indptr, indices, np_x, np_y ) 

        return y
else:
    def SpMV_viaMKL( A, x ):
        return A.dot(x)
