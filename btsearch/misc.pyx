"""
misc functions for data massaging
"""
cimport cython
# cython: linetrace=True
# cython: profile=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# numpy 
import numpy  as np
cimport numpy as np

ctypedef fused datype:
    np.ndarray[np.float32_t, ndim=2]
    np.ndarray[np.float64_t, ndim=2]

@cython.boundscheck (False)
@cython.wraparound  (False)
def Whiten(datype fb):
    """
    whiten the data block per axis=1

    write in place
    """
    # constants 
    cdef Py_ssize_t nchans   = fb.shape[0]
    cdef Py_ssize_t nsamps   = fb.shape[1]
    # arrays
    cdef np.ndarray[np.float64_t, ndim=1] rmean        = np.zeros ([nchans,], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] rstd         = np.zeros ([nchans,], dtype=np.float64)

    # action

    rmean[...]     = fb.mean (1)
    rstd[...]      = fb.std  (1)

    for i in range (nchans):
        if rstd[i] != 0.0:
            fb[i,...] = (fb[i,...] - rmean[i] ) / rstd[i]

@cython.boundscheck (False)
@cython.wraparound  (False)
@cython.profile (True)
@cython.linetrace (True)
@cython.cdivision (True)
def Detrend_ndm (datype bt, unsigned int gulp, unsigned int overlap):
    """
    detrend per axis=1

    preserves some overlap

    overlap should be atleast 20% ?

    detrend on gulp+overlap
    take only gulp

    this is horrendeously slow
    """
    # constants 
    cdef Py_ssize_t ndm      = bt.shape[0]
    cdef Py_ssize_t nsamps   = bt.shape[1]
    # cdef Py_ssize_t gulp     = gulp
    # cdef Py_ssize_t overlap  = overlap
    cdef Py_ssize_t gov      = gulp + overlap
    # arrays
    cdef datype ret          = np.zeros ([ndm, nsamps], dtype=bt.dtype)

    #### std divide
    cdef Py_ssize_t ichan, isamp
    cdef double rx, rxx, std
    cdef Py_ssize_t niter    = int ( (nsamps - overlap) / gulp )

    ## for every dm time series
    for idm in range ( ndm ):
        ## for every gulp+overlap
        for it in range ( niter ):
            ## get the indices
            ii    = it*gulp
            jj    = ii + gov
            # print ( "ii, jj=", ii, jj )
            ## run accumulation 
            rx   = 0.0
            rxx  = 0.0
            std  = 0.0
            for iijj  in range (ii, jj):
                rx   +=  bt[idm, iijj]
                rxx  +=  bt[idm, iijj]*bt[idm, iijj]
            ## corrections
            rx   /= gov
            rxx  /= gov
            std  = ( rxx - (rx*rx) ) ** 0.5
            if std == 0.0: std = 1.0
            ## write into output array
            for iij in range (ii, ii+gulp):
                ret[idm, iij] = ( bt[idm, iij] - rx ) / std
    return ret


@cython.boundscheck (False)
@cython.wraparound  (False)
@cython.profile (True)
@cython.linetrace (True)
@cython.cdivision (True)
def Detrend (datype bt, unsigned int gulp, unsigned int overlap):
    """
    detrend per axis=1

    preserves some overlap

    overlap should be atleast 20% ?

    detrend on gulp+overlap
    take only gulp

    first read:
        gulp+overlap

    """
    # constants 
    cdef Py_ssize_t ndm      = bt.shape[0]
    cdef Py_ssize_t nsamps   = bt.shape[1]
    # cdef Py_ssize_t gulp     = gulp
    cdef Py_ssize_t halflap  = overlap / 2
    # arrays
    cdef datype ret          = np.zeros ([ndm, nsamps], dtype=bt.dtype)
    cdef datype rmean        = np.zeros ([ndm, 1], dtype=bt.dtype)
    cdef datype rstd         = np.zeros ([ndm, 1], dtype=bt.dtype)

    #### std divide
    cdef Py_ssize_t ii, jj, uu, vv
    cdef Py_ssize_t niter    = int ( nsamps / gulp )


    #### vectorize over axis=0
    #### the likelihood of std-dev to be zero is 
    #### extremely low because this is de-dispersed time series

    ## first read: gulp+overlap
    ii = 0; jj = gulp + overlap
    uu = 0; vv = gulp
    ### action
    rmean[...,0]     = bt[...,ii:jj].mean (1)
    rstd[...,0]      = bt[...,ii:jj].std  (1)
    ret[..., uu:vv]  = ( bt[..., uu:vv] - rmean ) / rstd
    # cdef unsigned idx = 0
    # print (" @ {:d} workon = ({:d}...{:d}) thentake = ({:d}...{:d})".format(idx, ii, jj, uu, vv)); idx += 1
    ## loop 
    #### last iteration has only partial read
    for it in range ( 1, niter - 1 ):
        ## get the indices
        uu    = it * gulp
        vv    = uu + gulp
        ii    = uu - halflap
        jj    = vv + halflap
        ### action
        rmean[...,0] = bt[..., ii:jj].mean(1)
        rstd[...,0]  = bt[..., ii:jj].std(1)
        ret[..., uu:vv]     = ( bt[..., uu:vv] - rmean ) / rstd
        # print (" @ {:d} workon = ({:d}...{:d}) thentake = ({:d}...{:d})".format(idx, ii, jj, uu, vv)); idx += 1
    ## last read
    #### always left with half overlap
    uu = ( niter - 1 ) * gulp
    vv = nsamps
    ii = uu - halflap
    jj = nsamps
    ### action
    rmean[...,0]     = bt[...,ii:jj].mean (1)
    rstd[...,0]      = bt[...,ii:jj].std  (1)
    ret[..., uu:vv]  = ( bt[..., uu:vv] - rmean ) / rstd
    # print (" @ {:d} workon = ({:d}...{:d}) thentake = ({:d}...{:d})".format(idx, ii, jj, uu, vv)); idx += 1

    ## done
    return ret

