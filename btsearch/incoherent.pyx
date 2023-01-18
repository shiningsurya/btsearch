"""
Implements incoherent de-dispersion in cython

Two functions are written here:
    - Dedisperser_negative
    - Dedisperser_positive

positive/negative refer to the frequency offset
"""
# cython
cimport cython
# numpy 
import numpy  as np
cimport numpy as np

ctypedef fused datype:
    np.ndarray[np.float64_t, ndim=2]
    np.ndarray[np.float32_t, ndim=2]
    np.ndarray[np.uint8_t, ndim=2]

DDTYPE = np.uint64
ctypedef np.uint64_t DDTYPE_t

@cython.boundscheck (False)
@cython.wraparound  (False)
def Dedisperser_negative (datype fb, np.ndarray[DDTYPE_t, ndim=1] delays):
    if fb.shape[0] != delays.shape[0]:
        raise ValueError ("Incorrect delays size!")
    # constants 
    cdef Py_ssize_t nchans   = fb.shape[0]
    cdef Py_ssize_t nsamps   = fb.shape[1]
    cdef Py_ssize_t maxdelay = delays[nchans -1]
    if nsamps <= maxdelay:
        raise ValueError ("DM range too high!")
    cdef Py_ssize_t ddnsamps = nsamps - maxdelay
    cdef Py_ssize_t udx = 0, vdx = 0
    # output array
    cdef datype ret = np.zeros ([nchans, ddnsamps], dtype=fb.dtype)
    # algo
    # for isamp in range (ddnsamps):
        # for ichan in range (nchans):
            # tidx = isamp + delays[ichan]
            # ret[ichan, isamp] = fb[ichan, tidx]
    for ichan in range (nchans):
        udx = delays[ichan]
        vdx = udx + ddnsamps
        ret[ichan,...]   =  fb[ichan, udx:vdx]
    #
    return ret

@cython.boundscheck (False)
@cython.wraparound  (False)
def Dedisperser_positive (datype fb, np.ndarray[DDTYPE_t, ndim=1] delays):
    if fb.shape[0] != delays.shape[0]:
        raise ValueError ("Incorrect delays size!")
    # constants 
    cdef Py_ssize_t nchans   = fb.shape[0]
    cdef Py_ssize_t nsamps   = fb.shape[1]
    cdef Py_ssize_t maxdelay = delays[0]
    if nsamps <= maxdelay:
        raise ValueError ("DM range too high!")
    cdef Py_ssize_t ddnsamps = nsamps - maxdelay
    cdef Py_ssize_t tidx     = 0
    # output array
    cdef datype ret = np.zeros ([nchans, ddnsamps], dtype=fb.dtype)
    # algo
    # for isamp in range (ddnsamps):
        # for ichan in range (nchans):
            # tidx = isamp + delays[ichan]
            # ret[ichan, isamp] = fb[ichan, tidx]
    for ichan in range (nchans):
        udx = delays[ichan]
        vdx = udx + ddnsamps
        ret[ichan,...]   =  fb[ichan, udx:vdx]
    #
    return ret

