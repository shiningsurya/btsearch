
import numpy as np

from scipy.ndimage import label

__all__ = ["clusterer"]

def clusterer ( bt, threshold, ndm, istart, istop ):
    """
    Performs clustering on bt>threshold and identifies candidates
    takes care of the overlap 

    simply applies connected component on [bt>threshold]
    and identifies disjoint components
    and foreach component, measures the maximum value
    and returns
    values, dm_axis, time_axis of the 

    uses scipy.ndimage.label which is written in cython 
    and is just crazy fast
    """

    size = istop - istart
    ibt  = bt[...,istart:istop]
    jbt  = ibt >= threshold

    ##
    ## connected component labeling call
    labels, nf = label ( jbt )

    ## temp array
    zt   = np.zeros_like ( ibt )

    ##
    ## extracting components
    max_dm   = []
    max_sm   = []
    max_sn   = []
    ll       = 0
    for nnf in range (1, nf+1):
        idx   = np.where ( labels == nnf, ibt, zt ).argmax ()
        ii,jj = np.unravel_index ( idx, (ndm, size) )
        if ibt[ii,jj] > threshold:
            max_sn.append ( ibt[ii,jj] )
            max_sm.append ( jj )
            max_dm.append ( ii )
            ll += 1

    return  ll, max_dm, max_sm, max_sn 
