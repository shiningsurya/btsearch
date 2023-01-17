"""
Cython implementation of SAUF for CCL
"""
cimport cython
# cython: boundscheck=False
# cython: wraparound=False

import numpy  as np
cimport numpy as np

ctypedef fused datype:
    np.ndarray[np.float32_t, ndim=2]
    np.ndarray[np.float64_t, ndim=2]

cdef class UnionFind:
    """
    The datastructure at the core of SAUF
    """
    cdef public unsigned int  n
    cdef public np.ndarray    P
    def __init__ (self, unsigned int n):
        """ number of pixels """
        self.n     =  n
        self.P     = np.arange (self.n, dtype=np.uint16)

    cdef unsigned int findRoot (self, unsigned int i):
        """Find the root of the tree of node i"""
        # y is root
        cdef unsigned int y    = i
        while self.P [y] < y:
            y = self.P [y]
        # print (f" @ findRoot i={i:d} root={y:d}")
        return y

    cdef setRoot ( self, unsigned int i, unsigned int root ):
        """Make all nodes in the path of node i point to root"""
        cdef unsigned int j = 0
        # print (f" @ before setRoot P={self.P} i={i:d} root={root:d}")
        while self.P[i] < i:
            j          = self.P[i]
            self.P[i]  = root
            i          = j
        self.P[i]      = root
        # print (f" @ after  setRoot P={self.P} i={i:d} root={root:d}")

    def find (self, unsigned int i):
        """Find the root of tree of node i
        and compress the path in the process"""

        cdef unsigned int root       = self.findRoot ( i )
        self.setRoot ( i, root )
        return root

    def union (self, x, y):
        """ Unite the two trees containing nodes 
        x and y, and return the new root"""
        cdef unsigned int rootx      = self.findRoot ( x )
        cdef unsigned int rooty  = self.findRoot ( y )
        if rootx > rooty: rootx = rooty
        self.setRoot ( y, rootx )
        self.setRoot ( x, rootx )
        return rootx

    def flatten_old (self):
        """ Flatten the Union-Find tree and
        relabel the components"""
        for i in range (self.n):
            self.P[i] = self.P [ self.P [i] ]

    def flatten (self):
        """ FlattenL the Union-Find tree and
        relabel the components"""
        k = 1
        for i in range (1, self.n):
            if self.P[i] < i:
                self.P[i] = self.P [ self.P [i] ]
            else:
                self.P[i] = k
                k = k + 1

@cython.boundscheck(False)
@cython.wraparound(False)
def CCL ( datype bt, float threshold, unsigned int ndm, unsigned int istart, unsigned int istop ):
    """
    SAUF Connected Components algorithm
    https://sdm.lbl.gov/~kewu/ps/paa-final.pdf

    also see test_ccl
    """

    cdef unsigned int  size = istop - istart 
    cdef UnionFind    uf    = UnionFind ( ndm * size )
    cdef unsigned int largest_label  = 1
    cdef np.ndarray[np.uint32_t, ndim=2]  labels = np.zeros ( (ndm, size), dtype=np.uint32 )# + 324

    cdef unsigned int        i, j
    cdef unsigned int        a_label, b_label, c_label, d_label
    cdef bint                a_flag,  b_flag,  c_flag,  d_flag

    # i=0
    ###### istart=0
    if bt[0, istart] >= threshold:
        labels[0, 0]   = largest_label
        largest_label  = largest_label + 1
    # i=0,j={1..size}
    ###### range (1, size)
    for j in range ( 1, size ):
    # for j in range ( istart+1, istop ):
        if bt[ 0, j+istart ] >= threshold:
            ## d
            if bt[0, j-1+istart] >= threshold:
                ## copy (d)
                labels[0, j] = labels[0, j-1]
            else:
                ## newlabel
                labels[0,j]    = largest_label
                largest_label  = largest_label + 1
    # main loop
    for i in range (1, ndm):
        ## j=0
        # if bt [i, 0] >= threshold:
        if bt [i, istart] >= threshold:
            # b_flag      = bt[i-1, 0] >= threshold
            # c_flag      = bt[i-1, 1] >= threshold
            b_flag      = bt[i-1, istart] >= threshold
            c_flag      = bt[i-1, istart+1] >= threshold
            b_label     = labels[i-1,0]
            c_label     = labels[i-1,1]
            if b_flag:
                # copy (b)
                labels[i,0] = b_label
            elif c_flag:
                # copy (c)
                labels[i,0] = c_label
            else:
                # newlabel
                labels[i,0]   = largest_label
                largest_label = largest_label + 1
        ## loop
        for j in range ( 1, size-1 ):
        # for j in range ( istart+1, istop-1 ):
            ## if occupied
            if  bt [ i, j+istart ] >= threshold:  
                a_flag      = bt [i-1, j-1+istart] >= threshold
                b_flag      = bt [i-1, j+istart]   >= threshold
                c_flag      = bt [i-1, j+1+istart] >= threshold
                d_flag      = bt [i, j-1+istart]   >= threshold
                ### get labels
                a_label     = labels[i-1,j-1]
                b_label     = labels[i-1,j]
                c_label     = labels[i-1,j+1]
                d_label     = labels[i,j-1]
                # "b"
                if b_flag:
                    # copy (b)
                    labels[i, j] = b_label
                else:
                    # c
                    if c_flag:
                        # "a"
                        if a_flag:
                            # copy (c, a)
                            labels[i,j]  = uf.union ( c_label, a_label )
                        else:
                            # "d"    
                            if d_flag:
                                # copy (c,d)
                                labels[i,j]  = uf.union ( c_label, d_label )
                            else:
                                # copy (c)
                                labels[i, j] = c_label
                    else:
                        # "a"
                        if a_flag:
                            # copy (a)
                            labels[i, j] = a_label
                        else:
                            # "d"
                            if d_flag:
                                # copy (d)
                                labels[i, j] = d_label
                            else:
                                # new label
                                labels[i,j]   = largest_label
                                largest_label = largest_label + 1
        ### j=size-1 here
        # if bt [ i, size-1 ] >= threshold:
        if bt [ i, istop-1 ] >= threshold:
            # a_flag      = bt [i-1, size-2] >= threshold
            # b_flag      = bt [i-1, size-1] >= threshold
            # d_flag      = bt [i, size-2]   >= threshold
            a_flag      = bt [i-1, istop-2] >= threshold
            b_flag      = bt [i-1, istop-1] >= threshold
            d_flag      = bt [i, istop-2]   >= threshold
            ### get labels
            a_label     = labels[i-1, size-2]
            b_label     = labels[i-1, size-1]
            d_label     = labels[i, size-2]
            if b_flag:
                # copy (b)
                labels[i, size-1] = b_label
            elif a_flag:
                # copy (a)
                labels[i, size-1] = a_label
            elif d_flag:
                labels[i, size-1] = d_label
            else:
                # newlabel
                labels[i,size-1]   = largest_label
                largest_label      = largest_label + 1

    ## analysis
    uf.flatten ()

    ## get max
    cdef np.ndarray[np.float32_t, ndim=1] max_sn  = np.zeros ((largest_label,), dtype=np.float32)
    cdef np.ndarray[np.uint32_t, ndim=1]  max_sm  = np.zeros ((largest_label,), dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1]  max_dm  = np.zeros ((largest_label,), dtype=np.uint32)
    cdef unsigned int max_lab = 0

    ## relabeling
    for i in range ( ndm ):
        for j in range ( size ):
            if labels[i,j] > 0:
                idx          = uf.P [ labels[i,j] ]
                ## reassignment
                labels[i,j]  = idx
                ## updating
                if idx > max_lab: max_lab = idx
                if max_sn[idx] < bt [i, j+istart]:
                    max_sn[idx] = bt[i, j+istart]
                    max_dm[idx] = i
                    max_sm[idx] = j

    # print ( max_lab, max_sn, max_sm, max_dm )
    # return largest_label, labels
    max_lab = max_lab + 1
    return  max_lab-1, max_dm[1:max_lab], max_sm[1:max_lab], max_sn[1:max_lab] 
    # return max_lab-1, max_dm[1:max_lab], max_sm[1:max_lab], max_sn[1:max_lab]
    # return max_lab, labels
















