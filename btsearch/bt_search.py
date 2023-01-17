"""
BT search

There is a 10ms improvement when pre-doing the FFTs
than using the fftconvolve

the time it took to do that is too much to justify it
> but whatever

the CCL is still a bottleneck
over 90% of the time is spent in doing CCL
"""

import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve 

import tqdm

# import pyximport; pyximport.install (setup_args={"include_dirs":np.get_include()})
# import sauf
from test_ccl import clusterer_label

from candidates import Candidates

from convolution import ConvolutionEngine

def get_args():
    import argparse as arp
    ap = arp.ArgumentParser (prog='btsearch',description='Searches for bowtie using templates',)
    add = ap.add_argument
    add ('bowtie',help='Bowtie memmap array')
    add ('--dir', help='Directory to save the candidates', dest='dir', default='candidates')
    add ('-t','--template', help='Templates npz file', dest='temps', required=True)
    add ('-T','--threshold', help='S/N threshold', dest='sn',  type=float, default=6)
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    return ap.parse_args()

if __name__ == "__main__":
    ################################
    args      = get_args()
    ### make directory if it does not exist
    if not os.path.isdir (args.dir):
        os.mkdir (args.dir)
    ###############################
    ## read params
    filpath   = args.bowtie
    tempath   = args.temps
    filname,_ = os.path.splitext (filpath)
    infpath   = filname + ".json"
    thres     = args.sn
    ## read templates
    tempz     = np.load ( tempath )
    # tempz    = np.load ('test_paflike_boxcar_256.npz')
    # tempz    = np.load ('test_paflike_bowtie_256.npz')
    temps     = tempz['templates']
    widths    = tempz['widths']
    ntemps    = widths.shape[0]
    ## read bowtie and info
    #### first read info
    with open (infpath, 'r') as f:
        inf   = json.load ( f )
    bter      = np.memmap ( filpath, dtype=np.float32, mode='r', shape=(inf['ndm'], inf['nsamps']) )
    tsamp     = inf['tsamp']
    # bter     = np.memmap ('2020-12-09-16:37:40_dm57.bt.npy', dtype=np.float32, mode='r', shape=(256, 1327104))
    ndm, nsamps = bter.shape
    ###############################
    ## do sanity check
    # if ( inf['fch1'] != tempz['fch1'] )  or ( inf['foff'] != tempz['foff'] ) or ( inf['nchans'] != tempz['nchans'] ):
        # raise ValueError ("frequency axis not consistent ...")
    # if ( inf['tsamp'] != tempz['tsamp'] ):
        # raise ValueError ("time axis not consistent ...")
    ###############################

    ## geometry
    gulp      = 8 * ndm
    overlap   = 2  * ndm
    gov       = gulp + overlap
    halflap   = overlap // 2
    gulplap   = halflap + gulp
    niter     = int ( nsamps / gulp )

    ## setting up convolution engine
    ce        = ConvolutionEngine ( (ndm, gov), (ndm, ndm) )

    print (" manually setting gulp and overlap here = ", gulp, overlap)

    # adfa
    ## ignore the last samples
    # it        = tqdm.tqdm ( range(niter), desc='Chunk', unit='bt' )
    iterator   = tqdm.tqdm ( range(1, niter-1), desc='Chunk', unit='bt' )
    # it        = tqdm.tqdm ( range(100), desc='Chunk', unit='bt' )
    # iterator   = tqdm.tqdm ( [91,], desc='Chunk', unit='bt' )
    # it        = tqdm.tqdm ( [91,], desc='Chunk', unit='bt' )
    # it        = tqdm.tqdm ( [100,], desc='Chunk', unit='bt' )
    # it        = tqdm.tqdm ( [135], desc='Chunk', unit='bt' )
    # it        = tqdm.tqdm ( [191], desc='Chunk', unit='bt' )
    ##
    ## prepare temp ffts
    template_ffts = np.zeros ((ntemps, ce.shape0_fft, ce.shape1_rfft), dtype=np.complex128)
    for i in range (ntemps):
        """this division by ndm is required"""
        # the division by ndm is required
        ## prolly move this step to template creator?
        template_ffts[i]  = ce.do_fft ( temps[i] / ndm )

    #
    cands     = Candidates ( ntemps, tsamp )

    #### diagnostics
    """
    fig       = plt.figure ()
    (ax1,ax2, ax3) = fig.subplots (3, 1, sharex=True, sharey=True)
    # (ax1,ax2) = fig.subplots (2, 1, sharex=True, sharey=True)
    ax3.set_xlabel ('Time / index')
    ax1.set_ylabel ('DM / index')
    ax2.set_ylabel ('DM / index')
    # ax3.set_ylabel ('DM / index')
    shower   = lambda x,ax : ax.imshow (x, aspect='auto', cmap='plasma', interpolation='none', origin='lower')
    # shower   = lambda x,ax : ax.plot (x[128])
    """
    ####################################################################
    ## loop
    ####################################################################
    ## loop variables
    #### ii,jj to slice from bter
    #### uu,vv to slice after convolution
    ii = 0; jj = gulp + overlap
    uu = 0; vv = gulp
    ########## action
    bts_fft   = ce.do_fft ( bter [ ..., ii:jj ] )
    for i,iw in enumerate (widths):
        btc   = ce.do_ifft ( bts_fft * template_ffts[i] )
        ll, max_dm, max_sm, max_sn = clusterer_label ( btc, thres, ndm, 0, gulp )
        if ll > 0:
            cands ( iw, sn=max_sn, sm=max_sm, dm=max_dm, sm_offset=uu )
    cands.aggregate ()
    ########## action
    ################## time
    ## loop
    for  it in iterator:
        ## get the indices
        uu    = it * gulp
        vv    = uu + gulp
        ii    = uu - halflap
        jj    = vv + halflap
        ########## action
        bts_fft   = ce.do_fft ( bter [ ..., ii:jj ] )
        for i,iw in enumerate (widths):
            btc   = ce.do_ifft ( bts_fft * template_ffts[i] )
            ll, max_dm, max_sm, max_sn = clusterer_label ( btc, thres, ndm, halflap, gulplap )
            if ll > 0:
                cands ( iw, sn=max_sn, sm=max_sm, dm=max_dm, sm_offset=uu )
        cands.aggregate ()
        ########## action
        """
        cands.print ()
        shower ( bts, ax1 )
        shower ( btc, ax2 )
        # ll[ll<0] = np.nan
        # shower ( ll, ax3 )
        plt.show ()
        adf
        """
    ## last read
    #### always left with half overlap
    uu = ( niter - 1 ) * gulp
    vv = nsamps
    ii = uu - halflap
    jj = nsamps
    ########## action
    bts_fft   = ce.do_fft ( bter [ ..., ii:jj ] )
    for i,iw in enumerate (widths):
        btc   = ce.do_ifft ( bts_fft * template_ffts[i] )
        ll, max_dm, max_sm, max_sn = clusterer_label ( btc, thres, ndm, halflap, vv - uu + halflap)
        if ll > 0:
            cands ( iw, sn=max_sn, sm=max_sm, dm=max_dm, sm_offset=uu )
    cands.aggregate ()
    ########## action
    ####################################################################
    ## loop
    ####################################################################
    ########## action
    cands.save ("test_bt.cands")


