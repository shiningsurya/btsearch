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

import tqdm

import numpy as np

## btsearch imports
from ccl import clusterer
from candidates import Candidates
from convolution import ConvolutionEngine

def get_args():
    import argparse as arp
    ap = arp.ArgumentParser (prog='btsearch',description='Searches for bowtie using templates',)
    add = ap.add_argument
    add ('bowtie',help='Bowtie memmap array')
    add ('--dir', help='Directory to save the candidates', dest='dir', default='./')
    add ('-t','--template', help='Templates npz file', dest='temps', required=True)
    add ('-T','--threshold', help='S/N threshold', dest='sn',  type=float, default=6)
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('--gulp', help='Gulp size in multiples of ndm', default=8, type=int, dest='g')
    add ('--overlap', help='Overlap size in multiples of ndm. (must be less than gulp)', default=2, type=int, dest='o')
    return ap.parse_args()

if __name__ == "__main__":
    ################################
    args      = get_args()
    if args.o >= args.g:
        raise ValueError ("Overlap cannot be more than gulp")
    ### make directory if it does not exist
    if not os.path.isdir (args.dir):
        os.mkdir (args.dir)
    ###############################
    ## read params
    filpath   = args.bowtie
    tempath   = args.temps
    filname,_ = os.path.splitext (filpath)
    infpath   = filname + ".json"
    candpath  = filname + ".cands"
    thres     = args.sn
    candpath  = filname + ".cands"
    ## read bowtie and info
    #### first read info
    with open (infpath, 'r') as f:
        inf   = json.load ( f )
    bter      = np.memmap ( filpath, dtype=np.float32, mode='r', shape=(inf['ndm'], inf['nsamps']) )
    tsamp     = inf['tsamp']
    fch1      = inf['fch1']
    foff      = inf['foff']
    nchans    = inf['nchans']
    ndm, nsamps = bter.shape
    dm_axis   = inf['dm1'] + (np.arange (ndm, dtype=np.float32)*inf['dmoff'])
    ###############################
    ## do sanity check
    # if ( inf['fch1'] != tempz['fch1'] )  or ( inf['foff'] != tempz['foff'] ) or ( inf['nchans'] != tempz['nchans'] ):
        # raise ValueError ("frequency axis not consistent ...")
    # if ( inf['tsamp'] != tempz['tsamp'] ):
        # raise ValueError ("time axis not consistent ...")
    ###############################
    ## read templates
    """
    ## or make templates
    if args.v:
        print (" Building templates ... ", end='')
    hdm       = ndm // 2
    temper    = BowtieTemplate ( tsamp=tsamp, nchans=nchans, fch1=fch1, foff=foff, ndm=ndm )
    ntemps    = int ( np.log2 ( hdm ) )
    temps     = np.zeros ((n_temps+1, ndm, ndm))

    # Powers of two?
    widths    = np.power (2, np.arange(n_temps+1))
    ### call
    for idx, iwidth in enumerate (widths):
        temps[idx] = temper (iwidth, amplitude=1.0)
    """
    # old code
    tempz     = np.load ( tempath )
    temps     = tempz['templates']
    widths    = tempz['widths']
    ntemps    = widths.shape[0]
    if args.v:
        print (" done")

    if args.v:
        print (f" Precomputing template FFTs ... ", end='')
    ##
    ## prepare temp ffts
    template_ffts = np.zeros ((ntemps, ce.shape0_fft, ce.shape1_rfft), dtype=np.complex128)
    for i in range (ntemps):
        """this division by ndm is required"""
        # the division by ndm is required
        ## prolly move this step to template creator?
        template_ffts[i]  = ce.do_fft ( temps[i] / ndm )

    if args.v:
        print (" done")

    ## geometry
    gulp      = args.g * ndm
    overlap   = args.o  * ndm
    gov       = gulp + overlap
    halflap   = overlap // 2
    gulplap   = halflap + gulp
    niter     = int ( nsamps / gulp )

    ## setting up convolution engine
    ce        = ConvolutionEngine ( (ndm, gov), (ndm, ndm) )
    ## setting  up candidate sink
    cands     = Candidates ( ntemps, tsamp, dm_axis )
    ####################################################################
    ## loop
    iterator   = tqdm.tqdm ( range(1, niter-1), desc='Chunk', unit='bt' )
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
        ll, max_dm, max_sm, max_sn = clusterer ( btc, thres, ndm, 0, gulp )
        if ll > 0:
            cands ( iw, sn=max_sn, sm=max_sm, dm=max_dm, sm_offset=uu )
    cands.aggregate ()
    ########## action
    ################## time
    ## loop
    for  it in iterator:
        ## decorate my progressbar
        iterator.set_description (f"Found {cands.total_count:d} cands")
        ## get the indices
        uu    = it * gulp
        vv    = uu + gulp
        ii    = uu - halflap
        jj    = vv + halflap
        ########## action
        bts_fft   = ce.do_fft ( bter [ ..., ii:jj ] )
        for i,iw in enumerate (widths):
            btc   = ce.do_ifft ( bts_fft * template_ffts[i] )
            ll, max_dm, max_sm, max_sn = clusterer ( btc, thres, ndm, halflap, gulplap )
            if ll > 0:
                cands ( iw, sn=max_sn, sm=max_sm, dm=max_dm, sm_offset=uu )
        cands.aggregate ()
        ########## action
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
        ll, max_dm, max_sm, max_sn = clusterer ( btc, thres, ndm, halflap, vv - uu + halflap)
        if ll > 0:
            cands ( iw, sn=max_sn, sm=max_sm, dm=max_dm, sm_offset=uu )
    cands.aggregate ()
    ########## action
    ####################################################################
    ## loop
    ####################################################################
    ########## action
    if args.v:
        print (f" Saving {cands.total_count:d} candidates to {candpath}")
    cands.save (candpath)


