"""
Prepares bowtie array as a memory mapped file

"""
import os
import sys
import json

import datetime

import numpy as np

from btdd import BTDD

import pyximport; pyximport.install (setup_args={"include_dirs":np.get_include()})
import misc

# import sigpyproc as spp
try:
    from sigpyproc.readers import FilReader
except ImportError:
    from sigpyproc import FilReader

OFILE="{fname}_dm{dm:04.3f}_n{ndm:03d}_bt.{{ext}}"

def get_args():
    import argparse as arp
    ap = arp.ArgumentParser (prog='bt',description='Prepares bowtie array',)
    add = ap.add_argument
    add ('file',help='Filterbank file')
    add ('--dir', help='Directory to save the array', dest='dir', default='btarray')
    add ('-n','--ndm', help='Number of DM trials in bowtie', dest='ndm', default=256, type=int)
    add ('-g','--gulp', help='Gulp size', dest='gulp', default=8192, type=int)
    add ('-t','--detrend', help='Detrend size (4x ndm)', dest='detrend', default=1024, type=int)
    add ('-d','--dm', help='DM', required=True, type=float, dest='dm')
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
    filpath   = args.file
    filname,_   = os.path.splitext (os.path.basename (filpath))
    dm        = args.dm
    ndm       = args.ndm
    gulp_size = args.gulp
    detrend   = args.detrend
    if detrend < (ndm//4):
        print (f" detrend should be atleast 4x ndm")
    detrend_overlap = detrend//2
    if detrend_overlap < ndm//2:
        print (f" detrend_overlap should be atleast 2x ndm")
    ## prepare filenames
    OFI       = os.path.join ( args.dir, OFILE.format (fname=filname, dm=dm, ndm=ndm) )
    OBT       = OFI.format(ext='npy')
    OJS       = OFI.format(ext='json')
    ## read fil
    fil       = FilReader ( filpath )
    fh        = fil.header
    ###############################
    ## geometry
    nsamps    = fh.nsamples
    nchans    = fh.nchans
    foff      = fh.foff
    fch1      = fh.fch1
    tsamp     = fh.tsamp
    if args.v:
        print (f" BT array would require {nsamps*ndm*4/1E9:.2f} GB")
    ## setup 
    ##### setup btdd
    btdd      = BTDD ( fch1, foff, nchans, tsamp, ndm, False)
    btdd ( dm, 1 )
    max_delay = btdd.max_d
    ##### setup memmap 
    m_bt      = np.memmap ( OBT, dtype=np.float32, mode='w+', shape=(ndm, nsamps) )
    ##### setup read_plan
    gulp      = np.zeros ((nchans, gulp_size+max_delay), dtype=np.float32)
    gchan            = np.zeros ((nchans,1), dtype=np.float32)
    rp        = fil.readPlan ( gulp_size+max_delay, start=0, skipback=max_delay, verbose=args.v )
    ## prepare info
    info             = dict ()
    info['fch1']     = fch1
    info['foff']     = foff
    info['nchans']   = nchans
    info['tsamp']    = tsamp
    info['ndm']      = ndm
    info['nsamps']   = nsamps
    info['dm']       = dm
    info['dm1']      = float ( btdd.dm_axis[0] )
    info['dmoff']    = float ( btdd.dm_axis[1] - btdd.dm_axis[0] )
    info['maxd']     = max_delay
    info['creation_datetime'] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # detre            = np.zeros ((ndm, gulp_size//detrend, 1), dtype=np.float32)
    # detrend_overlap  = detrend//2

    if args.v:
        print (info)
    ###############################
    ##### work loop
    ii, jj    = 0,0
    for nread, _, data in rp:
        # read
        gulp[:,:nread] = data.reshape ( (nread, nchans) ).T
        if nread < gulp_size:
            gulp[:, nread:] = 0
        # do we do per-channel whitening?
        ### disabling it for now
        misc.Whiten ( gulp )
        # gchan[...,0]   = gulp.mean (1)
        # gulp           -= gchan
        # gchan[...,0]   = gulp.std (1)
        # gulp           = np.divide ( gulp, gchan, where=gchan!=0, out=gulp )
        # asking bc we anyway convert bt into S/N units
        # call btdd
        bt      = btdd.work_bt ( gulp, gulp_size )
        # convert into S/N units
        """
        detrending should happen on the `detrend` length scale

        detrending overlap should be atleast twice ndm
        detrending should be atleast twice detrending overlap
        """
        obt     = misc.Detrend ( bt, detrend, detrend_overlap)
        # detre[...,0]   = btv.mean ( 2 )
        # btv     -= detre

        ## bt      -= uniform_filter1d ( bt, detrend, axis=1 )

        ## btv     = bt.reshape ((ndm, gulp_size//detrend, detrend))
        ## detre[...,0]   = btv.std ( 2 )
        ## btv     /= detre
        
        # bt      -= uniform_filter1d ( bt, detrend, axis=1 )
        # bt      /= bt.std ( 1 )[:, np.newaxis]
        # write
        jj      = ii + gulp_size
        # print (f" ii:jj = {ii:d}, {jj:d} | btshape = {bt.shape} | nread = {nread:d}")
        m_bt[...,ii:jj] = obt[:ndm,...]
        ii      = jj
    #####
    with open (OJS, 'w') as f:
        json.dump ( info, f )
    #############################
    if args.v:
        print ('Done')
