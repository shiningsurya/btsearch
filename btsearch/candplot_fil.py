"""
candidate plot


|  unzoomed      |   zoomed     |
|----------------|--------------|
| Time profile   | Time profile |
| Frequency      | Frequency    |
| Bowtie         | Bowtie       |
"""
import os
import sys

import tqdm
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.colors as mc

import astropy.time as at
import astropy.units as au

# import sigpyproc as spp
from sigpyproc.readers import FilReader
from skimage.measure import block_reduce

from btdd import BTDD


STRING="S/N: {0:3.2f} DM: {1:3.3f}pc/cc\nTime: {2:3.2f}s DS: {3:d} dfac: {4:d} "

OFILE="{tmjd:15.10f}_dm{dm:04.3f}_sn{sn:04.1f}_ftop{freq:03.0f}_ds{ds:03d}_l{ln:04d}.{{ext}}"

########
def get_v ( d, m=1, M=3 ):
    med  = np.median ( d )
    std  = np.std ( d )
    vmin = med - m*std
    vmax = med + M*std
    return vmin, vmax

def get_args():
    import argparse as arp
    ap = arp.ArgumentParser (prog='candplotter',description='Plots candidates',)
    add = ap.add_argument
    add ('file',help='Filterbank file')
    add ('--sps',help='PRESTO Singlepulse file', required=True, dest='sps')
    add ('--dir', help='Directory to store plots', dest='dir', default='candplots')
    add ('-t,--thres', help='Threshold on sigma', dest='thres',default=5, type=int) 
    add ('-f,--fs', help='Frequency downsample', dest='fs',default=4, type=int) 
    add ('-T,--thresmax', help='Max threshold on sigma', dest='thresmax',default=None, type=int) 
    add ('-W,--max-width', help='Max threshold on width', dest='maxw',default=500, type=int) 
    add ('-n','--ndm', help='Number of DM trials in bowtie', dest='ndm', default=256, type=int)
    add ('-p','--padding', help='Padding, before burst (in candidate width units)', default=8, type=float, dest='padding')
    add ('-w','--width', help='Width in plots (in candidate width units)', default=20, type=float, dest='width')
    return ap.parse_args()
		
########
if __name__ == "__main__":
    ################################
    args      = get_args()
    ### make directory if it does not exist
    if not os.path.isdir (args.dir):
        os.mkdir (args.dir)
    ###############################
    ### argument checks
    if args.width < args.padding/2:
        raise ValueError(" Width should be atleast twice the padding to actually see the burst")
    ###############################
    ###  read sps file
    sps       = pd.read_csv (args.sps, delim_whitespace=True, comment='#', names=['dm','sigma','time','sample','dfac'])
    ### selection logic
    smask     = (sps.sigma >= args.thres) & (sps.dfac <= args.maxw)
    if args.thresmax:
        smask = smask & (sps.sigma < args.thresmax)
    if smask.sum() <= 0:
        sys.exit (0)
    ###### prepare the progress bar
    it  = tqdm.tqdm (sps.index[smask], unit='cand', desc='SPS')

    ### read filterbank
    fil       = FilReader (args.file)
    fh        = fil.header
    mjd       = fh.tstart
    tsamp     = fh.tsamp
    tsamp_ms  = tsamp * 1000.
    nchans    = fh.nchans
    fch1      = fh.fch1
    foff      = fh.foff

    ### setup 
    mjd       = at.Time (mjd, format='mjd')
    hdm       = args.ndm // 2
    btdd      = BTDD ( fch1, foff, nchans, tsamp, args.ndm, False)
    freqs     = btdd.f_axis.copy ()
    max_freq  = btdd.max_freq
    pfreqs    = block_reduce ( freqs, (args.fs,), func=np.mean, cval=np.nan )

    ### figure setup
    ddimdict    = {'cmap':'plasma', 'aspect':'auto', 'interpolation':'none', 'origin':'lower'}
    btimdict    = {'cmap':'plasma', 'aspect':'auto', 'interpolation':'none', 'origin':'lower'}
    sdict     = {'where':'mid', 'lw':1.5, 'c':'k'}

    figdict   = {'figsize':(6,10)}
    fig       = plt.figure (**figdict)
    [ [axpp, dxpp], [axdd, dxdd], [axbt, dxbt] ]      = fig.subplots ( 3, 2, sharex='col', sharey='row', gridspec_kw = dict(height_ratios=[0.2, 0.4, 0.4], hspace=0.05, wspace=0.1))

    for i in it:
        ## get parameters
        dm    = sps.dm[i]
        ds    = sps.dfac[i]
        sn    = sps.sigma[i]
        pt    = sps.time[i]
        tt    = mjd + (pt * au.second)
        ds_time = ds * tsamp * 1E3

        ## prepare outfile
        ofile = OFILE.format (tmjd = tt.mjd, dm = dm, sn = sn, freq = max_freq, ln=i, ds=ds).format(ext='png')

        ## prepare btdd
        btdd ( dm, 1 )

        ## slice logic
        start_sample  = int ( ( pt / tsamp ) - ( args.padding * ds ) )
        width_sample  = int (args.width * ds)
        take_sample   = width_sample + btdd.max_d
        ## slice
        fb            = fil.read_block ( start_sample, take_sample )
        ## get bt, dd
        bt,dd         = btdd.work ( fb, width_sample )

        ## downsample logic
        dfac          = max ( ds//2, 1 )
        btds          = block_reduce ( bt, (1, dfac), func=np.mean, cval=np.nan )
        ddds          = block_reduce ( dd, (args.fs, dfac), func=np.mean, cval=np.nan )

        ## time axis
        times         = np.linspace ( 0.0, args.width, bt.shape[1] )   - args.padding
        ptimes        = np.linspace ( 0.0, args.width, btds.shape[1] ) - args.padding

        times         *= (tsamp_ms * ds)
        ptimes        *= (tsamp_ms * ds)

        ###### axes
        axpp.step ( times, bt[btdd.hdm], **sdict )

        vmin, vmax = get_v ( dd )
        axdd.imshow ( dd, extent=[ times[0], times[-1], freqs[0], freqs[-1]], 
            vmin=vmin, vmax=vmax, **ddimdict)

        vmin, vmax = get_v ( bt )
        axbt.imshow ( bt, extent=[ times[0], times[-1], btdd.dm_axis[0], btdd.dm_axis[-1]], 
            vmin=vmin, vmax=vmax, **btimdict)

        dxpp.step ( ptimes, btds[btdd.hdm], **sdict )

        vmin, vmax = get_v ( ddds )
        dxdd.imshow ( ddds, extent=[ ptimes[0], ptimes[-1], pfreqs[0], pfreqs[-1]], 
            # vmin=vmin, vmax=vmax, 
            **ddimdict)

        vmin, vmax = get_v ( btds )
        dxbt.imshow ( btds, extent=[ ptimes[0], ptimes[-1], btdd.dm_axis[0], btdd.dm_axis[-1]], 
            # vmin=vmin, vmax=vmax, 
            **btimdict)

        ## labels 
        axpp.set_yticklabels ([])
        axpp.set_ylabel ('Intensity')
        axdd.set_ylabel ('Freq / MHz')
        axbt.set_ylabel ('DM / pc cc')
        axbt.set_xlabel ('Time / ms')
        dxbt.set_xlabel ('Time / ms')

        ## lines
        axpp.axvline (-0.5*ds_time, c='r',linestyle=':', alpha=0.6)
        axpp.axvline (0.5*ds_time, c='r',linestyle=':', alpha=0.6)
        dxpp.axvline (-0.5*ds_time, c='r',linestyle=':', alpha=0.6)
        dxpp.axvline (0.5*ds_time, c='r',linestyle=':', alpha=0.6)



        ## title and save
        fig.suptitle ( STRING.format(sn, dm, pt, ds, dfac), x=0.5, y=0.90, ha='center', va='bottom' )
        plt.show ()
        adfa
        # fig.savefig (os.path.join (args.dir, ofile), dpi=72, bbox_inches='tight')

        ## prepare for next
        axpp.cla(); axdd.cla(); axbt.cla();
        dxpp.cla(); dxdd.cla(); dxbt.cla();
