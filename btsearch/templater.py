"""
Prepares bowtie templates

bowtie search has limitation that the width of the pulse < ndm/2
"""
import os

import datetime

import numpy as np

# import sigpyproc as spp
from sigpyproc.readers import FilReader

import pyximport; pyximport.install (setup_args={"include_dirs":np.get_include()})

import fdmt as t_fdmt

OFILE="{fname}_n{ndm:03d}_template.{ext}"

def NormalizeTemplate (itemp):
    """
    Normalizes the template energy so that 
    post convolution, S/N unit of the input is preserved. 

    This is fixed by some trial and error in `investigate_btemp.py`

    The template is divided by the sqrt of the total energy in the template.

    Energy is defined as mean ( abs ( FFT ( . )^2 ) )
    or  SQRT ( SUM ( ABS ( POWER ( ., 2 ) ) , axis=1) )
    per every time series in the template

    earlier definition was not good

    hmm, what if i normalize energy over every time series

    do we really care about template energies?

    """
    # ten = np.sqrt (np.abs (np.fft.fft2 ( itemp )).sum ())
    # ten = np.sqrt ( np.mean ( np.abs ( np.fft.fft2 ( itemp ) ) ** 2 ) )
    # ten = np.sqrt ( np.sum ( np.abs ( ( itemp ) ) ** 2 ) )
    ten = np.sqrt ( np.sum ( np.abs ( np.power ( itemp, 2 ) ), axis=1) )[:,np.newaxis]
    return itemp / ten

def get_args():
    import argparse as arp
    ap = arp.ArgumentParser (prog='templater',description='Prepares bowtie templates',)
    add = ap.add_argument
    add ('file',help='Filterbank file')
    add ('--dir', help='Directory to save the array', dest='dir', default='templates')
    add ('-n','--ndm', required=True, help='Number of DM trials in bowtie', dest='ndm',  type=int)
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    return ap.parse_args()

class BowtieTemplate:
    """
    Bowtie template

    DM axis critically sampled
    """
    def __init__ (self, tsamp=None, fch1=None, foff=None, nchans=None, ndm=None):
        """
        Creates an instance of `BowtieTemplate`

        Arguments
        ---------
        tsamp: float
            Sampling time in seconds
        fch1: float
            Frequency of the first channel in the filterbank in MHz
        foff: float
            Frequency channel offset in the filterbank in MHz
        nchans: int
            Number of channels in the filterbank
        ndm: int  (default=half of nchans)
            Number of DM-trials. Read the template design choices
            ndm should always be less than nchans because of FDMT
            Should be a power of two.

        """
        ##
        self.ndm           = ndm
        self.hdm           = self.ndm // 2
        ###
        self.tsamp         = tsamp
        self.fch1          = fch1
        self.foff          = foff
        self.nchans        = nchans
        ##
        self.faxis         = np.arange (self.nchans) * self.foff
        self.faxis        += self.fch1
        ##
        self.fullband_delay =  (self.faxis[self.nchans-1]/1E3)**-2 - (self.faxis[0]/1E3)**-2
        ifch02              = (self.faxis[0]/1E3)**-2
        self.inband_delay   = np.zeros (self.nchans, dtype=np.uint32)
        for i in range(self.nchans):
            self.inband_delay[i] = self.hdm * ( (self.faxis[i]/1E3)**-2 -ifch02 ) / self.fullband_delay 

        ##
        ## initialize FDMT
        if self.foff > 0:
            self.fdmt          = t_fdmt.FDMT_positive (self.fch1, self.foff, self.nchans)
        else:
            self.fdmt          = t_fdmt.FDMT_negative (self.fch1, self.foff, self.nchans)

    def __call__ (self, width, amplitude=1.0):
        """
        Creates a bowtie template with width

        We inject a noise-free dispersed signal with DM-delay=hdm (ndm//2)

        Width cannot be more than ndm//2. It is better to do downsample before searching.
        We do this because at any DM, we only want to "see" neighbouring 256 DM-trials only.
        In a way, our maximum width we can search without downsampling is `ndm//2`.

        Template size = (ndm, ndm)

        No need to fliplr(flipud(template)) since bowtie is already LR-UD symmetry

        Arguments
        ---------
        width: int
            Width in time units

        amplitude: float
            Amplitude of the dispersed pulse

        """
        ##
        ## limit
        if width > self.hdm:
            raise ValueError ("Cannot work with width > dm-trials//2.")

        ##
        ## create dispersed with no noise
        hwidth  = width // 2
        fb      = np.zeros ((self.nchans, 2*self.ndm +1), dtype=np.float32)
        for i in range (self.nchans):
            ia  = int( self.hdm + self.inband_delay[i] - hwidth )
            ib  = int( ia + width )
            fb[i,ia:ib] = amplitude

        ##
        ## create bowtie 
        bt     = self.fdmt.Bowtie_maxdt (fb, self.ndm)[:,:self.ndm]

        ##
        ## normalize bowtie
        bt     = NormalizeTemplate ( bt )

        ##
        ## return 
        return bt

    def save (self, filename, amplitude=1.0):
        """
        Saves the state and all the possible templates in an `npz` file
        """
        ##
        state     = dict(tsamp=self.tsamp, fch1=self.fch1, foff=self.foff, nchans=self.nchans, ndm=self.ndm)
        n_templates = int(np.log2 (self.hdm))
        templates   = np.zeros ((n_templates+1, self.ndm, self.ndm))
        ## computing widths
        widths      = np.power (2, np.arange(n_templates+1))
        ##
        for idx, iwidth in enumerate (widths):
            templates[idx] = self.__call__ (iwidth, amplitude=amplitude)
        ## saving
        np.savez (filename, templates=templates, widths=widths, **state)

class BoxcarTemplate:
    """
    Boxcar template


    DM axis critically sampled
    """
    def __init__ (self, tsamp=None, fch1=None, foff=None, nchans=None, ndm=None):
        """
        Creates an instance of `BoxcarTemplate`

        Arguments
        ---------
        tsamp: float
            Sampling time in seconds
        fch1: float
            Frequency of the first channel in the filterbank in MHz
        foff: float
            Frequency channel offset in the filterbank in MHz
        nchans: int
            Number of channels in the filterbank
        ndm: int  (default=half of nchans)
            Number of DM-trials. Read the template design choices
            ndm should always be less than nchans because of FDMT
            Should be a power of two.

        """
        ##
        self.ndm           = nchans   // 2
        self.hdm           = self.ndm // 2
        ###
        self.tsamp         = tsamp
        self.fch1          = fch1
        self.foff          = foff
        self.nchans        = nchans
        ##
        self.faxis         = np.arange (self.nchans) * self.foff
        self.faxis        += self.fch1
        ##
        self.fullband_delay =  (self.faxis[self.nchans-1]/1E3)**-2 - (self.faxis[0]/1E3)**-2
        ifch02              = (self.faxis[0]/1E3)**-2
        self.inband_delay   = np.zeros (self.nchans, dtype=np.uint32)
        for i in range(self.nchans):
            self.inband_delay[i] = self.hdm * ( (self.faxis[i]/1E3)**-2 -ifch02 ) / self.fullband_delay 

        ##
        ## initialize FDMT
        if self.foff > 0:
            self.fdmt          = t_fdmt.FDMT_positive (self.fch1, self.foff, self.nchans)
        else:
            self.fdmt          = t_fdmt.FDMT_negative (self.fch1, self.foff, self.nchans)

    def __call__ (self, width, amplitude=None):
        """
        Creates a boxcar template with width

        We inject a noise-free dispersed signal with DM-delay=hdm (ndm//2)

        Width cannot be more than ndm//2. It is better to do downsample before searching.
        We do this because at any DM, we only want to "see" neighbouring 256 DM-trials only.
        In a way, our maximum width we can search without downsampling is `ndm//2`.

        Template size = (ndm, ndm)

        No need to fliplr(flipud(template)) since bowtie is already LR-UD symmetry

        Normalize by `np.sum(template)`
        This normalizing is inspired by `averaging filter`, `laplacian filter` since all of them
        sum to unity.

        Arguments
        ---------
        width: int
            Width in time units

        amplitude: float
            Amplitude of the dispersed pulse
            Default is 1.0/sqrt (width)

        """
        ##
        ## limit
        if width > self.hdm:
            raise ValueError ("Cannot work with width > dm-trials//2.")
        
        if width == 1:
            raise ValueError ("Cannot work with width=1 in boxcar case ")

        if amplitude is None:
            amplitude = 1./np.sqrt (width)

        ##
        ## template
        hwidth   = width // 2
        template = np.zeros((self.ndm, self.ndm), dtype=np.float32)
        template[:,(self.hdm-hwidth):(self.hdm+hwidth)] = amplitude

        ##
        ## normalize template
        template = NormalizeTemplate ( template )

        ## return 
        return template

    def save (self, filename, amplitude=None):
        """
        Saves the state and all the possible templates in an `npz` file
        """
        ##
        state     = dict(tsamp=self.tsamp, fch1=self.fch1, foff=self.foff, nchans=self.nchans, ndm=self.ndm)
        n_templates = int(np.log2 (self.hdm))
        templates   = np.zeros ((n_templates, self.ndm, self.ndm))
        ## computing widths
        widths      = np.power (2, np.arange(1,n_templates+1))
        ##
        for idx, iwidth in enumerate (widths):
            templates[idx] = self.__call__ (iwidth, amplitude=amplitude)
        ## saving
        np.savez (filename, templates=templates, widths=widths, **state)

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
    ndm       = args.ndm
    hdm       = ndm // 2
    ## prepare filenames
    OFI       = os.path.join ( args.dir, OFILE.format (fname=filname, ndm=ndm, ext='npz') )
    ###############################
    ## read fil
    fil       = FilReader ( filpath )
    fh        = fil.header
    ## geometry
    nsamps    = fh.nsamples
    nchans    = fh.nchans
    foff      = fh.foff
    fch1      = fh.fch1
    tsamp     = fh.tsamp
    ## prepare info
    info             = dict ()
    info['fch1']     = fch1
    info['foff']     = foff
    info['nchans']   = nchans
    info['tsamp']    = tsamp
    info['ndm']      = ndm
    info['creation_datetime'] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    print (info)
    ###############################
    temper    = BowtieTemplate ( tsamp=tsamp, nchans=nchans, fch1=fch1, foff=foff, ndm=ndm )
    n_temps   = int ( np.log2 ( hdm ) )
    templates = np.zeros ((n_temps+1, ndm, ndm))
    widths    = np.power (2, np.arange(n_temps+1))
    ### call
    for idx, iwidth in enumerate (widths):
        templates[idx] = temper (iwidth, amplitude=1.0)
    ###############################
    np.savez ( OFI, templates=templates, widths=widths, **info )
