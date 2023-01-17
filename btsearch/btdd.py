import numpy as np

import pyximport; pyximport.install (setup_args={"include_dirs":np.get_include()})

import fdmt          as trishul_fdmt
import incoherent    as trishul_incoherent

DM_FAC = 4.148741601E-3
# v for MHz
# DM_CONST = 4.148741601E6

class BTDD:
    """
    Class to streamline computation of bowtie plane and de-dispersed filterbank.

    FDMT algorithm requires the number of DM-trials to be less than number-of-chans. 

    This class makes use of the optimization trick.
    Bowtie (A..B)  --> Incoherent (A) & Bowtie (B-A)
    Incoherent (C) --> Incoherent (A) & Incoherent (C-A)
    where A,B,C are delays

    For simplicity sake, the steps are numbered:
    -1-   Incoherent (A)
    -2.1- Incoherent (C-A) 
    -2.2- Bowtie (B-A)
    """
    def __init__ (self,
            fch1=None,foff=None,nchans=None,
            tsamp=None, 
            ndm=None,
            verbose=False,
        ):
        """

        Arguments
        ---------
        -All freq in MHz
        -tsamp in seconds
        -ndm = number of DM trials
        
        """
        self.v       = verbose
        ##
        self.fch1    = fch1
        self.foff    = foff
        self.nchans  = nchans
        self.tsamp   = tsamp
        self.ndm     = ndm
        self.hdm     = self.ndm//2
        self.f_axis  = self.fch1 + (np.arange (nchans, dtype=np.float32) * self.foff)
        self.bandpos = self.foff > 0
        ##
        if self.v:
            print (f"BTDD::init frequency {self.f_axis[0]:.2f}...{self.f_axis[-1]:.2f} MHz ({self.nchans:d} channels)")
            print (f"BTDD::init tsamp {self.tsamp*1E6:.3f} us")
            print (f"BTDD::init DM trials {self.ndm:d}")
        ##
        self.max_freq,self.min_freq = None, None
        if self.bandpos:
            self.max_freq = self.f_axis[self.nchans-1]
            self.min_freq = self.f_axis[0]
        else:
            self.max_freq = self.f_axis[0]
            self.min_freq = self.f_axis[self.nchans-1]
        self.fullband_delay      = DM_FAC * (  (self.min_freq/1E3)**-2 - (self.max_freq/1E3)**-2 )
        self.inband_delay        = np.zeros (nchans, dtype=np.float32)
        for i,f in enumerate (self.f_axis):
            self.inband_delay[i] = DM_FAC * ( (f/1E3)**-2 - (self.max_freq/1E3)**-2 )
        ##
        ##
        if self.bandpos:
            self.fdmt    = trishul_fdmt.FDMT_positive (self.fch1, self.foff, self.nchans)
            self.incoh   = trishul_incoherent.Dedisperser_positive
        else:
            self.fdmt    = trishul_fdmt.FDMT_negative (self.fch1, self.foff, self.nchans)
            self.incoh   = trishul_incoherent.Dedisperser_negative
        self.max_d   = 0
        self.dm_axis = None

    def __call__ (self, dm, dit=1):
        """
        Sets DM and calculates the delay arrays

        Arguments
        ---------
        dm: float
            Dispersion measure of the candidate

        dit: int
            Decimation in time performed while reading

        """
        ## 
        itsamp              = self.tsamp * dit
        if self.v:
            print (f"BTDD::call Downsampling = {dit:d}")
            print (f"BTDD::call Effective time resolution = {itsamp*1E6:.3f} us")
        ## reset
        self.__reset__ ()
        ## for step 1
        ddm                 = int (dm * self.fullband_delay  / itsamp )
        step1_ddm           = int(max (ddm-self.hdm, 0))
        step1_dm            = step1_ddm * itsamp / self.fullband_delay
        if self.v:
            print (f"BTDD::call Step-1 DM  = {step1_dm:.2f} units")
        self.step1_delays        = np.uint64 (self.inband_delay * step1_dm / itsamp )
        ## for step 2.1
        step21_dm           = dm - step1_dm
        self.step21_delays       = np.uint64 (self.inband_delay * step21_dm / itsamp )
        ## for step 2.2
        ## this bowtie computation is {0..256}
        ##
        self.max_d          = int (self.ndm + self.step1_delays[self.nchans-1])
        if self.v:
            print (f"BTDD::call Maximum delay = {self.max_d:d} samples")
        ## dm axis for plotting
        self.dm_axis        = np.arange (step1_ddm, step1_ddm+self.ndm, dtype=np.float32) * itsamp / self.fullband_delay

    def __reset__ (self):
        self.max_d           = 0
        self.step1_delays    = None
        self.step21_delays   = None
        self.dm_axis         = None

    def work_bt (self, fb, take):
        """
        fb is (nchans, nsamps)

        No slicing is done here. That should be done by the caller.
        """
        ff                  = fb - fb.mean ( 1 ).reshape ((-1, 1)) 
        fs                  = ff.std ( 1 ).reshape ((-1, 1))
        ff                  = np.divide ( ff, fs, where=fs!=0, out=np.zeros_like(fb))
        ## actual computation
        step1_dd            = self.incoh (ff, self.step1_delays)
        step22_bt           = self.fdmt.Bowtie_maxdt (step1_dd, self.ndm)

        ## return
        return step22_bt[...,:take]

    def work_btdd (self, fb, take):
        """
        fb is (nchans, nsamps)

        No slicing is done here. That should be done by the caller.
        """
        ff                  = fb - fb.mean ( 1 ).reshape ((-1, 1)) 
        fs                  = ff.std ( 1 ).reshape ((-1, 1))
        ff                  = np.divide ( ff, fs, where=fs!=0, out=np.zeros_like(fb))
        ## actual computation
        step1_dd            = self.incoh (ff, self.step1_delays)
        step21_dd           = self.incoh (step1_dd, self.step21_delays)
        step22_bt           = self.fdmt.Bowtie_maxdt (step1_dd, self.ndm)

        ## return
        return step22_bt[...,:take], step21_dd[...,:take]

if __name__ == "__main__":
    """
    tests with 1200...1600 MHz, 512 channels, 64us time resolution and 256 DM trials
    """

    btdd     = BTDD ( 1600, -400/512., 512, 64E-6, 256, False )

    btdd ( 100.0, 1 )
    nsamp    = int ( btdd.max_d * 3 )
    test     = np.ones ((btdd.nchans, nsamp), dtype=np.float32)
    bt,dd    = btdd.work ( test )

    print (bt)
