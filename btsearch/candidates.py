"""
efficient? data structure to keep track 

of (s/n, sample, dm_index) per width

this should be cythonized

"""

import numpy as np
# import pandas as pd

ISN = 0 
ISM = 1
IWD = 2
IDM = 3

HEAD_STR = "   {sn:7s}      {time:12s}      {dm:4s}      {dfac:4s}      {sample:8s}\n"
CAND_STR = "{sn: 7.2f}      {time: 12.8f}      {dm: 4.2f}      {dfac: 4d}      {sample: 8d}\n"

class Candidates:
    """
    class performing candidate aggregation
    online

    usage = 
    for every block
        reset
        for every width
            call
        aggregate
    save

    the same event will not have multiple width candidates at different times.
    this is because of averaging effect of bowtie template. 
    even if it so, we expect then there are two sub-bursts (bursts with two distinct peaks)

    given collection of candidates, 
    we sort over (time, width).
    Then, we perform a linear scan and keep track of the overlapping candidates.
    For the overlapping candidates, we pick the one with the maximum s/n.

    """
    def __init__ (self, ntemplates, tsamp, d):
        """
        does aggregation for every template
        """
        self.count  = 0
        self.n      = ntemplates
        self.tsamp  = tsamp
        self.pkg    = []
        self.agg    = dict(sn=[], time=[], dm=[], dfac=[], sample=[])
        self.total_count = 0
        ##
        self.dmx    = [float(x) for x in d]

    def __reset (self):
        """ resets pkg """
        self.pkg   = []
        self.count = 0

    def aggregate (self):
        """
        workhorse

        we are not using DM here. 
        The DM level pruning is already done by sauf.CCL
        """
        if self.count <= 0:
            return

        ## sort by ( time, width )
        self.pkg.sort ( key=lambda x : (x[ISM],x[IWD]) )

        ## linear scan over self.pkg
        ## do overlap test
        ## if overlap: aggreate over s/n (pick max s/n one)
        ## if no overlap: current candidate is a candidate store it
        current_cand  = self.pkg[0]
        ## bool to track if ended on overlap
        on_overlap    = False
        for i in range (1, self.count):
            i_cand    = self.pkg[i]
            if self.__overlap ( current_cand, i_cand ):
                current_cand = self.__pick_one ( current_cand, i_cand )
                on_overlap = True
            else:
                self.__store_cand ( current_cand )
                current_cand = i_cand
                on_overlap = False
        # if on_overlap:
        self.__store_cand ( current_cand )

        ## reset
        self.__reset ()

    def __store_cand (self, a):
        """a is an ordered tuple"""
        self.agg['sn'].append ( a[ISN] )
        self.agg['sample'].append ( a[ISM] )
        self.agg['dm'].append ( self.dmx[ a[IDM] ] )
        self.agg['dfac'].append ( a[IWD] )
        self.agg['time'].append ( a[ISM] * self.tsamp )
        self.total_count += 1

    def __pick_one (self, a, b):
        """a,b are ordered tuples
            
            picks one with more s/n
        """
        ds  = a[ISN] > b[ISN]
        if ds: return a
        else: return b

    def __overlap (self, a, b):
        """a,b are ordered tuples

            overlap test:
                | ta - tb | < 0.5 * | wa + wb |

            returns True if overlap
        """
        dt = abs ( int( a[ISM] ) - int( b[ISM] ) )
        dw = abs ( a[IWD] + b[IWD] )
        # print (f" Candidates::__overlap a={a},b={b} dt={dt}, dw={dw}, ret={dt < dw//2}")
        return dt < (dw//2)

    def print (self):
        """
        print
        """
        import pandas as pd
        df = pd.DataFrame ( self.agg )
        print (df)


    def save (self, filename):
        """
        do like a presto.singlepulse 
        but with no time

        dm | sn | time | dfac
        ".cands"
        """
        if not filename.endswith (".cands"):
            filename += ".cands"

        ## make a df
        ### pandas.DataFrame.to_csv does not allow
        ### to set precision per column
        ### which is a bummer

        ## hence, writing the old fashioned way
        with open (filename, 'w') as f:
            # print ( HEAD_STR.format (dm='dm', sn='sn', time='time_s', dfac='dfac', sample='sample'), end='' )
            f.write ( HEAD_STR.format (dm='dm', sn='sn', time='time_s', dfac='dfac', sample='sample') )
            for dm, sn, time, dfac, sample in zip (    \
                    self.agg['dm'], self.agg['sn'], self.agg['time'], \
                    self.agg['dfac'], self.agg['sample']
            ):
                # print ( CAND_STR.format (dm=dm, sn=sn, time=time, dfac=dfac, sample=sample), end='' )
                f.write ( CAND_STR.format (dm=dm, sn=sn, time=time, dfac=dfac, sample=sample) )

    def __call__ (self, iw, sn=[], sm=[], dm=[], sm_offset=0):
        """
        sn, sm, dm
        """
        for s,m,d in zip ( sn, sm, dm ):
            self.pkg.append ( (s,m+sm_offset,iw,d) )
            self.count += 1

if __name__ == "__main__":
    cc = Candidates ( 3, 216E-6 )

    #####
    cc ( 1, [6., 8.],  [10, 16], [0., 0.] )
    cc ( 2, [9., 12.], [12, 15], [1., 1.] )
    cc ( 4, [4., 9.],  [9, 19],  [2., 2.] )
    #####
    cc.aggregate ()
    #####
    cc.save ("lmao")

