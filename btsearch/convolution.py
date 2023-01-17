"""
convolution engine
"""
import numpy  as np

from scipy import fft as sp_fft
from numpy import fft as np_fft


########## use scipy-fft?
FFT    = sp_fft.rfftn
IFFT   = sp_fft.irfftn
########## use numpy-fft?
FFT    = np_fft.rfftn
IFFT   = np_fft.irfftn

class ConvolutionEngine:
    """
    Why write a new class?

    Performs 2D FFT convolution and takes care of the 
    sizes and shapes to make shit optimized

    assume x,y are real

    Problem x, y
    x = bt-array
    y = template

    x.shape is always preserved
    """
    def __init__ (self, xshape, yshape):
        """
        initial arrays and shit
        """
        ## take shapes
        self.shape0_x   = xshape[0]
        self.shape1_x   = xshape[1]
        self.shape0_y   = yshape[0]
        self.shape1_y   = yshape[1]

        # print (f" shapes x=({self.shape0_x:d}, {self.shape1_x:d}) y=({self.shape0_y:d}, {self.shape1_y:d})")

        ## convolution size
        self.chape0     = self.shape0_x + self.shape0_y - 1
        self.chape1     = self.shape1_x + self.shape1_y - 1

        # print (f" convolution shape = ({self.chape0:d},{self.chape1:d})")

        ## fast fft always
        self.shape0_fft   = sp_fft.next_fast_len ( self.chape0, True )
        self.shape1_fft   = sp_fft.next_fast_len ( self.chape1, True )

        # print (f" fast_fft shape = ({self.shape0_fft:d}, {self.shape1_fft:d})")
        
        ## rfft changes the last dimension
        self.shape1_rfft  = ( self.shape1_fft//2 ) + 1

        ## initial arrays?
        # self.fft_shape    = ( self.shape0_fft, self.shape1_rfft )
        self.fft_shape    = ( self.shape0_fft, self.shape1_fft )
        self.ifft_shape   = ( self.shape0_fft, self.shape1_fft )

        # print (f"  fft shape = {self.fft_shape}")
        # print (f" ifft shape = {self.ifft_shape}")

        ## slice_shape
        c0            = ( self.shape0_y - 1 ) // 2
        c1            = ( self.shape1_y - 1 ) // 2
        self.thape0   = slice ( c0, c0  + self.shape0_x )
        self.thape1   = slice ( c1, c1  + self.shape1_x )

        # print (f" slices x={self.thape0} y={self.thape1}")

    def do_fft ( self, x ):
        """
        no truncation is done 

        intput shape = xshape || yshape
        output shape = self.fft_shape
        """
        # fx = FFT ( x, self.fft_shape )
        # print (fx.dtype)
        # return fx
        return FFT ( x, self.fft_shape )

    def do_ifft ( self, y ):
        """
        slicing is done

        intput shape = self.fftshape
        output shape = xshape
        """
        fy   = IFFT ( y, self.fft_shape )
        return fy[self.thape0, self.thape1]


if __name__ == "__main__":
    ce  = ConvolutionEngine ( (256, 8192+256), (256, 256) )
