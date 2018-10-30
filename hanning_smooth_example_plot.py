import numpy

# def smooth(x,window_len=11,window='hanning'):
#     """smooth the data using a window with requested size.
    
#     This method is based on the convolution of a scaled window with the signal.
#     The signal is prepared by introducing reflected copies of the signal 
#     (with the window size) in both ends so that transient parts are minimized
#     in the begining and end part of the output signal.
    
#     input:
#         x: the input signal 
#         window_len: the dimension of the smoothing window; should be an odd integer
#         window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#             flat window will produce a moving average smoothing.

#     output:
#         the smoothed signal
        
#     example:

#     t=linspace(-2,2,0.1)
#     x=sin(t)+randn(len(t))*0.1
#     y=smooth(x)
    
#     see also: 
    
#     numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
#     scipy.signal.lfilter
 
#     TODO: the window parameter could be the window itself if an array instead of a string
#     NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
#     """

#     if x.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.")

#     if x.size < window_len:
#         raise ValueError("Input vector needs to be bigger than window size.")


#     if window_len<3:
#         return x


#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


#     s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#     print(len(s))
#     if window == 'flat': #moving average
#         w=numpy.ones(window_len,'d')
#     else:
#         w=eval('numpy.'+window+'(window_len)')

#     y=numpy.convolve(w/w.sum(),s,mode='valid')
#     return y


def smooth( x, window_len=11, window='hanning' ):
    '''
    see here: https://stackoverflow.com/questions/5515720/python-smooth-time-series-data
    which is a working modification of a the scipy cookbook smooth() example
    '''
    import numpy as np

    if x.ndim != 1:
        raise ValueError( "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError( "Input vector needs to be bigger than window size." )
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError( "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    
    if window == 'flat': # moving average
        w=np.ones(window_len,'d')
    else:
        windows = { 'hanning':np.hanning, 'hamming':np.hamming, 'bartlett':np.bartlett, 'blackman':np.blackman }
        w = windows[ window ]( window_len )
    
    y=np.convolve( w/w.sum(), s, mode='same' )
    return y[ window_len:-window_len+1 ]


def smooth2( x, window_len, window ):
    from scipy import signal
    if window == 'flat': # moving average
        win=np.ones(window_len,'d')
    else:
        # win = signal.hann( window_len )
        windows = { 'hanning':np.hanning, 'hamming':np.hamming, 'bartlett':np.bartlett, 'blackman':np.blackman }
        win = windows[ window ]( window_len )
    filtered = signal.convolve(x, win, mode='same') / sum(win)
    return filtered


from numpy import *
from pylab import *

output = dict()

import pandas as pd
profile = pd.read_csv('/Users/malindgren/Documents/repos/seaice_noaa_indicators/tmp_data/profile_dat.csv', index_col=0, header=None, parse_dates=True)
profile = profile.loc[ slice('1979','2016') ]
xn = profile[1].values

# ws=len(xn)
ws_list = range(3,32)
# ws = 11
for ws in ws_list:
    subplot(211)
    plot(ones(ws))

    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    # windows=['flat', 'hanning', 'hamming'] 
    hold(True)

    # deals with the flat filter.
    for w in windows[1:]:
        eval('plot('+w+'(ws) )')

    axis([0,30,0,1.1])

    legend(windows)
    title("The smoothing windows")
    subplot(212)

    w = 'hanning'

    for w in windows:
        # plot( smooth( xn, ws, w ) )
        output[ws] = smooth2( xn, ws, w )

    df = pd.DataFrame(output)
    df.corrwith(profile[1].reset_index(drop=True), axis=1)

    df['raw'] = xn
    show()