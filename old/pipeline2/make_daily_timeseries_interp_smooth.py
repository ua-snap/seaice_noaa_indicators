# # # # # # # # # # # # # # # # # # # # 
# # make a full daily array with and 
# # interpolate missing dates linearly
# # # # # # # # # # # # # # # # # # # # 

# initial numba imports needed for the func load:
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer


def nan_helper( y ):
    '''
    Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    '''
    return np.isnan( y ), lambda z: z.nonzero()[0]

def interp_1d_along_axis( y ):
    ''' interpolate across 1D timeslices of a 3D array. '''
    nans, x = nan_helper( y )
    y[nans] = np.interp( x(nans), x(~nans), y[~nans] )
    return y

def make_datetimes( timestr ):
    # timestr = '19790703'
    year = int(timestr[:4])
    month = int(timestr[4:6])
    day = int(timestr[6:])
    return dt.datetime(year,month,day)

def open_raster( fn ):
    with rasterio.open( fn ) as rst:
        arr = rst.read(1)
    return arr

def coordinates( fn=None, meta=None, numpy_array=None, input_crs=None, to_latlong=False ):
    '''
    take a raster file as input and return the centroid coords for each 
    of the grid cells as a pair of numpy 2d arrays (longitude, latitude)

    User must give either:
        fn = path to the rasterio readable raster
    OR
        meta & numpy ndarray (usually obtained by rasterio.open(fn).read( 1 )) 
        where:
        meta = a rasterio style metadata dictionary ( rasterio.open(fn).meta )
        numpy_array = 2d numpy array representing a raster described by the meta

    input_crs = rasterio style proj4 dict, example: { 'init':'epsg:3338' }
    to_latlong = boolean.  If True all coordinates will be returned as EPSG:4326
                         If False all coordinates will be returned in input_crs
    returns:
        meshgrid of longitudes and latitudes

    borrowed from here: https://gis.stackexchange.com/a/129857
    ''' 
    
    import rasterio
    import numpy as np
    from affine import Affine
    from pyproj import Proj, transform

    if fn:
        # Read raster
        with rasterio.open( fn ) as r:
            T0 = r.transform  # upper-left pixel corner affine transform
            p1 = Proj( r.crs )
            A = r.read( 1 )  # pixel values

    elif (meta is not None) & (numpy_array is not None):
        A = numpy_array
        if input_crs != None:
            p1 = Proj( input_crs )
            T0 = meta[ 'transform' ]
        else:
            p1 = None
            T0 = meta[ 'transform' ]
    else:
        BaseException( 'check inputs' )

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))
    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation( 0.5, 0.5 )
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: ( c, r ) * T1
    # All eastings and northings -- this is much faster than np.apply_along_axis
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)

    if to_latlong == False:
        return eastings, northings
    elif (to_latlong == True) & (input_crs != None):
        # Project all longitudes, latitudes
        longs, lats = transform(p1, p1.to_latlong(), eastings, northings)
        return longs, lats
    else:
        BaseException( 'cant reproject to latlong without an input_crs' )

def make_xarray_dset( arr, times, rasterio_meta_dict ):
    meta = rasterio_meta_dict
    xc,yc = coordinates(meta=meta, numpy_array=arr[1,...])
    attrs = {'proj4string':'EPSG:3411', 'proj_name':'NSIDC North Pole Stereographic', 
            'affine_transform': str(list(meta['transform']))}

    ds = xr.Dataset({'sic':(['time','yc', 'xc'], arr)},
                coords={'xc': ('xc', xc[0,]),
                        'yc': ('yc', yc[:,0]),
                        'time':times }, attrs=attrs )
    return ds

def mean_filter_2D( arr, footprint ):
    from scipy.ndimage import generic_filter
    out = generic_filter( arr, np.nanmean, footprint=footprint, origin=0 )
    out[np.isnan(arr)] = np.nan
    return out



# def mean_filter_2D(arr, size):
#     from scipy.ndimage import uniform_filter
#     return uniform_filter(arr, size=3, mode='constant')

def smooth2( x, window_len, window ):
    ''' [ currently unused ]
    hanning smooth the data along the time axis... this is one of the versions
    '''
    from scipy import signal
    if window == 'flat': # moving average
        win=np.ones(window_len,'d')
    else:
        # win = signal.hann( window_len )
        windows = { 'hanning':np.hanning, 'hamming':np.hamming, 'bartlett':np.bartlett, 'blackman':np.blackman }
        win = windows[ window ]( window_len )
    filtered = signal.convolve(x, win, mode='same') / sum(win)
    return filtered

def smooth3( x ):
    ''' smoothing to mimick the smoothing from meetings with Mark/Hajo'''
    from scipy import signal
    win = np.array([0.25,0.5,0.25])
    return signal.convolve(x, win, mode='same') / sum(win)

# def smooth4( x ):
#     from scipy.ndimage import generic_filter
#     win = np.array([0.25,0.5,0.25])
#     return generic_filter( arr, np.nanmean, footprint=footprint, origin=0 )


def stack_rasters( files, ncpus=32 ):
    pool = mp.Pool( ncpus )
    arr = np.array( pool.map( open_raster, files ) )
    pool.close()
    pool.join()
    return arr

def spatial_smooth( arr, footprint, ncpus=32 ):
    arr_list = [a.copy() for a in arr] # unpack 3d (time,rows,cols) array to 2d list
    f = partial( mean_filter_2D, footprint=footprint )
    pool = mp.Pool( ncpus )
    out_arr = pool.map( f, arr_list )
    pool.close()
    pool.join()
    return np.array(out_arr)

# def spatial_smooth( arr, size=3, ncpus=32 ):
#     f = partial( mean_filter_2D, size=size )
#     pool = mp.Pool( ncpus )
#     out_arr = pool.map( f, [a for a in arr] )
#     pool.close()
#     pool.join()
#     return np.array(out_arr)

def spatial_smooth_serial( arr, footprint ):
    arr_list = [a for a in arr][:30]
    f = partial( mean_filter_2D, footprint=footprint )
    out_arr = np.array([f(i.copy()) for i in arr_list])
    return out_arr


@cfunc(intc(CPointer(float64), intp,
        CPointer(float64), voidptr))
def spatial_smooth_serial_numba(values_ptr, len_values, result, data):
    ''' 
    perform a mean filter over a window using scipy.ndimage.generic_filter
    faster with this compiled numba function.
    '''
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = values[0]
    for v in values[1:]:
        result[0] = result[0] + v
    result[0] = result[0] / len_values
    return 1

def make_output_dirs( dirname ):
    if not os.path.exists( dirname ):
        _ = os.makedirs( dirname )
    return dirname


if __name__ == '__main__':
    import os, rasterio
    import datetime as dt
    import pandas as pd
    import numpy as np
    import xarray as xr
    from functools import partial
    import multiprocessing as mp
    import argparse
    from scipy.ndimage import generic_filter
    from scipy import LowLevelCallable

    # parse some args
    parser = argparse.ArgumentParser( description='stack the hourly outputs from raw WRF outputs to NetCDF files of hourlies broken up by year.' )
    parser.add_argument( "-b", "--base_path", action='store', dest='base_path', type=str, help="input hourly directory containing the NSIDC_0051 data converted to GTiff" )
    parser.add_argument( "-n", "--ncpus", action='store', dest='ncpus', type=int, help="number of cpus to use" )
    
    # unpack args
    args = parser.parse_args()
    base_path = args.base_path
    ncpus = args.ncpus

    # # # TESTING
    base_path = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051'
    ncpus = 32
    # # # # # 

    # list all data
    input_path = os.path.join( base_path,'prepped','north' )
    files = sorted([ os.path.join(r,fn) for r,s,files in os.walk(input_path) for fn in files if fn.endswith('.tif') ])
    data_times = [ make_datetimes( os.path.basename(fn).split('.')[0].split('_')[1] ) for fn in files ]

    # date-fu for filenames and slicing
    begin = data_times[0]
    end = data_times[-1]
    begin_str = begin.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')

    # stack the irregularly spaced data to a netcdf
    with rasterio.open( files[0] ) as template:
        meta = template.meta.copy()
        height,width = template.shape

    arr = stack_rasters( files, ncpus=ncpus )
    ds = make_xarray_dset( arr.copy(), pd.DatetimeIndex(data_times), meta )
    da = ds['sic'].copy()
    # del arr # cleanup

    # get a masks layer from the raw files.  These are all values > 250
    # ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
    # 251 Circular mask used in the Arctic to cover the irregularly-shaped data 
    #       gap around the pole (caused by the orbit inclination and instrument swath)
    # 252 Unused
    # 253 Coastlines
    # 254 Superimposed land mask
    # 255 Missing data
    mask = arr[0].copy()
    mask[mask < 251] = 0 # convert all non-mask values to 0

    # RESAMPLE TO DAILY...
    dat = da.values.copy()
    polemask = (dat != 251)
    dat[ np.where(dat > 100)] = np.nan
    da.data = dat.copy()
    del dat, arr # cleanup
    
    # # # interpolate the timeseries to regular dailies
    da_interp = da.resample(time='1D').asfreq() # test
    
    def interpolate(x):
        if not np.isnan(x).all():
            index = np.arange(len(x))
            notnan = np.logical_not(np.isnan(x))
            return np.interp(index, index[notnan], x[notnan])

    da_interp.data = np.apply_along_axis(interpolate, axis=0, arr=da_interp).round(4)
    # # # 

    # # # # TEST doesnt deal with np.nan's in a clean way
    # from scipy.ndimage.filters import convolve
    # arr2 = convolve(arr, weights=np.full((3, 3), 1.0/9), origin=0, mode='nearest')
    # convolve2d(arr,np.full((3, 3), 1.0/9),max_missing=0.5,verbose=True)
    # # # # 
    
    # old ts interpolation...
    # da_interp_old = da.resample(time='1D').interpolate('linear')
    # da_interp

    # spatial smoothing...
    # spatially smooth the 2-D daily slices of data using a mean generic filter. (without any aggregation)
    footprint_type = 'queens'
    footprint_lu = {'rooks':np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), 
                    'queens':np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])}

    footprint = footprint_lu[ footprint_type ]
    print('spatial smooth')
    # spatial_smoothed = spatial_smooth( da_interp.values, footprint=footprint, ncpus=30 )
    # spatial_smoothed = spatial_smooth_serial( da_interp.values, footprint=footprint )
    # spatial_smoothed = spatial_smooth( da_interp.values, size=3, ncpus=ncpus )
    spatial_smoothed = [generic_filter(a.copy(), LowLevelCallable(spatial_smooth_serial_numba.ctypes), footprint=footprint) for a in da_interp.values]

    # # # # # TEST NEW SMOOTHING:
    # arr_list = [a.copy() for a in da_interp.values]
    # f = partial( mean_filter_2D, footprint=footprint )
    # pool = mp.Pool( ncpus-1 )
    # spatial_smoothed = pool.map( f, arr_list )
    # pool.close()
    # pool.join()
    # # spatial_smoothed = np.array(out_arr)
    # # # # # # END TEST

    # mask the spatial smoothed outputs with the mask at each 2D slice.
    def _maskit(x, mask):
        x[mask != 0] = np.nan
        return x
    
    smoothed = np.array([_maskit(i, mask) for i in spatial_smoothed]).copy()

    # hanning smooth -- we do this 3x according to Mark
    
    print('hanning smooth')
    
    # # # # # TEST
    # from functools import reduce
    
    # n = 3
    # def hann_smooth( x ):
    #     return np.apply_along_axis( smooth3, arr=x, axis=0 )
    
    # test = reduce(lambda x, _: hann_smooth(x), range(n), smoothed)
    # # # # # END TEST
    
    n = 3
    # explicit method
    for i in range(n):
        smoothed = np.apply_along_axis( smooth3, arr=smoothed, axis=0 )

    # hanning_smoothed = np.apply_along_axis( smooth3, arr=hanning_smoothed, axis=0 )
    # hanning_smoothed = np.apply_along_axis( smooth3, arr=hanning_smoothed, axis=0 )

    # make sure no values < 0, set to 0
    smoothed[ np.where(smoothed < 0) ] = 0

    # make sure no values > 1, set to 1
    smoothed[ np.where(smoothed > 1) ] = 1

    # write this out as a GeoTiff
    out_fn = os.path.join( base_path,'smoothed','GTiff','nsidc_0051_sic_nasateam_{}-{}_north_smoothed.tif'.format(str(begin.year),str(end.year)) )
    _ = make_output_dirs( os.path.dirname(out_fn) )
    meta.update(count=smoothed.shape[0], compress='lzw')
    with rasterio.open( out_fn, 'w', **meta ) as out:
        out.write( smoothed.astype(np.float32) )

    # write it out as a NetCDF
    out_ds = da_interp.copy(deep=True)
    out_ds.values = smoothed.astype(np.float32)
    out_ds = out_ds.to_dataset( name='sic' )
    out_ds.attrs = ds.attrs

    # output encoding
    encoding = out_ds.sic.encoding.copy()
    encoding.update({ 'zlib':True, 'comp':5, 'contiguous':False, 'dtype':'float32' })
    out_ds.sic.encoding = encoding

    out_fn = os.path.join( base_path,'smoothed','NetCDF','nsidc_0051_sic_nasateam_{}-{}_north_smoothed.nc'.format(str(begin.year),str(end.year)) )
    _ = make_output_dirs( os.path.dirname(out_fn) )
    out_ds.to_netcdf( out_fn ) # , format='NETCDF4' )
