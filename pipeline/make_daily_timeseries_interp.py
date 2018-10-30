# make a full daily array with np.nan array slices at times that are missing and interpolate missing values linearly
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
    attrs = {'proj4string':'EPSG:3411', 'proj_name':'NSIDC North Pole Stereographic', 'affine_transform': str(list(meta['transform']))}

    ds = xr.Dataset({'sic':(['time','yc', 'xc'], arr)},
                coords={'xc': ('xc', xc[0,]),
                        'yc': ('yc', yc[:,0]),
                        'time':times }, attrs=attrs )
    return ds

def mean_filter_2D( arr, footprint ):
    from scipy.ndimage import generic_filter
    return generic_filter( arr, np.nanmean, footprint=footprint )

def smooth3( x ):
    from scipy import signal
    win = np.array([0.25,0.5,0.25])
    return signal.convolve(x, win, mode='same') / sum(win)

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# # hanning smooth the data along the time axis... [check with Mark whether this is correct]
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

def stack_rasters( files, ncpus=32 ):
    pool = mp.Pool( ncpus )
    arr = np.array( pool.map( open_raster, files ) )
    pool.close()
    pool.join()
    return arr

def spatial_smooth( arr, footprint='rooks', ncpus=32 ):
    f = partial( mean_filter_2D, footprint=footprint )
    pool = mp.Pool( ncpus )
    out_arr = pool.map( f, [a for a in arr] )
    pool.close()
    pool.join()
    return out_arr

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

    # parse some args
    parser = argparse.ArgumentParser( description='stack the hourly outputs from raw WRF outputs to NetCDF files of hourlies broken up by year.' )
    parser.add_argument( "-b", "--base_path", action='store', dest='base_path', type=str, help="input hourly directory containing the NSIDC_0051 data converted to GTiff" )
    parser.add_argument( "-w", "--window_len", action='store', dest='window_len', type=str, help="window length to add to the output NetCDF file name" )
    parser.add_argument( "-n", "--ncpus", action='store', dest='ncpus', type=int, help="number of cpus to use" )
    
    # unpack args
    args = parser.parse_args()
    base_path = args.base_path
    window_len = args.window_len
    ncpus = args.ncpus

    # # # # # # 
    # base_path = '/atlas_scratch/malindgren/nsidc_0051'
    # window_len = 'paper_weights'
    # ncpus = 32
    # window_len = 4
    # # # # # # 

    # list all data
    input_path = os.path.join( base_path,'GTiff','alaska' )
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

    arr = stack_rasters( files, ncpus=32 )
    ds = make_xarray_dset( arr, pd.DatetimeIndex(data_times), meta )
    da = ds['sic']

    # RESAMPLE TO DAILY...
    da_interp = da.resample(time='1D').interpolate('slinear')

    # spatial smoothing...
    # spatially smooth the 2-D daily slices of data using a mean generic filter. (without any aggregation)
    footprint_type = 'rooks'
    footprint_lu = {'rooks':np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), 
                    'queens':np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])}

    footprint = footprint_lu[ footprint_type ]
    spatial_smoothed = np.array(spatial_smooth( da_interp.values, footprint=footprint, ncpus=ncpus ))

    if window_len == 'paper_weights':
        # hanning smooth
        hanning_smoothed = np.apply_along_axis( smooth3, arr=spatial_smoothed, axis=0 )
        hanning_smoothed[(da_interp.values > 1) | (da_interp.values < 0)] = da_interp.values[(da_interp.values > 1) | (da_interp.values < 0)]
    else:
        fsmooth2 = partial( smooth2, window_len=window_len, window='hanning' )
        new_arr = np.apply_along_axis( fsmooth2, arr=new_arr, axis=0 )


    # write this out as a GeoTiff
    out_fn = os.path.join( base_path,'GTiff','alaska_singlefile','nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_{}.tif'.format(str(begin.year),str(end.year),window_len) )
    _ = make_output_dirs( os.path.dirname(out_fn) )
    meta.update(count=hanning_smoothed.shape[0])
    with rasterio.open( out_fn, 'w', **meta ) as out:
        out.write( hanning_smoothed.astype(np.float32) )

    # write it out as a NetCDF
    out_ds = da_interp.copy(deep=True)
    out_ds.values = hanning_smoothed
    out_ds = da_interp.to_dataset( name='sic' )
    out_ds.attrs = ds.attrs

    # output encoding
    encoding = out_ds.sic.encoding.copy()
    encoding.update({ 'zlib':True, 'comp':5, 'contiguous':False, 'dtype':'float32' })
    out_ds.sic.encoding = encoding

    out_fn = os.path.join( base_path,'NetCDF','nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_{}.nc'.format(str(begin.year),str(end.year),window_len) )
    _ = make_output_dirs( os.path.dirname(out_fn) )
    out_ds.to_netcdf( out_fn, format='NETCDF3_64BIT' )



