"""Smooth daily SIC data

Usage:
    Functions for step #3 of main data pipeline, used for 
    smoothing the daily NSIDC-0051 data.

Notes:
    Interpolates missing dates linearly
    (2D spatial / 1D profile hann smoothed)
"""

import os
import math
import numba
import datetime as dt
import multiprocessing as mp
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from affine import Affine
from pyproj import Proj, transform
from numba import cfunc, carray
from numba.types import intc, CPointer, float64, intp, voidptr
from scipy import LowLevelCallable
from scipy.ndimage import generic_filter


def nan_helper(y):
    """
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
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interp_1d_along_axis(y):
    """ interpolate across 1D timeslices of a 3D array. """
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def interpolate(x):
    if not np.isnan(x).all():
        index = np.arange(len(x))
        notnan = np.logical_not(np.isnan(x))
        return np.interp(index, index[notnan], x[notnan])


def make_datetimes(timestr):
    # timestr = '19790703'
    year = int(timestr[:4])
    month = int(timestr[4:6])
    day = int(timestr[6:])
    return dt.datetime(year, month, day)


def open_raster(fn):
    """ 
    open a raster using `rasterio` and return
    the `numpy` array representing band 1
    """
    with rio.open(fn) as rst:
        arr = rst.read(1)
    return arr


def coordinates(fn=None, meta=None, numpy_array=None, input_crs=None, to_latlong=False):
    """
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
    """
    if fn:
        # Read raster
        with rio.open(fn) as r:
            T0 = r.transform  # upper-left pixel corner affine transform
            p1 = Proj(r.crs)
            A = r.read(1)  # pixel values

    elif (meta is not None) & (numpy_array is not None):
        A = numpy_array
        if input_crs != None:
            p1 = Proj(input_crs)
            T0 = meta["transform"]
        else:
            p1 = None
            T0 = meta["transform"]
    else:
        BaseException("check inputs")

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))
    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: T1 * (c, r)
    # All eastings and northings -- this is much faster than np.apply_along_axis
    eastings, northings = np.vectorize(rc2en, otypes=[np.float32, np.float32])(
        rows, cols
    )

    if to_latlong == False:
        return eastings, northings
    elif (to_latlong == True) & (input_crs != None):
        # Project all longitudes, latitudes
        longs, lats = transform(p1, p1.to_latlong(), eastings, northings)
        return longs, lats
    else:
        BaseException("cant reproject to latlong without an input_crs")


def make_xarray_dset(arr, times, rasterio_meta_dict):
    meta = rasterio_meta_dict
    xc, yc = coordinates(meta=meta, numpy_array=arr[1, ...])
    attrs = {
        "proj4string": "EPSG:3411",
        "proj_name": "NSIDC North Pole Stereographic",
        "affine_transform": str(list(meta["transform"])),
    }

    ds = xr.Dataset(
        {"sic": (["time", "yc", "xc"], arr)},
        coords={"xc": ("xc", xc[0,]), "yc": ("yc", yc[:, 0]), "time": times},
        attrs=attrs,
    )
    return ds


# def mean_filter_2D( arr, footprint ):
#     '''
#     2D mean filter that overlooks np.nan and -9999 masks
#     while averaging across the footprint window.

#     input is a 2D array and footprint

#     output is a smoothed 2D array
#     '''
#     from scipy.ndimage import generic_filter

#     indmask = np.where(arr == -9999)
#     indnodata = np.where(np.isnan(arr) == True)
#     arr[indmask] = np.nan # make mask nodata
#     out = generic_filter( arr, np.nanmean, footprint=footprint, origin=0 )
#     out[indmask] = -9999 # mask
#     out[indnodata] = np.nan # nodata
                                                                                                                 
#     return out


def jit_filter_function(filter_function):
    """Numba decorator for JIT-compiling numpy.nanmean

    Notes:
        Code borrowed from https://ilovesymposia.com/2017/03/15/prettier-lowlevelcallables-with-numba-jit-and-decorators/
    """
    jitted_function = numba.jit(filter_function, nopython=True)

    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
    def wrapped(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=float64)
        result[0] = jitted_function(values)
        return 1

    return LowLevelCallable(wrapped.ctypes)


@jit_filter_function
def jit_nanmean(arr):
    """JIT-compiiled numpy.nanmean"""
    return np.nanmean(arr)


def jit_mean_filter(arr, footprint):
    """ 
    2D mean filter that overlooks np.nan and -9999 masks
    while averaging across the footprint window.

    input is a 2D array and footprint

    output is a smoothed 2D array
    """
    indmask = np.where(arr == -9999)
    indnodata = np.where(np.isnan(arr) == True)
    arr[indmask] = np.nan  # make mask nodata
    out = generic_filter(arr, jit_nanmean, footprint=footprint, origin=0)
    out[indmask] = -9999  # mask
    out[indnodata] = np.nan  # nodata

    return out


def run_jit_mean_filter(arr, footprint):
    """Run loop of ji_mean_filter_2D"""
    return np.array([jit_mean_filter(a, footprint) for a in arr])


def chunkit(size, n_chunks):
    """get slices of array indices that approximately equally partition the array"""
    chunk_length = math.floor(size / n_chunks)
    chunk_slices = [
        slice(x * chunk_length, (x + 1) * chunk_length) for x in range(n_chunks - 1)
    ]
    chunk_slices.append(slice((n_chunks - 1) * chunk_length, size))

    return chunk_slices


# def run_meanfilter(x):
#     return mean_filter_2D( *x )


def hanning_smooth(x):
    """ smoothing to mimick the smoothing from meetings with Mark/Hajo"""
    from scipy import signal

    win = np.array([0.25, 0.5, 0.25])
    return signal.convolve(x, win, mode="same") / sum(win)


def stack_rasters(files, ncpus=32):
    pool = mp.Pool(ncpus)
    arr = np.array(pool.map(open_raster, files))
    pool.close()
    pool.join()
    return arr


def _maskit(x, mask):
    """masking function"""
    x[mask == True] = -9999
    return x
