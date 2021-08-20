"""Make a full array of daily sea ice concentration values

Usage:
    pipenv run python make_daily_timeseries.py -n <number of CPUs>
    Script #3 of data pipeline

Returns:
    Smoothed daily NSIDC-0051 data written to 
    $BASE_PATH/nsidc_0051/smoothed/

Notes:
    Interpolates missing dates linearly
    (2D spatial / 1D profile hann smoothed)
"""

import argparse
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
from functools import partial
from pathlib import Path
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


# # # MULTIPROCESSING APPROACHES TO GENERIC FILTER BUT DONT WORK DUE TO SOME OpenBLAS ISSUE.
# def spatial_smooth( arr, footprint, ncpus=32 ):
#     arr_list = [a.copy() for a in arr] # unpack 3d (time,rows,cols) array to 2d list
#     f = partial( mean_filter_2D, footprint=footprint )
#     pool = mp.Pool( ncpus )
#     out_arr = pool.map( f, arr_list )
#     pool.close()
#     pool.join()
#     return np.array(out_arr)

# def spatial_smooth( arr, size=3, ncpus=32 ):
#     f = partial( mean_filter_2D, size=size )
#     pool = mp.Pool( ncpus )
#     out_arr = pool.map( f, [a for a in arr] )
#     pool.close()
#     pool.join()
#     return np.array(out_arr)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":
    # parse some args
    parser = argparse.ArgumentParser(
        description="stack the hourly outputs from raw WRF outputs to NetCDF files of hourlies broken up by year."
    )
    parser.add_argument(
        "-n",
        "--ncpus",
        action="store",
        dest="ncpus",
        type=int,
        help="number of cpus to use",
    )

    # unpack args
    args = parser.parse_args()
    ncpus = args.ncpus

    # list all data
    base_dir = Path(os.getenv("BASE_DIR"))

    in_dir = base_dir.joinpath("nsidc_0051/prepped")
    fps = sorted(list(in_dir.glob("*")))
    data_times = [make_datetimes(fp.name.split(".")[0].split("_")[1]) for fp in fps]

    # date-fu for filenames and slicing
    begin = data_times[0]
    end = data_times[-1]
    begin_str = begin.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    # stack the irregularly spaced data to a netcdf
    with rio.open(fps[0]) as template:
        meta = template.meta.copy()
        height, width = template.shape

    arr = stack_rasters(fps, ncpus=ncpus)
    ds = make_xarray_dset(arr.copy(), pd.DatetimeIndex(data_times), meta)
    da = ds["sic"].copy()

    # interpolate to daily
    da_interp = da.resample(time="1D").asfreq()

    # get a masks layer from the raw files.  These are all values > 250
    # ------------ ------------ ------------ ------------ ------------ ------------ ------------
    # 251 Circular mask used in the Arctic to cover the irregularly-shaped data
    #       gap around the pole (caused by the orbit inclination and instrument swath)
    # 252 Unused
    # 253 Coastlines
    # 254 Superimposed land mask
    # 255 Missing data
    # make a mask of the known nodata values when we start...
    mask = (arr[0] > 250) & (arr[0] < 300)

    # set masks to nodata
    dat = da_interp.values.copy()

    # make the nodata mask np.nan for computations
    out_masked = []
    for i in dat:
        i[mask] = np.nan
        out_masked = out_masked + [i]

    # put the cleaned up data back into the stacked NetCDF
    da_interp.data = np.array(out_masked)
    da_interp.data = np.apply_along_axis(interpolate, axis=0, arr=da_interp).round(4)

    # spatially smooth the 2-D daily slices of data using a mean generic filter. (without any aggregation)
    print("spatial smooth")
    footprint_type = "queens"
    footprint_lu = {
        "rooks": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        "queens": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    }

    footprint = footprint_lu[footprint_type]

    # 'run using multiprocessing -- YMMV this is a tad flaky at times.'
    # ^^ comment from original author, not sure where flakiness manifested
    # previously but left this note anyway.

    #   Some profiling revealed that paritioning the interpolated array
    # and looping over it with 10 cores on Atlas was about optimal.
    filter_ncpus = 10
    filter_slices = chunkit(da_interp.values.shape[0], filter_ncpus)
    args = [(da_interp.values[sl], footprint) for sl in filter_slices]
    with mp.Pool(filter_ncpus) as pool:
        out = pool.starmap(run_jit_mean_filter, args)

    def _maskit(x, mask):
        """masking function"""
        x[mask == True] = -9999
        return x

    # mask the spatial smoothed outputs with the mask at each 2D slice.
    smoothed = np.array([_maskit(i, mask) for i in np.concatenate(out)]).copy()

    print("hanning smooth")
    n = 3  # perform 3 iterative smooths on the same series
    for i in range(n):
        smoothed = np.apply_along_axis(hanning_smooth, arr=smoothed, axis=0)

    # make sure no values < 0, set to 0
    smoothed[np.where((smoothed < 0) & (~np.isnan(smoothed)))] = 0

    # make sure no values > 1, set to 1
    smoothed[np.where((smoothed > 1) & (~np.isnan(smoothed)))] = 1

    # mask it again to make sure the nodata and land are properly masked following hanning.
    smoothed = np.array([_maskit(i, mask) for i in smoothed]).copy()

    # write it out as a NetCDF
    out_ds = da_interp.copy(deep=True)
    out_ds.values = smoothed.astype(np.float32)
    out_ds = out_ds.to_dataset(name="sic")
    out_ds.attrs = ds.attrs

    # output encoding
    encoding = out_ds.sic.encoding.copy()
    encoding.update({"zlib": True, "comp": 5, "contiguous": False, "dtype": "float32"})
    out_ds.sic.encoding = encoding

    out_fp = in_dir.parent.joinpath(
        f"smoothed/nsidc_0051_sic_{str(begin.year)}-{str(end.year)}_smoothed.nc"
    )
    out_fp.parent.mkdir(exist_ok=True, parents=True)
    out_ds.to_netcdf(out_fp, format="NETCDF4")
