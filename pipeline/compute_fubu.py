"""compute Freeze-Up / Break-Up dates from NSIDC-0051 Daily Sea Ice Concentrations

Usage:
    pipenv run python compute_fubu.py -n <number of CPUs> [-b] [-e]
    Script #4 of data pipeline

Returns:
    FUBU dates written to $BASE_DIR/nsidc_0051/outputs/nsidc_0051_<begin(YYYY)>-<end(YYYY)>_fubu.nc
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import xarray as xr
from functools import partial
from multiprocessing import Pool
from pathlib import Path


def get_summer(month):
    return (month == 8) | (month == 9)


def get_winter(month):
    return (month == 1) | (month == 2)


def get_doy(x, start_ordinalday, first=True):
    """Return the doy of first or last in boolean array"""
    (vals,) = np.where(x == True)
    if len(vals) > 0:
        if first:
            return np.nanmin(vals) + start_ordinalday
        else:
            return np.nanmax(vals) + start_ordinalday
    else:
        return np.nan


def freezeup_start(ds_sic, summer_mean, summer_std, year):
    """Find first instance of sic exceeding summer mean + 1 std dev"""

    # def first_freezeup(x, start_ordinalday):
    #     """Return doy of first instance"""
    #     (vals,) = np.where(x == True)
    #     # get the first one only! for freezeup
    #     if len(vals) > 0:
    #         return vals.min() + start_ordinalday
    #     else:
    #         return np.nan  # -9999

    # Search 9/1 to 1/31 of following year
    daily_vals = ds_sic.sel(
        time=slice(f"{year}-09-01", f"{int(year) + 1}-01-31")
    ).copy()
    start_ordinalday = int(daily_vals.time.to_index().min().strftime("%j"))
    # threshold and set minimum value to .15 or 15% conc
    smean = summer_mean.sel(year=int(year)).copy()
    mean_plus_std = smean + summer_std.sel(year=int(year)).copy()
    threshold = mean_plus_std.data.copy()
    threshold[threshold < 0.15] = 0.15
    # make a 2D mask of np.nan
    mask = np.isnan(daily_vals.isel(time=0).data)
    # when does the daily sic exceed the threshold sic
    arr = (daily_vals >= threshold).values
    # apply through time
    ordinal_days_freezeup_start = np.apply_along_axis(
        # first_freezeup, axis=0, arr=arr, start_ordinalday=start_ordinalday
        get_doy,
        axis=0,
        arr=arr,
        start_ordinalday=start_ordinalday,
    ).astype(np.float32)
    ordinal_days_freezeup_start[mask] = np.nan
    # if the Aug-Sept mean/threshold is >25%, then that date is NAN. [from Mark/Hajo]
    ordinal_days_freezeup_start[smean > 0.25] = np.nan
    return ordinal_days_freezeup_start


def find_indicator_end_day(sic_bool_arr, indicator_start, first_day):
    """Determine the ordinal day of freezeup or breakup end, i.e. the first
    day where full freezeup or breakup occurs for some fixed threshold.

    Args:
        sic_bool_arr (numpy.ndarray): Boolean array for whether the SIC
            exceeded or dropped below the SIC threshold for the duration 
            threshold, depending on the indicator being calcualted
        indicator_start (int): ordinal day of freezeup or breakup start for the pixel
        first_day (int): first ordinal day of the search window

    Returns:
        (int) the ordinal day of the last freezeup/breakup
    
    Notes:
        Defined globally for use with Pool.
        Designed to be called on a single pixel.
    """
    # indices of potential freezeup/breakup end dates
    # I.e., sic_bool_arr == True where the subsequent x days
    # were above/below the freezeup/breakup end SIC threshold,
    # where x is the minimum duration in days for freezeup/breakup
    (end_idx,) = np.where(sic_bool_arr)
    if not np.isnan(indicator_start):
        # only consider values that are > the indicator start day
        end_idx = end_idx[np.where((end_idx + first_day) > indicator_start)]
    else:
        return np.nan
    # only consider values if pixel is not > threshold for entire period
    if (len(end_idx) > 0) and (len(end_idx) < len(sic_bool_arr)):
        minval = end_idx.min()
        end_day = minval + first_day
    else:
        end_day = np.nan

    return end_day


def freezeup_end(ds_sic, winter_mean, freezeup_start_arr, year, ncpus):
    """Find the first instance of sic exceeding the winter mean threshold"""
    # select the current year values from Sept. 1 - Feb. 28
    daily_vals = ds_sic.sel(
        time=slice(f"{year}-09-01", f"{int(year) + 1}-02-28")
    ).copy()

    start_ordinalday = int(daily_vals.time.to_index().min().strftime("%j"))
    times, nrows, ncols = daily_vals.shape
    # make a np.nan mask
    mask = np.isnan(daily_vals.isel(time=0).values)
    # remove 10% sic from winter mean values, set min threshold to 15%
    threshold = winter_mean.sel(year=int(year) + 1).copy().values - 0.10
    threshold[threshold < 0.15] = 0.15
    # lower threshold to 50% if above
    threshold[threshold > 0.5] = 0.5
    arr = daily_vals.values > threshold

    # find freezeup end date.
    # ordinal_days_freezeup_end = np.zeros_like(arr[0, ...].astype(np.float32))
    # for i, j in np.ndindex(arr.shape[-2:]):
    #     ordinal_days_freezeup_end[i, j] = last_freezeup(
    #         arr[:, i, j], freezeup_start_arr[i, j], start_ordinalday
    #     )

    # this is for finding the first date for which the following two weeks are above
    #   the threshold
    freeze_bool_arr = np.array(
        [arr[idx : (idx + 14), ...].all(axis=0) for idx in np.arange(times)[:-14]]
    )

    # make args for Pool-ing
    dims = freeze_bool_arr.shape[-2:]
    args = [
        (freeze_bool_arr[:, i, j], freezeup_start_arr[i, j], start_ordinalday)
        for i, j in np.ndindex(dims)
    ]
    with Pool(ncpus) as pool:
        fu_end = pool.starmap(find_indicator_end_day, args)

    ordinal_days_freezeup_end = np.array(fu_end).reshape(dims).astype(np.float32)
    ordinal_days_freezeup_end[mask] = np.nan

    return ordinal_days_freezeup_end


def breakup_start(ds_sic, winter_mean, winter_std, summer_mean, year):
    """Find the day that breakup starts"""
    # % "last day for which previous two weeks are above threshold -2std"
    # get daily values for the time range in Mark's Algo (Feb.1 is Feb.14 - 14days search window)
    daily_vals = ds_sic.sel(time=slice(f"{year}-02-01", f"{year}-08-01")).copy()
    start_ordinalday = int(daily_vals.time.to_index().min().strftime("%j")) + 13
    # this is for determining the date if it goes beyond 8/1
    last_ordinalday = int(daily_vals.time.to_index()[-1].strftime("%j"))
    end_ordinalday = int(pd.to_datetime(f"{str(year)}-08-01").strftime("%j"))  # 8/1
    times, rows, cols = daily_vals.shape
    # make mask
    mask = np.isnan(daily_vals.isel(time=0).data)
    # threshold data
    threshold = (
        winter_mean.sel(year=int(year)) - (2 * winter_std.sel(year=int(year)))
    ).copy(deep=True)
    arr = (daily_vals > threshold).values

    # def alltrue(x):
    #     return x.all()  # skips np.nan

    # def wherefirst(x, start_ordinalday):
    #     (vals,) = np.where(x == True)
    #     if len(vals) > 0:
    #         return vals[~np.isnan(vals)].max() + start_ordinalday
    #     else:
    #         return np.nan  # -9999

    # find where the 2week groups are all sic are below the threshold to start breakup
    twoweek_groups_alltrue = [
        arr[(idx - 14) : idx, ...].all(axis=0) for idx in np.arange(times)[14:]
    ]
    # stack to a 3D array/grab a copy
    arr = np.array(twoweek_groups_alltrue).copy()
    # ordinal_day_first_breakup_twoweeks
    ordinal_days_breakup_start = np.apply_along_axis(
        # wherefirst, arr=arr, axis=0, start_ordinalday=start_ordinalday
        get_doy,
        arr=arr,
        axis=0,
        start_ordinalday=start_ordinalday,
        first=False,
    ).astype(np.float32)
    ordinal_days_breakup_start[mask] = np.nan  # mask it
    # other conditions from the old code:
    #  if summer means are greater than 40% we are making it NO BREAKUP
    summer = summer_mean.sel(year=int(year)).copy(deep=True)
    ordinal_days_breakup_start[np.where(summer > 0.40)] = np.nan
    # if the day chosen is the last day of the time window (or beyond), make it NA <-- new from Hajo
    ordinal_days_breakup_start[
        ordinal_days_breakup_start + 1 == end_ordinalday
    ] = np.nan
    # ordinal_days_breakup_start[ ordinal_days_breakup_start == end_ordinalday ] = np.nan
    return ordinal_days_breakup_start


def breakup_end(ds_sic, summer_mean, summer_std, breakup_start_arr, year, ncpus):
    """Compute the breakup ending date for a given year"""
    # % find the last day where conc exceeds summer value plus 1std
    # set a minimum threshold of 50%

    # def last_breakup(x, start_ordinalday):
    #     """Find the last instance in time where sic exceeds threshold"""
    #     (vals,) = np.where(x == True)
    #     # add a 1 to the index since it is zero based and we are using it as a way to increment days
    #     if vals.shape[0] > 0:
    #         return vals.max() + start_ordinalday  # add ordinal start day to the index
    #     else:
    #         return np.nan  # -9999

    daily_vals = ds_sic.sel(time=slice(f"{year}-02-01", f"{year}-09-30")).copy()
    #         deep=True
    #     )
    start_ordinalday = int(daily_vals.time.to_index().min().strftime("%j"))
    # handle potential case of end_year not included in input series
    end_ordinalday = int(pd.to_datetime(f"{year}-09-30").strftime("%j"))
    times, rows, cols = daily_vals.shape
    # make a mask
    mask = np.where(np.isnan(daily_vals.isel(time=0).data))
    summer = summer_mean.sel(year=int(year)).copy()
    mean_plus_std = summer + summer_std.copy().sel(year=int(year))
    threshold = mean_plus_std.data.copy()
    # threshold[threshold < 0.15] = 0.15
    # need different threshold, too conservative
    threshold[threshold < 0.5] = 0.5
    # arr = (daily_vals > threshold).values
    arr = (daily_vals < threshold).values
    # find
    # for determining if the following two weeks are BELOW the threshold
    # testing definition of indicator being the first day where the following two
    #   weeks are below the threshold
    melt_bool_arr = np.array(
        [arr[idx : (idx + 14), ...].all(axis=0) for idx in np.arange(times)[:-14]]
    )
    # make args for Pool-ing
    dims = melt_bool_arr.shape[-2:]
    args = [
        (melt_bool_arr[:, i, j], breakup_start_arr[i, j], start_ordinalday)
        for i, j in np.ndindex(dims)
    ]
    with Pool(ncpus) as pool:
        bu_end = pool.starmap(find_indicator_end_day, args)

    ordinal_days_breakup_end = np.array(bu_end).reshape(dims).astype(np.float32)
    ordinal_days_breakup_end[mask] = np.nan

    # # stack to a 3D array/grab a copy
    # arr = np.array(twoweek_groups_alltrue).copy()

    # ordinal_days_breakup_end = np.apply_along_axis(
    #     # last_breakup, axis=0, arr=arr, start_ordinalday=start_ordinalday
    #     get_doy,
    #     axis=0,
    #     arr=arr,
    #     start_ordinalday=start_ordinalday,
    #     # first=False
    # ).astype(np.float32)
    # # if the summer mean is greater than 25% make it NA
    # # ordinal_days_breakup_end[summer.values > 0.25] = np.nan
    # # if all two week periods are below threshold make it NA
    # ordinal_days_breakup_end[arr.all(axis=0)] = np.nan

    # if it is the last day of the time-window, (sept. 30th) make it NA
    ordinal_days_breakup_end[ordinal_days_breakup_end >= end_ordinalday] = np.nan
    ordinal_days_breakup_end[ordinal_days_breakup_end < start_ordinalday] = np.nan
    # require breakup_start be defined
    # ordinal_days_breakup_end[np.isnan(breakup_start_arr)] = np.nan
    # mask it
    ordinal_days_breakup_end[mask] = np.nan

    return ordinal_days_breakup_end


def wrap_fubu(year, ds_sic, summer_mean, summer_std, winter_mean, winter_std, ncpus):
    """
    wrapper function to run an entire year in one process
    * this is mainly used as a wrapper function for multiprocessing.
    """
    freezeup_start_arr = freezeup_start(ds_sic, summer_mean, summer_std, year)
    freezeup_end_arr = freezeup_end(
        ds_sic, winter_mean, freezeup_start_arr, year, ncpus
    )
    breakup_start_arr = breakup_start(
        ds_sic, winter_mean, winter_std, summer_mean, year
    )
    breakup_end_arr = breakup_end(
        ds_sic, summer_mean, summer_std, breakup_start_arr, year, ncpus
    )

    # Require that both metrics be defined, otherwise neither
    freezeup_start_arr[np.isnan(freezeup_end_arr)] = np.nan
    freezeup_end_arr[np.isnan(freezeup_start_arr)] = np.nan
    breakup_start_arr[np.isnan(breakup_end_arr)] = np.nan
    breakup_end_arr[np.isnan(breakup_start_arr)] = np.nan

    print(f"{year} complete.")
    return {
        "freezeup_start": freezeup_start_arr,
        "freezeup_end": freezeup_end_arr,
        "breakup_start": breakup_start_arr,
        "breakup_end": breakup_end_arr,
    }


def make_avg_ordinal(fubu, years, metric):
    """
    metrics = ['freezeup_start','freezeup_end','breakup_start','breakup_end']
    [ NOTE ]: we are masking the -9999 values for these computations...  which will
                most likely need to change for proper use.
    """
    # stack it
    arr = np.array([fubu_years[year][metric] for year in years])
    mask = np.isnan(arr[0]).copy()
    # arr = np.ma.masked_where(arr == -9999, arr, copy=True)
    arr = np.rint(np.ma.mean(arr, axis=0)).data
    arr[mask] = np.nan
    return arr.astype(np.float32)


def make_xarray_dset_years(arr_dict, years, coords, transform):
    """ make a NetCDF file output computed metrics for FU/BU in a 3-D variable-based way """

    xc, yc = (coords["xc"], coords["yc"])
    attrs = {
        "proj4string": "EPSG:3411",
        "proj_name": "NSIDC North Pole Stereographic",
        "affine_transform": transform,
    }

    ds = xr.Dataset(
        {
            metric: (["year", "yc", "xc"], arr_dict[metric])
            for metric in arr_dict.keys()
        },
        coords={"xc": ("xc", xc), "yc": ("yc", yc), "year": years},
        attrs=attrs,
    )
    return ds


def make_xarray_dset_mean(arr_dict, coords, transform):
    """ make a NetCDF file output computed metrics for FU/BU in a 3-D variable-based way """

    xc, yc = (coords["xc"], coords["yc"])
    attrs = {
        "proj4string": "EPSG:3411",
        "proj_name": "NSIDC North Pole Stereographic",
        "affine_transform": transform,
    }

    ds = xr.Dataset(
        {
            metric: (["yc", "xc"], arr_dict[metric].squeeze())
            for metric in arr_dict.keys()
        },
        coords={"xc": ("xc", xc), "yc": ("yc", yc)},
        attrs=attrs,
    )
    return ds


if __name__ == "__main__":
    print("program start")
    tic = time.perf_counter()

    # parse some args
    parser = argparse.ArgumentParser(
        description="compute freezeup/breakup dates from NSIDC-0051 prepped dailies"
    )
    parser.add_argument(
        "-b",
        "--begin",
        action="store",
        dest="begin",
        type=str,
        default="1979",
        help="beginning year of the climatology",
    )
    parser.add_argument(
        "-e",
        "--end",
        action="store",
        dest="end",
        type=str,
        default="2019",
        help="ending year of the climatology",
    )
    parser.add_argument(
        "-n",
        "--ncpus",
        action="store",
        dest="ncpus",
        type=int,
        help="number of cpus to use",
    )

    # unpack the args here...  It is just cleaner to do it this way...
    args = parser.parse_args()
    begin = args.begin
    end = args.end
    ncpus = args.ncpus

    base_dir = Path(os.getenv("BASE_DIR"))
    sic_fp = base_dir.joinpath(
        "nsidc_0051/smoothed/nsidc_0051_sic_1978-2019_smoothed.nc"
    )

    np.warnings.filterwarnings("ignore")  # filter annoying warnings.

    # # open the NetCDF that we made...
    ds = xr.load_dataset(
        sic_fp
    )  # load it so it processes a LOT faster plus it is small...

    # slice the data to the full years... currently this is 1979-20**
    ds_sic = ds.sel(time=slice(begin, end))["sic"]
    years = (
        ds_sic.time.to_index().map(lambda x: x.year).unique().tolist()[:-1]
    )  # cant compute last year

    # set all nodata pixels to np.nan
    ds_sic.data[ds_sic.data > 1] = np.nan

    # make a no data mask
    mask = np.isnan(ds_sic.isel(time=0).data)

    # get the summer and winter seasons that were determined in table 1 in the paper
    summer = ds_sic.sel(time=get_summer(ds_sic["time.month"]))
    winter = ds_sic.sel(time=get_winter(ds_sic["time.month"]))

    # get the means and standard deviations of these season aggregates
    summer_mean = summer.groupby("time.year").mean(dim="time").round(4)
    summer_std = summer.groupby("time.year").std(dim="time", ddof=1).round(4)
    winter_mean = winter.groupby("time.year").mean(dim="time").round(4)
    winter_std = winter.groupby("time.year").std(dim="time", ddof=1).round(4)

    f = partial(
        wrap_fubu,
        ds_sic=ds_sic,
        summer_mean=summer_mean,
        summer_std=summer_std,
        winter_mean=winter_mean,
        winter_std=winter_std,
        ncpus=ncpus,
    )
    # [FIX ME] run parallel (notworking due to an issue that I think is solved in Py3.7.*)
    # with mp.Pool as
    # pool = mp.Pool( ncpus )
    # fubu_years = dict(zip(years, pool.map(f, years)))
    # pool.close()
    # pool.join()

    # run serial
    fubu_years = {year: f(year) for year in years}

    # # make NC file with metric outputs as variables and years as the time dimension
    # # --------- --------- --------- --------- --------- --------- --------- ---------
    # stack into arrays by metric
    transform = ds.affine_transform
    metrics = ["freezeup_start", "freezeup_end", "breakup_start", "breakup_end"]
    stacked = {
        metric: np.array([fubu_years[year][metric] for year in years])
        for metric in metrics
    }
    ds_fubu = make_xarray_dset_years(stacked, years, ds.coords, transform)
    out_fp = base_dir.joinpath(f"nsidc_0051/outputs/nsidc_0051_{begin}-{end}_fubu.nc")
    out_fp.parent.mkdir(exist_ok=True)

    # out_fn = os.path.join(
    #     base_path,
    #     "outputs",
    #     "NetCDF",
    #     "nsidc_0051_sic_nasateam_{}-{}{}_north_smoothed_fubu_dates.nc".format(
    #         str(begin), str(end), suffix
    #     ),
    # )

    # dump to disk
    ds_fubu.to_netcdf(out_fp, format="NETCDF4")

    # make averages in ordinal day across all fu/bu
    metrics = ["freezeup_start", "freezeup_end", "breakup_start", "breakup_end"]
    averages = {
        metric: make_avg_ordinal(fubu_years, years, metric) for metric in metrics
    }

    # write the average dates (clim) to a NetCDF
    ds_fubu_avg = make_xarray_dset_mean(averages, ds.coords, transform)

    mean_out_fp = str(out_fp).replace("fubu.nc", "fubu_mean.nc")

    # output_filename = os.path.join(
    #     base_path,
    #     "outputs",
    #     "NetCDF",
    #     "nsidc_0051_sic_nasateam_{}-{}{}_north_smoothed_fubu_dates_mean.nc".format(
    #         begin, end, suffix
    #     ),
    # )
    # make the output dir if needed:

    ds_fubu_avg.to_netcdf(mean_out_fp, format="NETCDF4")

    print(f"program finished, duration: {round((time.perf_counter() - tic) / 60, 1)}m")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# SOME NOTES ABOUT TRANSLATING FROM MLAB :
# ----------------------------------------
# [ 1 ]: to make an ordinal date that matches the epoch used by matlab in python
# ordinal_date = date.toordinal(date(1971,1,1)) + 366
# if not the number will be 366 days off due to the epoch starting January 0, 0000 whereas in Py Jan 1, 0001.
# [ 2 ]: when computing stdev it is important to set the ddof=1 which is the matlab default.  Python leaves it at 0 default.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #