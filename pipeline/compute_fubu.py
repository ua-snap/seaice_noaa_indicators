"""Compute Freeze-Up / Break-Up indicators from smoothed NSIDC-0051 
daily sea ice concentrations

Usage:
    Functions for step #4 of main data pipeline, which creates a 
    CF-compliant FUBU indicators dataset
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing import Pool
from pyproj.crs import CRS


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
    threshold.data[threshold.data < 0.15] = 0.15
    arr = (daily_vals > threshold).values

    # find where the 2week groups are all sic are below the threshold to start breakup
    twoweek_groups_alltrue = [
        arr[(idx - 14) : idx, ...].all(axis=0) for idx in np.arange(times)[14:]
    ]
    # stack to a 3D array/grab a copy
    arr = np.array(twoweek_groups_alltrue).copy()
    ordinal_days_breakup_start = np.apply_along_axis(
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

    return ordinal_days_breakup_start


def breakup_end(ds_sic, summer_mean, summer_std, breakup_start_arr, year, ncpus):
    """Compute the breakup ending date for a given year"""
    daily_vals = ds_sic.sel(time=slice(f"{year}-02-01", f"{year}-09-30")).copy()
    start_ordinalday = int(daily_vals.time.to_index().min().strftime("%j"))
    # handle potential case of end_year not included in input series
    end_ordinalday = int(pd.to_datetime(f"{year}-09-30").strftime("%j"))
    times, rows, cols = daily_vals.shape
    # make a mask
    mask = np.where(np.isnan(daily_vals.isel(time=0).data))
    summer = summer_mean.sel(year=int(year)).copy()
    mean_plus_std = summer + summer_std.copy().sel(year=int(year))
    threshold = mean_plus_std.data.copy()
    # using a higher minimum threshold, 15% was too conservative
    threshold[threshold < 0.5] = 0.5
    arr = (daily_vals < threshold).values
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
    # if it is the last day of the time-window, (sept. 30th) make it NA
    ordinal_days_breakup_end[ordinal_days_breakup_end >= end_ordinalday] = np.nan
    ordinal_days_breakup_end[ordinal_days_breakup_end < start_ordinalday] = np.nan
    # mask it
    ordinal_days_breakup_end[mask] = np.nan

    return ordinal_days_breakup_end


def wrap_fubu(year, ds_sic, summer_mean, summer_std, winter_mean, winter_std, ncpus):
    """Compute the freeze-up/break-up start/end dates for a single year"""
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

    # Require that both start/end metrics be defined, otherwise neither
    freezeup_start_arr[np.isnan(freezeup_end_arr)] = np.nan
    freezeup_end_arr[np.isnan(freezeup_start_arr)] = np.nan
    breakup_start_arr[np.isnan(breakup_end_arr)] = np.nan
    breakup_end_arr[np.isnan(breakup_start_arr)] = np.nan
    
    year_di = {
        "freezeup_start": freezeup_start_arr,
        "freezeup_end": freezeup_end_arr,
        "breakup_start": breakup_start_arr,
        "breakup_end": breakup_end_arr,
    }
    
    # Set nodata value to -9999 and convert to integer for writing
    for indicator in year_di:
        year_di[indicator][np.isnan(year_di[indicator])] = -9999
        year_di[indicator] = year_di[indicator].astype(np.int32)

    print(f"{year} complete.", end=" ")
    
    return year_di


def make_cf_dataset(arr_dict, years, coords):
    """Make an xarray dataset from the FU/BU dates that is CF compliant"""

    xc, yc = (coords["xc"].data, coords["yc"].data)
    
    # global CF-conventions attributes to add when creating DataSet
    global_attrs = {
        "Conventions": "CF-1.8",
        "title": "Arctic sea ice freeze-up and break-up dates derived from passive microwave satellite data, 1979-2018",
        "institution": "Scenarios Network for Alaska and Arctic Planning, International Arctic Research Center, University of Alaska Fairbanks",
        "source": "Data variables were derived from the NSIDC-0051 passive microwave data (https://doi.org/10.5067/8GQ8LZQVL0VL) using methods described in the 'comment' global attribute",
        "comment": "This dataset was developed as an extension of the work presented in Johnson and Eicken (2016; see 'references' global attribute).\n",
        "references": "Mark Johnson, Hajo Eicken; Estimating Arctic sea-ice freeze-up and break-up from the satellite record: A comparison of different approaches in the Chukchi and Beaufort Seas. Elementa: Science of the Anthropocene 1 January 2016; 4 000124. doi: https://doi.org/10.12952/journal.elementa.000124"
    }
    with open("indicators_criteria.txt", mode="r") as f:
        global_attrs["comment"] += f.read()
    
    indicators = arr_dict.keys()
    ds = xr.Dataset(
        {
            indicator: (["year", "yc", "xc"], arr_dict[indicator])
            for indicator in indicators
        },
        coords={"xc": ("xc", xc), "yc": ("yc", yc), "year": years},
        attrs=global_attrs,
    )
    
    # add CRS metadata via grid mapping variable
    grid_mapping_varname = "crs"
    # the data type does not matter, it is just a dummy variable. 
    ds[grid_mapping_varname] = xr.DataArray().astype(np.int32)
    # add CF grid mapping attributes for EPSG 3411 using pyproj.crs.CRS.to_cf()
    ds[grid_mapping_varname].attrs = CRS.from_epsg(3411).to_cf()
    # this value is not provided by to_cf() method but is required in conventions
    ds[grid_mapping_varname].attrs["latitude_of_projection_origin"] = 90.
    # standad names and units for xc and yc according to CF-conventions
    ds["xc"].attrs["standard_name"] = "projection_x_coordinate"
    ds["yc"].attrs["standard_name"] = "projection_y_coordinate"
    for coord_var in ["xc", "yc"]:
        ds[coord_var].attrs["units"] = "m"
        
    # add attributes to indicator data variables
    for indicator in indicators:
        group, status = indicator.split("_")
        indicator_attrs = {
            "_FillValue": -9999,
            "valid_range": [1, 428],
            "grid_mapping": grid_mapping_varname,
            "long_name": f"Day-of-year of {group[:-2].title()}-up {status}",
            "comment": "Units are 'day-of-year'. No 'units' attribute is provided for this variable because UDUNITS does not have support for a 'day-of-year' unit.",
        }
    
    # add info about year discrete axis coordinate
    ds["year"].attrs["long_name"] = "year of indicator observation"
    ds["year"].attrs["comment"] = "Year is given as a discrete axis instead of a time coordinate variable to aid useability. This variable provides the ineger value of the year (standard calendar) in which each indicator was defined in a way that is more straightfoward than representing the year of observation as the 'days since <timestamp>' format with bounds."
        
    return ds
