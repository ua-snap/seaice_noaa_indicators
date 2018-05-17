def get_summer(month):
	return (month == 8) | (month == 9)
def get_winter(month):
	return (month == 1) | (month == 2)

if __name__ == '__main__':
	import xarray as xr
	import numpy as np
	import os, dask
	from functools import partial

	# open the NetCDF that we made...
	fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska.nc'
	ds = xr.open_dataset( fn )

	# slice the data to the full years... currently this is 1979-2016
	ds = ds.sel( time=slice( '1979', '2016' ) )

	# # make climatology? --> 0-366 includes leaps --> this may make more sense to read in as a dask array, compute, write out
	# ds_day_clim = ds.groupby( 'time.dayofyear' ).mean( dim='time' ).compute()

	# get the summer and winter seasons htat were determined in table 1 in the paper
	summer = ds['sic'].sel( time=get_summer( ds[ 'time.month' ] ) )
	winter = ds['sic'].sel( time=get_winter( ds[ 'time.month' ] ) )

	summer_ind = np.where( summer.data )
	winter_ind = np.where( winter.data )

	summer_oob_vals = summer[ summer_ind ]
	winter_oob_vals = winter[ winter_ind ]

	# get the means and standard deviations of these season aggregates
	summer_mean = summer[ 'sic' ].groupby( 'time.year' ).mean( dim='time' )
	summer_std = summer[ 'sic' ].groupby( 'time.year' ).std( dim='time' )
	winter_mean = winter[ 'sic' ].groupby( 'time.year' ).mean( dim='time' )
	winter_std = winter[ 'sic' ].groupby( 'time.year' ).std( dim='time' )

	# [ NOTE ]: to make an ordinal date that matches the somewhat asinine epoch used by matlab in python
	# ordinal_date = date.toordinal(date(1971,1,1)) + 366 
	# if not the number will be 366 days off due to the epoch starting January 0, 0000 whereas in Py Jan 1, 0001.