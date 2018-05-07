def get_summer(month):
	return (month == 8) | (month == 9)
def get_winter(month):
	return (month == 1) | (month == 2)
# def mean_filter_2D( arr, footprint ):
# 	from scipy.ndimage import generic_filter
# 	return generic_filter( arr, np.nanmean, footprint=footprint )

if __name__ == '__main__':
	import xarray as xr
	import numpy as np
	import os
	from functools import partial
	# import multiprocessing as mp

	# open the NetCDF that we made...
	fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska.nc'
	ds = xr.open_dataset( fn )

	# slice the data to the full years... currently this is 1979-2016
	ds = ds.sel( time=slice( '1979', '2016' ) )

	# # how to perform a spatial 2D mean filter on the data... (without any aggregation)
	# footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
	# f = partial( mean_filter_2D, footprint=footprint )

	# arr = ds.sic.data.copy()
	# ind = np.where( arr > 1 )
	# arr[ ind ] = np.nan

	# pool = mp.Pool( 32 )
	# done = np.array( pool.map( f, arr ) )
	# pool.close()
	# pool.join()
	# done[ ind ] = ds.sic.data[ ind ]

	# # put the newly smoothed data to the existing xarray object
	# ds['sic'] = (('time', 'yc', 'xc'), done )

	# make climatology?
	# ds_day_clim = ds.groupby( 'time.day' ).mean()


	# get the summer and winter seasons htat were determined in table 1 in the paper
	summer = ds.sel( time=get_summer( ds[ 'time.month' ] ) )
	winter = ds.sel( time=get_winter( ds[ 'time.month' ] ) )

	# get the means and standard deviations of these season aggregates
	summer_mean = summer[ 'sic' ].groupby( 'time.year' ).mean( 'time' )
	summer_std = summer[ 'sic' ].groupby( 'time.year' ).std( 'time' )
	winter_mean = winter[ 'sic' ].groupby( 'time.year' ).mean( 'time' )
	winter_std = winter[ 'sic' ].groupby( 'time.year' ).std( 'time' )