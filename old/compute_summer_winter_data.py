def show_2D_array_aspatial( arr, output_filename ):
	img = plt.imshow( arr, interpolation='nearest' )
	plt.colorbar( img ) 
	plt.savefig( output_filename )
	plt.close()
	return output_filename

def convert_ordinalday_year_to_datetime( year, ordinal_day ):
	'''
	ordinal_day = [int] range(1, 365+1) or range(1, 366+1) # if leapyear
	year = [int] four digit date that can be read in datetime.date
	'''
	return datetime.date( year=year, month=1, day=1 ) + datetime.timedelta( ordinal_day - 1 )

def get_summer(month):
	return (month == 8) | (month == 9)

def get_winter(month):
	return (month == 1) | (month == 2)

def freezeup_start( ds_sic, summer_mean, summer_std, year ):
	''' find the first instance of sic exceeding the summer mean + 1 standard deviation threshold '''

	if calendar.isleap( int(year) ):
		start_ordinalday = 244 # this is 9/1 for non-leap year
	else:
		start_ordinalday = 245 # this is 9/1 for leap year

	# the slice to September 1st 'thing' is mapped from Mark's Matlab code. I am choosing to slice to endofyear...
	# [ML NOTE] -- I think this is hairy and since we do find that there are start days of Sept1, it wont work for all Arctic
	daily_vals = ds_sic.sel( time=slice(str(year)+'-09-01', str(year)+'-12-31') ).copy()
	mean_plus_std = (summer_mean.sel(year=int(year)).copy() + summer_std.sel(year=int(year)).copy())
	
	# copy the 2D data slice for all locations in map and make a lower threshold of .15 or 15% conc
	threshold = mean_plus_std.data.copy()
	threshold[ threshold < .15 ] = .15

	# make a 2D mask of np.nan
	mask = np.isnan( daily_vals.isel( time=0 ).data )
	
	# when does the daily sic exceed (or is equal-to) the threshold sic
	arr = daily_vals >= threshold

	def first_freezeup( x ):
		''' find the first instance of sic exceeding the threshold '''
		vals, = np.where( x == True )
		if vals.shape[0] > 0:
			return vals.min() # get the first one only! for freezeup
		else:
			return -9999

	# apply through time
	ordinal_days_freezeup_start_index = np.apply_along_axis( first_freezeup, axis=0, arr=arr ).astype( np.float32 )
	ordinal_days_freezeup_start_index[mask] = np.nan
	# add our index to the ordinal day start (Sept.1)
	ordinal_days_freezeup_start = ordinal_days_freezeup_start_index + start_ordinalday

	# handle the off-case where we have a -9999 that is outside of the mask.
	if np.any(ordinal_days_freezeup_start[~mask] < 0):
		raise ValueError( 'ERROR! -- {}'.format(year) )

	return ordinal_days_freezeup_start
	
def freezeup_end( ds_sic, winter_mean, freezeup_start_arr, year ):
	''' find the first instance of sic exceeding the winter mean threshold '''

	if calendar.isleap( int(year) ):
		start_ordinalday = 244 # this is 9/1 for non-leap year
	else:
		start_ordinalday = 245 # this is 9/1 for leap year

	# select the current year values from Sept.1 - Dec.31
	daily_vals = ds_sic.sel( time=slice(str(year)+'-09-01',str(year)+'-12-31') ).copy()

	# make a mask from the nan's
	mask = np.isnan( daily_vals.isel( time=0 ).data )

	# remove 10% sic from winter mean values
	winter_mean_year = winter_mean.sel( year=int(year) ).copy() - .10

	arr = daily_vals >= winter_mean_year

	def last_freezeup( x ):
		vals, = np.where( x == True )
		if vals.shape[0] > 0:
			return vals.min() # grab the first instance
		else:
			return -9999

	ordinal_days_freezeup_end_index = np.apply_along_axis( last_freezeup, axis=0, arr=arr ).astype( np.float32 )
	ordinal_days_freezeup_end_index[ mask ] = np.nan # mask it
	ordinal_days_freezeup_end = ordinal_days_freezeup_end_index + start_ordinalday

	# get index of locations where freezeup start and end are the same ordinal day
	end_same_as_start = np.where(ordinal_days_freezeup_end[~mask] == freezeup_start_arr[~mask])

	# # increment our freezeup end values that are the same date as the freezeup start values by 1
	# update the values where they are the same
	ordinal_days_freezeup_end[~mask][end_same_as_start] = ordinal_days_freezeup_end[~mask][end_same_as_start] + 1

	# handle the off-case where we have a -9999 that is outside of the mask.
	if np.any(ordinal_days_freezeup_end[~mask] < 0):
		raise ValueError( 'ERROR! -- {}'.format(year) )
	
	return ordinal_days_freezeup_end

def breakup_start( ds_sic, winter_mean, winter_std, year ):
	''' find the day that breakup starts '''

	start_ordinalday = 45 # Feb. 14th per Mark's code -- no leap day math needed. 

	# get daily values for the time range in Mark's Algo
	daily_vals = ds_sic.sel( time=slice(str(year)+'-02-14', str(year)+'-08-01') ).copy()
	time = daily_vals.time.to_index().to_pydatetime()
	
	threshold = (winter_mean.sel( year=int(year) ).copy() - (2*winter_std)).sel( year=int(year) ).copy()
	
	# get a 2D mask, which is common in every layer
	mask = np.isnan( daily_vals.isel(time=0).data )

	def alltrue( x ):
		return x.all()

	def wherefirst( x ):
		vals, = np.where( x > 0 )
		if len(vals) > 0:
			return vals.min()
		else:
			return -9999

	times,rows,cols = daily_vals.shape
	# find where the 2week groups are all sic are below the threshold to start breakup
	twoweek_groups_alltrue = [np.apply_along_axis(alltrue, axis=0, arr=(daily_vals[idx:idx+14,...] < threshold) ).astype(int) for idx in range(times)]
	
	arr = np.array( twoweek_groups_alltrue ) # stack to a 3D array

	# ordinal_day_first_breakup_twoweeks
	ordinal_days_breakup_start_index = np.apply_along_axis( wherefirst, arr=arr, axis=0 ).astype( np.float32 )
	ordinal_days_breakup_start_index[ mask ] = np.nan # mask it
	ordinal_days_breakup_start = ordinal_days_breakup_start_index + start_ordinalday

	# handle the off-case where we have a -9999 that is outside of the mask.
	if np.any(ordinal_days_breakup_start[~mask] < 0):
		raise ValueError( 'ERROR! -- {}'.format(year) )
		
	return ordinal_days_breakup_start

def breakup_end( ds_sic, summer_mean, summer_std, year ):
	if calendar.isleap( int(year) ):
		start_ordinalday = 152 # June 1
	else:
		start_ordinalday = 153 # June 1

	daily_vals = ds_sic.sel( time=slice(str(year)+'-06-01', str(year)+'-09-30') ).copy()

	# make a mask
	mask = np.isnan( daily_vals.isel(time=0).data )

	mean_plus_std = (summer_mean.sel(year=int(year)) + summer_std.sel(year=int(year)))
	threshold = mean_plus_std.data.copy()
	threshold[ threshold < .15 ] = .15

	arr = daily_vals < threshold

	def last_breakup( x ):
		''' find the last instance lessthan below the threshold along time axis '''
		vals, = np.where( np.diff( x.astype(int) ) != 0 )
		# add a 1 to the index since it is zero based and we are using it as a way to increment days
		if vals.shape[0] > 0:
			return vals.max()+1 # get the first one only! for freezeup
		else:
			return -9999

	ordinal_days_breakup_end_index = np.apply_along_axis( last_breakup, axis=0, arr=arr ).astype( np.float32 )
	ordinal_days_breakup_end_index[ mask ] = np.nan
	ordinal_days_breakup_end = ordinal_days_breakup_end_index + start_ordinalday

	if np.any(ordinal_days_breakup_end[~mask] < 0):
		raise ValueError( 'ERROR! -- {}'.format(year) )
	
	return ordinal_days_breakup_end

def wrap_fubu( year, ds_sic, summer_mean, summer_std, winter_mean, winter_std ):
	freezeup_start_arr = freezeup_start( ds_sic, summer_mean, summer_std, year )
	freezeup_end_arr = freezeup_end( ds_sic, winter_mean, freezeup_start_arr, year )
	breakup_start_arr = breakup_start( ds_sic, winter_mean, winter_std, year )
	breakup_end_arr = breakup_end( ds_sic, summer_mean, summer_std, year )
	return {'freezeup_start':freezeup_start_arr, 'freezeup_end':freezeup_end_arr, 
			'breakup_start':breakup_start_arr, 'breakup_end':breakup_end_arr }

def make_avg_ordinal( fubu, years, metric ):
	''' 
	metrics = ['freezeup_start','freezeup_end','breakup_start','breakup_end']
	'''
	return np.rint( np.mean(np.array([ fubu_years[year][metric] for year in years ]), axis=0) )



if __name__ == '__main__':
	import matplotlib
	matplotlib.use( 'agg' )
	from matplotlib import pyplot as plt
	import xarray as xr
	import numpy as np
	import os, dask
	from functools import partial
	import calendar, datetime
	import multiprocessing as mp
	import rasterio

	# open the NetCDF that we made...
	fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska.nc'
	ds = xr.open_dataset( fn ).load()

	# slice the data to the full years... currently this is 1979-2016
	ds_sic = ds.sel( time=slice( '1979', '2016' ) )['sic']
	years = range( 1979, 2016 )

	# set all nodata pixels to np.nan
	ds_sic.data[ ds_sic.data > 1 ] = np.nan

	# make a no data mask
	mask = np.isnan( ds_sic.isel(time=0).data )

	# # make climatology? --> 0-366 includes leaps --> this may make more sense to read in as a dask array, compute, write out
	# ds_day_clim = ds.groupby( 'time.dayofyear' ).mean( dim='time' ).compute()

	# get the summer and winter seasons htat were determined in table 1 in the paper
	summer = ds_sic.sel( time=get_summer( ds_sic[ 'time.month' ] ) )
	winter = ds_sic.sel( time=get_winter( ds_sic[ 'time.month' ] ) )

	# get the means and standard deviations of these season aggregates
	summer_mean = summer.groupby( 'time.year' ).mean( dim='time' )
	summer_std = summer.groupby( 'time.year' ).std( dim='time', ddof=1 )
	winter_mean = winter.groupby( 'time.year' ).mean( dim='time' )
	winter_std = winter.groupby( 'time.year' ).std( dim='time', ddof=1 )

	# run it. -- maybe make this parallel?
	f = partial( wrap_fubu, ds_sic=ds_sic, summer_mean=summer_mean, summer_std=summer_std, winter_mean=winter_mean, winter_std=winter_std )
	pool = mp.Pool( 32 )
	fubu_years = dict(zip(years,pool.map(f, years)))
	pool.close()
	pool.join()

	# make a mask 
	mask = np.isnan( ds_sic.isel( time=0 ).data )

	# make averages in ordinal day across all fu/bu
	metrics = ['freezeup_start','freezeup_end','breakup_start','breakup_end']  
	averages = { metric:make_avg_ordinal( fubu_years, years, metric) for metric in metrics }

	# # show it not spatially warped... -- for testing....
	output_filename = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/PNG/freezup_avg_allyears_ordinal.png'
	show_2D_array_aspatial( averages['freezeup_end'], output_filename )

	# Make a raster with the outputs
	template_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/GTiff/alaska/1981/nt_19810101_n07_v1-1_n.tif'
	with rasterio.open( template_fn ) as tmp:
		meta = tmp.meta

	for metric in metrics:
		arr = averages[ metric ]
		output_filename = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/PNG/{}_avg_allyears_ordinal.tif'.format(metric)
		meta.update( compress='lzw', count=1 )
		with rasterio.open( output_filename, 'w', **meta ) as out:
			out.write( arr, 1 )

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# SOME NOTES ABOUT TRANSLATING FROM MLAB :
	# ----------------------------------------
	# [ 1 ]: to make an ordinal date that matches the somewhat asinine epoch used by matlab in python
	# ordinal_date = date.toordinal(date(1971,1,1)) + 366 
	# if not the number will be 366 days off due to the epoch starting January 0, 0000 whereas in Py Jan 1, 0001.
	# [ 2 ]: when computing stdev it is important to set the ddof=1 which is the matlab default.  Python leaves it at 0 default.
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 