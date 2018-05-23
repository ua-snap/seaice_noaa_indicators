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

def freezeup_start( ds_sic, summer_mean_annual, summer_std_annual, year ):
	ds_sic_year = ds_sic.sel( time=str(year) )
	nlayers, rows, cols = ds_sic_year.shape
	
	if nlayers == 365:
		leap = False
		start_ordinalday = 244 # this is 9/1 for non-leap year
	else:
		leap = True
		start_ordinalday = 245 # this is 9/1 for leap year

	# the slice to September 1st 'thing' is mapped from Mark's Matlab code. I am choosing to slice to endofyear
	daily_vals = ds_sic_year.sel( time=slice(str(year)+'-09-01', str(year)+'-12-31') )
	mean_plus_std = (summer_mean_annual.sel(year=int(year)) + summer_std_annual.sel(year=int(year)))
	
	# copy the 2D data slice for all locations in map and make a lower threshold of .15 or 15% conc
	threshold = mean_plus_std.data.copy()
	threshold[ threshold < .15 ] = .15
	
	# comparison with the threshold
	arr = daily_vals <= threshold

	# make a 2D mask of np.nan
	mask = np.isnan( daily_vals.isel( time=0 ).data )

	def first_freezeup( x ):
		''' find the first instance of freezing below the threshold along time axis '''
		vals = np.where(np.diff( x.astype( np.int ) ) != 0)[0]
		if vals.shape[0] > 0:
			return vals.min() # get the first one only! for freezeup
		else:
			return -1

	# apply through time
	ordinal_days_freezeup_start = np.apply_along_axis( first_freezeup, axis=0, arr=arr ).astype( np.float32 )
	ordinal_days_freezeup_start[mask] = np.nan

	# handle the off-case where we have a -1 that is outside of the mask.
	if not ordinal_days_freezeup_start.any() == -1:
		output_arr = np.array(ordinal_days_freezeup_start + start_ordinalday)
	else:
		BaseException( 'ERROR!' )
	return output_arr
	
def freezeup_end( ds_sic, winter_mean_annual, freezeup_start_arr, year ):
	# start reading from the freezup start day+1
	start_day_current_year = freezeup_start_arr + 1
	# select current year
	ds_sic_year = ds_sic.sel( time=str(year) )
	winter_mean_annual = winter_mean_annual.sel( year=int(year) )

	def last_freezeup( arr, winter_mean_annual_value ):
		if not np.isnan(winter_mean_annual_value):
			threshold = winter_mean_annual_value - .10 # we are removing 10% from the mean value
			ind = np.where( arr >= threshold )[0][0] # grab the first instance....
		else:
			ind = -1
		return ind

	z,x,y = ds_sic_year.shape
	result = np.zeros((x,y))
	for i,j in np.ndindex(x,y):	
		result[i,j] = last_freezeup( ds_sic_year[:,i,j], winter_mean_annual[i,j] )

	# this would be the day of the year that the free
	freezeup_end_day = result + start_day_current_year
	
	# if calendar.isleap( int(year) ):
	# 	year_ndays = 366
	# else:
	# 	year_ndays = 365

	# # [Not Yet Implemented] handle situations where the end of freezeup straddles another year?
	# if freezeup_end_day.any() > year_ndays:
	# 	freezeup_end_day[ np.where( freezeup_end_day > year_ndays ) ]
	return freezeup_end_day

def wrap_fu_se( ds_sic, summer_mean_annual, summer_std_annual, winter_mean_annual, year ):
	freezeup_start_arr = freezeup_start( ds_sic, summer_mean_annual, summer_std_annual, year )
	freezeup_end_arr = freezeup_end( ds_sic, winter_mean_annual, freezeup_start_arr, year )
	return {'start':freezeup_start_arr, 'end':freezeup_end_arr }

def breakup_start( ds_sic, winter_mean, winter_std, year ):
	ds_sic_year = ds_sic.sel( time=str(year) )
	threshold = (winter_mean.sel( year=int(year) ) - (2*winter_std)).sel( year=int(year) )

	start_ordinalday = 45 # Feb. 14th per Mark's code

	# get a 2D mask, which is common in every layer
	mask = np.isnan( ds_sic_year.isel(time=0).data )
	daily_vals = ds_sic_year.sel( time=slice(str(year)+'-02-14', str(year)+'-08-01') )
	time = daily_vals.time.to_index().to_pydatetime()

	# # # # # TESTING
	# x = daily_vals.data[ :,row,col ]
	# thresh = threshold.data[ row,col ]
	# # # # # END TESTING

	def all_below_thresh_twoweeks_breakup( x, thresh ):
		''' 
		find the first instance of freezing below the threshold along time axis 
		---
		NOTES:
		# goal here is to apply along axis of the daily_vals ds
		# where we want to:
		# 1. iterate through each day and 13 subsequent days (2 weeks total)
		# 	and find when the entire 2 week period is below the given threshold value
		# 2. once found, mark that day as the start of breakup
		# 3. return that ordinal day number.
		'''
		below_thresh = [ idx for idx in range(len(x)) if (x[idx:idx+14] < thresh).all() ]
		if len(below_thresh) > 0:
			return below_thresh[0]
		else:
			return -1
	
	# perform apply_along_axis the hard way...
	z,x,y = daily_vals.shape
	result = np.zeros((x,y))
	for i,j in np.ndindex(x,y):	
		result[i,j] = all_below_thresh_twoweeks_breakup( daily_vals[:,i,j], threshold[i,j] )

	result[ mask ] = np.nan
	ordinal_days_breakup_start = result + start_ordinalday

	if ordinal_days_breakup_start.any() == -1:
		raise BaseException( 'ERROR!' )
		
	return ordinal_days_breakup_start

def breakup_end( ds_sic, summer_mean_annual, summer_std_annual, breakup_start_arr, year ):
	ds_sic_year = ds_sic.sel( time=str(year) )

	mean_plus_std = (summer_mean_annual.sel(year=int(year)) + summer_std_annual.sel(year=int(year)))
	threshold = mean_plus_std.data.copy()
	threshold[ threshold < .15 ] = .15

	arr = ds_sic_year < threshold

	def last_breakup( x ):
		''' find the first instance of freezing below the threshold along time axis '''
		vals = np.where(np.diff( x.astype( np.int ) ) != 0)[0]
		if vals.shape[0] > 0:
			return vals.min() # get the first one only! for freezeup
		else:
			return -1

	ordinal_days_breakup_end = np.apply_along_axis( last_breakup, axis=0, arr=arr ).astype( np.float32 )


if __name__ == '__main__':
	import matplotlib
	matplotlib.use( 'agg' )
	from matplotlib import pyplot as plt
	import xarray as xr
	import numpy as np
	import os, dask
	from functools import partial
	import calendar, datetime

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

	# [ NOTE ]: to make an ordinal date that matches the somewhat asinine epoch used by matlab in python
	# ordinal_date = date.toordinal(date(1971,1,1)) + 366 
	# if not the number will be 366 days off due to the epoch starting January 0, 0000 whereas in Py Jan 1, 0001.

	# for testing...
	summer_mean_annual=summer_mean
	summer_std_annual=summer_std
	winter_mean_annual=winter_mean
	# year = '2012'
	year = 2000

	freezup_start_end_by_year = { year:wrap_fu( ds_sic, summer_mean_annual, summer_std_annual, winter_mean_annual.sel(year=year), str(year) ) for year in years }

	# show it
	output_filename = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/PNG/freezeup_date_test.png'
	show_2D_array_aspatial( freezup_ordinal_day, output_filename )


	# winter_mean_annual = winter_mean.sel(year=int(year))
