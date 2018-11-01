# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#     make the figure 5/6 from Hajo and Marks paper.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import xarray as xr
import pandas as pd
import geopandas as gpd
from affine import Affine
import numpy as np
import rasterio, os, calendar, datetime
import argparse

# parse some args
parser = argparse.ArgumentParser( description='stack the hourly outputs from raw WRF outputs to NetCDF files of hourlies broken up by year.' )
parser.add_argument( "-b", "--base_path", action='store', dest='base_path', type=str, help="input hourly directory containing the NSIDC_0051 data converted to GTiff" )
parser.add_argument( "-w", "--window_len", action='store', dest='window_len', type=int, help="window length to add to the output NetCDF file name" )

# unpack args
args = parser.parse_args()
base_path = args.base_path
window_len = args.window_len

# handle custom hann
if window_len == 1:
	window_len = 'paper_weights'

# data
# window_len = 4
# netcdf_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska.nc'
netcdf_fn = os.path.join( base_path, 'NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}.nc'.format(window_len) )
ds = xr.open_dataset( netcdf_fn )
a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

# make barrow points and get their row/col locs
points_fn = os.path.join( base_path,'selection_points','barrow_points.shp' )
points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
colrows = [ ~a*pt for pt in points ]
colrows = [ (int(c),int(r)) for c,r in colrows ]
cols = [c for c,r in colrows]
rows = [r for c,r in colrows]

# make a climatology
clim_fn = netcdf_fn.replace( '.nc', '_climatology.nc' )
if not os.path.exists( clim_fn ):
	clim_sel = ds.sel( time=slice('1978','2013') )
	clim = clim_sel.groupby('time.dayofyear').mean('time')
	clim.to_netcdf( clim_fn )
else:
	clim = xr.open_dataset( clim_fn )

for sl in [slice('1982-09-01','1986-09-30'), slice('2007-09-01','2012-09-30')]:
	ds_sel = ds.sel( time=sl )
	hold = [ ds_sel.sic[:,r,c].values for c,r in colrows ]
	annual_dat = pd.Series( np.mean(hold, axis=0), ds_sel.time.to_index() )

	clim_repeat = xr.concat([ clim if calendar.isleap(i) else clim.drop(60,'dayofyear') for i in range(int(sl.start.split('-')[0]), int(sl.stop.split('-')[0])+1)], dim='dayofyear')['sic']
	clim_repeat_sel = clim_repeat.isel(dayofyear=slice((244-1)-1,-122+(30-1))) # double -1's is for 0 index and one for leapday

	clim_hold = [ clim_repeat_sel[:,r,c].values for c,r in colrows ]
	clim_mean = np.mean( clim_hold, axis=0 )
	
	fubu_fn = os.path.join( base_path,'NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}_fubu_dates.nc'.format(window_len) )
	fubu_ds = xr.open_dataset( fubu_fn )
	begin_year,end_year = sl.start.split('-')[0], sl.stop.split('-')[0]
	fubu_ds_sel = fubu_ds.sel(year=slice(begin_year, end_year))
	years = list(range(int(begin_year), int(end_year)+1))

	metrics = [ 'freezeup_start','freezeup_end','breakup_start','breakup_end' ]
	fubu = {}
	for metric in metrics:
		year_dict = {}
		for year in years:
			arr = fubu_ds[metric].sel(year=year)[rows,cols]
			arr.values[ np.where(arr.values == -9999) ] = np.nan
			year_dict[year] = np.nanmean(arr).round(0).astype(int)
		fubu[metric] = year_dict

	fubu_df = pd.DataFrame(fubu)

	# FUBU dates
	day_list = pd.Series( np.concatenate([ np.arange(1,367) if calendar.isleap(year) else np.arange(1,366) for year in fubu_df.index ]), index=pd.date_range('{}-01-01'.format(begin_year), '{}-12-31'.format(end_year), freq='D') )
	day_list = day_list.loc[sl.start:sl.stop]
	day_list[:] = np.nan

	out = []
	for dt,df in day_list.groupby( pd.Grouper(freq='Y') ):
		year = dt.year
		for metric in metrics:
			val = fubu_df.loc[ year, metric ] - 1
			ts = pd.Timestamp( datetime.datetime.strptime(str(year)+str(val), '%Y%j') )
			if ts > day_list.index[0] and ts < day_list.index[-1] :
				day_list.loc[ts] = annual_dat.loc[ts]

	plt.figure(figsize=(10, 6))
	# plot the 'annual' data
	plt.plot( annual_dat.values )

	# plot extended climatology
	plt.plot( clim_mean )
	plt.plot( day_list.values, 'bo' )

	plt.tight_layout()
	plt.savefig(os.path.join( base_path,'png','barrow_avg_hann_{}_{}-{}_fig5-6.png'.format(window_len, sl.start.split('-')[0],sl.stop.split('-')[0])), dpi=300)
	plt.cla()
	plt.close()

