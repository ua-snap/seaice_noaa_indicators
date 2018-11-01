# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#     make the figure 3 from Hajo and Marks paper.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == '__main__':
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
	parser = argparse.ArgumentParser( description='plot fig 3 paper' )
	parser.add_argument( "-b", "--base_path", action='store', dest='base_path', type=str, help="input hourly directory containing the NSIDC_0051 data converted to GTiff" )
	parser.add_argument( "-w", "--window_len", action='store', dest='window_len', type=int, help="window length to add to the output NetCDF file name" )

	# unpack args
	args = parser.parse_args()
	base_path = args.base_path
	window_len = args.window_len

	# window_len = 4
	# base_path = '/atlas_scratch/malindgren/nsidc_0051'

	# handle custom hann
	if window_len == 1:
		window_len = 'paper_weights'


	netcdf_fn = os.path.join( base_path, 'NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}.nc'.format(window_len))
	ds = xr.open_dataset( netcdf_fn )
	a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

	# [ HARDWIRED ] make barrow points and get their row/col locs
	points_fn = os.path.join( base_path,'selection_points','barrow_points.shp' )
	points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
	colrows = [ ~a*pt for pt in points ]
	colrows = [ (int(c),int(r)) for c,r in colrows ]
	cols = [c for c,r in colrows]
	rows = [r for c,r in colrows]

	# make a climatology
	clim_fn = netcdf_fn.replace( '.nc', '_1979-2007_climatology.nc' )
	# if not os.path.exists( clim_fn ):
	# 	clim_sel = ds.sel( time=slice('1979','2007') )
	# 	clim = clim_sel.groupby('time.dayofyear').mean('time')
	# 	clim.to_netcdf( clim_fn, format='NETCDF3_64BIT' )
	# else:
	# 	clim = xr.open_dataset( clim_fn )

	clim_sel = ds.sel( time=slice('1979','2007') )
	clim = clim_sel.groupby('time.dayofyear').mean('time')
	clim.to_netcdf( clim_fn, format='NETCDF3_64BIT' )

	clim_sel = clim.sel( dayofyear=slice(121, 366) )
	clim_hold = [ clim_sel.sic[:,r,c].values for c,r in colrows ]
	clim_mean = pd.Series( np.mean( clim_hold, axis=0 ), index=clim_sel.dayofyear.to_index() )

	plt.figure(figsize=(10, 4))
	clim_mean.plot( kind='line' )

	plt.tight_layout()
	plt.savefig(os.path.join(base_path, 'png','barrow_avg_hann_{}_fig3.png'.format(window_len)), figsize=(20,2), dpi=300)
	plt.cla()
	plt.close()
