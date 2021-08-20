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
	parser.add_argument( "-n", "--netcdf_fn", action='store', dest='netcdf_fn', type=str, help="input filename of the NSIDC_0051 smoothed NetCDF file" )
	parser.add_argument( "-c", "--clim_fn", action='store', dest='clim_fn', type=str, help="input filename of the NSIDC_0051 smoothed climatology NetCDF file" )
	parser.add_argument( "-p", "--points_fn", action='store', dest='points_fn', type=str, help="input filename of the points shapefile to use in extraction for plot generation " )
	parser.add_argument( "-o", "--out_fn", action='store', dest='out_fn', type=str, help="output filename of the plot to be generated -- PNG format" )

	# unpack args
	args = parser.parse_args()
	netcdf_fn = args.netcdf_fn
	clim_fn = args.clim_fn
	points_fn = args.points_fn
	out_fn = args.out_fn

	# # # # TESTING
	# netcdf_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1978-2017_north_smoothed.nc'
	# clim_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1979-2007_north_smoothed_climatology.nc'
	# points_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/selection_points/chuckchi-beaufort_points.shp'
	# out_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/outputs/png/chuckchi-beaufort_avg_fig3.png'
	# # # # 

	ds = xr.open_dataset( netcdf_fn )
	a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

	# open the points and get their row/col locs
	points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
	colrows = [ ~a*pt for pt in points ]
	colrows = [ (int(c),int(r)) for c,r in colrows ]
	cols = [c for c,r in colrows]
	rows = [r for c,r in colrows]

	# read in an already produced climatology
	clim = xr.open_dataset( clim_fn )

	clim_sel = clim.sel( dayofyear=slice(121, 366) )
	clim_hold = [ clim_sel.sic[:,r,c].values for c,r in colrows ]
	clim_mean = pd.Series( np.mean( clim_hold, axis=0 ), index=clim_sel.dayofyear.to_index() )

	plt.figure(figsize=(10, 4))
	clim_mean.plot( kind='line' )

	# set up the output dir (if needed)
	dirname = os.path.dirname(out_fn)
	if not os.path.exists(dirname):
		_ = os.makedirs(dirname)

	plt.tight_layout()
	plt.savefig(out_fn, figsize=(20,2), dpi=300)
	plt.cla()
	plt.close()
