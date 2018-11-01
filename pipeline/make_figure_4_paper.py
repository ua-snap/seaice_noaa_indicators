# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#     make the figure 4/5 from Hajo and Marks paper.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
if __name__ == '__main__':
	import matplotlib
	matplotlib.use('agg')
	from matplotlib import pyplot as plt
	import xarray as xr
	import geopandas as gpd
	from affine import Affine
	import numpy as np
	import rasterio, os
	import argparse

	# parse some args
	parser = argparse.ArgumentParser( description='plot fig 4 paper' )
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
	# window_len = 'paper_weights'
	# base_path = '/atlas_scratch/malindgren/nsidc_0051'

	# netcdf_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska.nc'
	netcdf_fn = os.path.join( base_path, 'NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}.nc'.format(window_len) )
	ds = xr.open_dataset( netcdf_fn )
	a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

	# make barrow points and get their row/col locs
	points_fn = os.path.join(base_path,'selection_points','barrow_points.shp')
	points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
	colrows = [ ~a*pt for pt in points ]
	colrows = [ (int(c),int(r)) for c,r in colrows ]
	cols = [ c for r,c in colrows ]
	rows = [ r for r,c in colrows ]

	# make a climatology
	clim_fn = os.path.join(base_path,'NetCDF','nsidc_0051_sic_nasateam_1979-2017_Alaska_hann_paper_weights_climatology.nc')
	# clim_fn = netcdf_fn.replace( '.nc', '_climatology.nc' )
	if not os.path.exists( clim_fn ):
		clim_sel = ds.sel( time=slice('1979','2013') )
		clim = clim_sel.groupby('time.dayofyear').mean('time')
		clim.to_netcdf( clim_fn )
	else:
		clim = xr.open_dataset( clim_fn )

	clim_vals_pts = [ clim.sic[:,r,c].values for c,r in colrows ]
	clim_vals_mean = np.mean( clim_vals_pts, axis=0 )
	clim_new = np.concatenate( (clim_vals_mean, clim_vals_mean) ) # to extend the series properly

	for sl in [slice('1997-09','1998-10'), slice('2005-09','2006-10')]:
		ds_sel = ds.sel( time=sl )
		hold = [ ds_sel.sic[:,r,c].values for c,r in colrows ]
		annual_dat = np.mean(hold, axis=0)

		# # # # NEW
		fubu_fn = os.path.join(base_path,'NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}_fubu_dates.nc'.format(window_len))
		fubu_ds = xr.open_dataset( fubu_fn )
		metrics = [ 'freezeup_start','freezeup_end','breakup_start','breakup_end' ]
		year = int(sl.start.split('-')[0])
		yrlu = {'freezeup_start':year,'freezeup_end':year,'breakup_start':year+1,'breakup_end':year+1}

		tmp_vals = []
		for metric in metrics:
			year = yrlu[metric]
			arr = fubu_ds[metric].sel(year=year)[rows,cols]
			arr[ np.where(arr == -9999) ] = np.nan
			tmp_vals = tmp_vals + [np.nanmean(arr).round(0).astype(int)]

		freezeup_begin, freezeup_end, breakup_begin, breakup_end = tmp_vals
			
		fubu_dat = np.empty_like( annual_dat )
		fubu_dat[:] = np.nan

		times = ds_sel.time.to_index()
		ordinals = np.array([ int(x.strftime('%j')) for x in times ])

		ind = [ np.where( ordinals == x )[0] for x in [freezeup_begin,freezeup_end,breakup_begin,breakup_end] ]
		ind = [j[i] if j.shape[0] > 1 else j[0] for i,j in zip([0,0,1,1],ind) ]
		fubu_dat[ ind ] = annual_dat[ind]

		# clim
		os.chdir(os.path.join(base_path, 'outputs'))
		with rasterio.open('freezeup_start_avg_allyears_ordinal_hann_{}_climatology.tif'.format(window_len)) as rst:
			freezeup_begin = np.nanmean(rst.read(1)[rows,cols]).round(0).astype(int)
		
		with rasterio.open('freezeup_end_avg_allyears_ordinal_hann_{}_climatology.tif'.format(window_len)) as rst:
			freezeup_end = np.nanmean(rst.read(1)[rows,cols]).round(0).astype(int)

		with rasterio.open('breakup_start_avg_allyears_ordinal_hann_{}_climatology.tif'.format(window_len)) as rst:
			# arr = rst.read(1)[rows,cols]
			breakup_begin = np.nanmean(rst.read(1)[rows,cols]).round(0).astype(int)
		
		with rasterio.open('breakup_end_avg_allyears_ordinal_hann_{}_climatology.tif'.format(window_len)) as rst:
			breakup_end = np.nanmean(rst.read(1)[rows,cols]).round(0).astype(int)
		
		fubu_clim = np.empty_like(clim_vals_mean)
		fubu_clim[:] = np.nan
		fubu_clim[ [freezeup_begin,freezeup_end,breakup_begin,breakup_end] ] = clim_vals_mean[[ freezeup_begin,freezeup_end,breakup_begin,breakup_end ]]

		plt.figure(figsize=(10, 4))
		# plot the 'annual' data
		plt.plot( annual_dat )

		# plot extended climatology
		plt.plot( clim_new[244:-122] )
		plt.plot( np.concatenate([fubu_clim,fubu_clim])[244:-122], 'bo' )
		plt.plot( fubu_dat, 'ro')

		plt.tight_layout()
		plt.savefig( os.path.join(base_path,'png','barrow_avg_hann_{}_{}-{}.png'.format(window_len, sl.start.split('-')[0],sl.stop.split('-')[0])), figsize=(20,2), dpi=300)
		plt.cla()
		plt.close()
