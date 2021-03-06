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
	parser.add_argument( "-n", "--netcdf_fn", action='store', dest='netcdf_fn', type=str, help="input filename of the NSIDC_0051 smoothed NetCDF file" )
	parser.add_argument( "-c", "--clim_fn", action='store', dest='clim_fn', type=str, help="input filename of the NSIDC_0051 smoothed climatology NetCDF file" )
	parser.add_argument( "-f", "--fubu_fn", action='store', dest='fubu_fn', type=str, help="input filename of the computed freeze/break-up dates (FUBU) " )
	parser.add_argument( "-fc", "--fubu_clim_fn", action='store', dest='fubu_clim_fn', type=str, help="input filename of the computed freeze/break-up dates (FUBU) CLIMATOLOGY file " )
	parser.add_argument( "-p", "--points_fn", action='store', dest='points_fn', type=str, help="input filename of the points shapefile to use in extraction for plot generation " )
	parser.add_argument( "-o", "--out_fn", action='store', dest='out_fn', type=str, help="output filename of the plot to be generated -- PNG format" )

	# unpack args
	args = parser.parse_args()
	netcdf_fn = args.netcdf_fn
	clim_fn = args.clim_fn
	fubu_fn = args.fubu_fn
	points_fn = args.points_fn
	out_fn = args.out_fn


	# # # # TESTING
	# netcdf_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1978-2017_north_smoothed.nc'
	# clim_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1979-2013_north_smoothed_climatology.nc'
	# fubu_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/outputs/NetCDF/nsidc_0051_sic_nasateam_1979-2013_north_smoothed_fubu_dates.nc'
	# fubu_clim_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/outputs/NetCDF/nsidc_0051_sic_nasateam_1979-2013_clim_north_smoothed_fubu_dates.nc'
	# points_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/selection_points/barrow_points.shp'
	# out_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/outputs/png/barrow_avg_fig4_north.png'
	# # # # 

	ds = xr.open_dataset( netcdf_fn )
	a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

	# make barrow points and get their row/col locs
	points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
	colrows = [ ~a*pt for pt in points ]
	colrows = [ (int(c),int(r)) for c,r in colrows ]
	cols = [ c for r,c in colrows ]
	rows = [ r for r,c in colrows ]

	# read in climatology
	clim = xr.open_dataset( clim_fn )

	clim_vals_pts = [ clim.sic[:,r,c].values for c,r in colrows ]
	clim_vals_mean = np.mean( clim_vals_pts, axis=0 )
	clim_new = np.concatenate( (clim_vals_mean, clim_vals_mean) ) # to extend the series properly

	for sl in [slice('1997-09','1998-10'), slice('2005-09','2006-10')]:
		ds_sel = ds.sel( time=sl )
		hold = [ ds_sel.sic[:,r,c].values for c,r in colrows ]
		annual_dat = np.mean(hold, axis=0)

		# # # # NEW
		# fubu_fn = os.path.join(base_path,'outputs','NetCDF','nsidc_0051_sic_nasateam_1978-2017_ak_smoothed_fubu_dates.nc')
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

		# fubu_ds_clim = xr.open_dataset( fubu_clim_fn )

		fubu_clim_ds = xr.open_dataset( fubu_clim_fn )
		freezeup_begin = np.nanmean( fubu_clim_ds['freezeup_start'][rows,cols]).round(0).astype(int)
		freezeup_end = np.nanmean( fubu_clim_ds['freezeup_end'][rows,cols]).round(0).astype(int)
		breakup_begin = np.nanmean( fubu_clim_ds['breakup_start'][rows,cols]).round(0).astype(int)
		breakup_end = np.nanmean( fubu_clim_ds['breakup_end'][rows,cols]).round(0).astype(int)

		fubu_clim = np.empty_like(clim_vals_mean)
		fubu_clim[:] = np.nan
		fubu_clim[ [freezeup_begin,freezeup_end,breakup_begin,breakup_end] ] = clim_vals_mean[[ freezeup_begin,freezeup_end,breakup_begin,breakup_end ]]


		# PLOT FUBU CLIMATOLOGY
		# FU-BEGIN
		fubu_clim_fu_begin = np.empty_like(clim_vals_mean)
		fubu_clim_fu_begin[:] = np.nan
		fubu_clim_fu_begin[freezeup_begin] = clim_vals_mean[freezeup_begin]
		# FU-END
		fubu_clim_fu_end = np.empty_like(clim_vals_mean)
		fubu_clim_fu_end[:] = np.nan
		fubu_clim_fu_end[freezeup_end] = clim_vals_mean[freezeup_end]
		# BU-BEGIN
		fubu_clim_bu_begin = np.empty_like(clim_vals_mean)
		fubu_clim_bu_begin[:] = np.nan
		fubu_clim_bu_begin[breakup_begin] = clim_vals_mean[breakup_begin]
		# BU-END
		fubu_clim_bu_end = np.empty_like(clim_vals_mean)
		fubu_clim_bu_end[:] = np.nan
		fubu_clim_bu_end[breakup_end] = clim_vals_mean[breakup_end]

		# # # PLOT FUBU FROM THE DATA SERIES SHOWN

		freezeup_begin,freezeup_end,breakup_begin,breakup_end = ind
		# FU-BEGIN
		fubu_dat = np.empty_like( annual_dat )
		fubu_dat_fu_begin = np.empty_like(annual_dat)
		fubu_dat_fu_begin[:] = np.nan
		fubu_dat_fu_begin[freezeup_begin] = annual_dat[freezeup_begin]
		# FU-END
		fubu_dat_fu_end = np.empty_like(annual_dat)
		fubu_dat_fu_end[:] = np.nan
		fubu_dat_fu_end[freezeup_end] = annual_dat[freezeup_end]
		# BU-BEGIN
		fubu_dat_bu_begin = np.empty_like(annual_dat)
		fubu_dat_bu_begin[:] = np.nan
		fubu_dat_bu_begin[breakup_begin] = annual_dat[breakup_begin]
		# BU-END
		fubu_dat_bu_end = np.empty_like(annual_dat)
		fubu_dat_bu_end[:] = np.nan
		fubu_dat_bu_end[breakup_end] = annual_dat[breakup_end]


		# PLOTTING
		fig,ax = plt.subplots(figsize=(10, 4))

		# plot the 'annual' data
		ax.plot( annual_dat )
		xindex = ds_sel.time.to_index()
		months_lookup = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
		xindex_months = np.array([ months_lookup[i.month] for i in xindex ])
		# do it the hard way --hacky...
		new_ticks = np.arange( 1, len(annual_dat), 70)
		new_labels = xindex_months[new_ticks]

		ax.set_xticklabels( new_labels )
		ax.set_ylabel( 'Sea Ice Concentration' )

		# plot extended climatology
		ax.plot( clim_new[244:-122] )
		# plot fubu markers clim
		ax.plot( np.concatenate([fubu_clim_fu_begin,fubu_clim_fu_begin])[244:-122], 'bs')
		ax.plot( np.concatenate([fubu_clim_fu_end,fubu_clim_fu_end])[244:-122], 'bs' )
		ax.plot( np.concatenate([fubu_clim_bu_begin,fubu_clim_bu_begin])[244:-122], 'rs' )
		ax.plot( np.concatenate([fubu_clim_bu_end,fubu_clim_bu_end])[244:-122], 'rs' )
	

		# plot fubu from data series
		ax.plot( fubu_dat_fu_begin, marker='s', fillstyle='none', color='blue')
		ax.plot( fubu_dat_fu_end, marker='s', fillstyle='none', color='blue')
		ax.plot( fubu_dat_bu_begin, marker='s', fillstyle='none', color='red')
		ax.plot( fubu_dat_bu_end, marker='s', fillstyle='none', color='red')

		plt.tight_layout()
		plt.savefig( out_fn.replace('.png', '_'+'-'.join([sl.start.split('-')[0],sl.stop.split('-')[0]])+'.png' ), figsize=(20,2), dpi=300)
		plt.cla()
		plt.close()
