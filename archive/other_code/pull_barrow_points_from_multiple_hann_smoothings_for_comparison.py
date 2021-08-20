# TEST THE SMOOTHING NTIMES OVER THE MEAN OF THE BARROW POINTS.
if __name__ == '__main__':
	import os, glob
	import xarray as xr
	import numpy as np
	import pandas as pd
	import geopandas as gpd
	from affine import Affine

	raw_interp_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/intermediates/NetCDF/nsidc_0051_sic_nasateam_1978-2017_north_interp_daily.nc'
	points_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/selection_points/barrow_points.shp'
	files = glob.glob('/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1978-2017_hann*.nc')
	
	out = {}
	for fn in files:
		ds = xr.open_dataset( fn )
		# ds.attrs.update(affine_transform='[25000.0, 0.0, -3850000.0, 0.0, -25000.0, 5850000.0, 0.0, 0.0, 1.0]')
		a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups
		da = ds.sic

		# make barrow points and get their row/col locs
		points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
		colrows = [ ~a*pt for pt in points ]
		rowcols = [ (int(r),int(c)) for c,r in colrows ] # flip 'em
		# breakout rows/cols
		rows = [ r for r,c in rowcols ]
		cols = [ c for r,c in rowcols ]
		out[os.path.basename(fn).split('_')[-3]]=da[:,rows,cols].mean(axis=(1,2)).values
		
	ds_raw_interp = xr.open_dataset(raw_interp_fn)
	out['raw'] = ds_raw_interp.sic[:,rows,cols].mean(axis=(1,2)).values
	df = pd.DataFrame(out, index=ds.time.to_index())
	df.to_csv('/rcs/other/ML/sic_hann_smooth_ntimes.csv')
