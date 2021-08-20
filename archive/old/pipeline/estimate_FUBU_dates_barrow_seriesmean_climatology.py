import os, rasterio
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
from affine import Affine
import datetime

base_path = '/atlas_scratch/malindgren/nsidc_0051'

out_dict = {}
metrics = ['freezeup_start','freezeup_end','breakup_end','breakup_start']
points_regions = ['barrow', 'chuckchi', 'beaufort', 'cb']
for points_region in points_regions:
	for window_len in ['paper_weights']:
		# with xr.open_dataset('/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}.nc'.format(str(window_len))) as tmp:
			# a = Affine(*eval( tmp.affine_transform )[:6]) # make an affine transform for lookups
		
		ds_sel = {}
		for metric in metrics:
			# fn = '/atlas_scratch/malindgren/nsidc_0051/outputs/{}_avg_allyears_ordinal_hann_{}_climatology.tif'.format(metric, str(window_len))
			fn = '/atlas_scratch/malindgren/nsidc_0051/outputs/{}_avg_daily-clim_ordinal_hann_{}.tif'.format( metric, str(window_len) )
			with rasterio.open( fn ) as rst:
				arr = rst.read(1)
				a = rst.transform
			ds_sel[metric] = arr
				
		# read points and get their row/col locs
		if points_region == 'cb':
			points_fn1 = os.path.join(base_path,'selection_points','{}_points.shp'.format('chuckchi'))
			points_fn2 = os.path.join(base_path,'selection_points','{}_points.shp'.format('beaufort'))
			points1 = gpd.read_file( points_fn1 ).geometry.apply(lambda x: (x.x, x.y)).tolist()
			points2 = gpd.read_file( points_fn2 ).geometry.apply(lambda x: (x.x, x.y)).tolist()
			points = points1+points2
		else:
			points_fn = os.path.join(base_path,'selection_points','{}_points.shp'.format(points_region))
			points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()

		colrows = [ ~a*pt for pt in points ]
		colrows = [ (int(c),int(r)) for c,r in colrows ]
		cols = [ c for r,c in colrows ]
		rows = [ r for r,c in colrows ]
		
		out_vals = {}
		for metric in metrics:
			out = ds_sel[metric].copy()
			out[np.where(out < 0)] = np.nan # set any -9999 vals from FUBU processing to np.nan
			# out = out.mean('year') # make an average
			day = np.mean([ out[r,c] for c,r in colrows ]).round(0)
			date = datetime.datetime(2007, 1, 1) + datetime.timedelta(int(day))
			out_vals[metric] = date.strftime('%m-%d')

		out_dict[window_len] = out_vals

	# dump to disk
	df = pd.DataFrame(out_dict)
	df = df.loc[metrics] # sort the rows?
	df.to_csv('/atlas_scratch/malindgren/nsidc_0051/{}_FUBU_average_date_daily-clim.csv'.format(points_region))
	
