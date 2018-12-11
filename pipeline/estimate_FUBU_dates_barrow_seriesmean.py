import os
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
from affine import Affine
import datetime

base_path = '/atlas_scratch/malindgren/nsidc_0051'

out_dict = {}
for window_len in [3,4,5,6,7,8,9,10]:
	with xr.open_dataset('/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}.nc'.format(str(window_len))) as tmp:
		a = Affine(*eval( tmp.affine_transform )[:6]) # make an affine transform for lookups

	fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}_fubu_dates.nc'.format(str(window_len))
	ds = xr.open_dataset(fn)
	ds_sel = ds.sel(year=slice(1970,2013))
	# make barrow points and get their row/col locs
	points_fn = os.path.join(base_path,'selection_points','barrow_points.shp')
	points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
	colrows = [ ~a*pt for pt in points ]
	colrows = [ (int(c),int(r)) for c,r in colrows ]
	cols = [ c for r,c in colrows ]
	rows = [ r for r,c in colrows ]
	
	out_vals = {}
	metrics = ['freezeup_start','freezeup_end','breakup_end','breakup_start']
	for metric in metrics:
		out = ds_sel[metric].mean('year')
		day = np.mean([ out[r,c].values for c,r in colrows ]).round(0)
		date = datetime.datetime(2007, 1, 1) + datetime.timedelta(int(day))
		out_vals[metric] = date.strftime('%m-%d')

	out_dict[window_len] = out_vals

# dump to disk
pd.DataFrame(out_dict).to_csv('/atlas_scratch/malindgren/nsidc_0051/barrow_FUBU_average_date.csv')