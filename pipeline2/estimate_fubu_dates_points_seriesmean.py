import os
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
from affine import Affine
import datetime

base_path = '/atlas_scratch/malindgren/nsidc_0051'
begin, end = (1979,2007) # 2013
out_dict = {}
metrics = ['freezeup_start','freezeup_end','breakup_start','breakup_end',]
points_regions = ['barrow', 'chuckchi', 'beaufort', 'chuckchi-beaufort',]
for points_region in points_regions:
	with xr.open_dataset('/atlas_scratch/malindgren/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_smoothed.nc') as tmp:
		a = Affine(*eval( tmp.affine_transform )[:6]) # make an affine transform for lookups

	fn = '/atlas_scratch/malindgren/nsidc_0051/outputs/NetCDF/nsidc_0051_sic_nasateam_1979-2013_Alaska_hann_smoothed_fubu_dates.nc'
	ds = xr.open_dataset(fn)
	ds_sel = ds.sel( year=slice(begin,end) )

	# read points and get their row/col locs
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
		out = out.mean('year')
		day = np.nanmean([ out[r,c].values for c,r in colrows ]).round(0)
		date = datetime.datetime(2007, 1, 1) + datetime.timedelta(int(day))
		out_vals[metric] = date.strftime('%m-%d')
	
	out_dict[points_region] = out_vals

# dump to disk
df = pd.DataFrame( out_dict )
df = df.loc[metrics] # sort the rows?
df.to_csv('/atlas_scratch/malindgren/nsidc_0051/outputs/csv/fubu_average_date_{}-{}.csv'.format(points_region,begin,end))
