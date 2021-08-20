# make Table 3 from Marks Paper

# get fubu averages
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine

fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/outputs/NetCDF/nsidc_0051_sic_nasateam_1979-2013_north_smoothed_fubu_dates_climatology.nc'
tmp_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1978-2017_north_smoothed.nc' # for affine
chuckchi_points_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/selection_points/chuckchi_points.shp'
beaufort_points_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/selection_points/beaufort_points.shp'

# read in the fubu dates clim
ds = xr.open_dataset(fn)

with xr.open_dataset(tmp_fn) as tmp:
	affine_transform = tmp.affine_transform

# make affine transform and use to get the data we want for comparison
a = Affine(*eval( affine_transform )[:6])

# read in the points data
chuckchi_points = gpd.read_file(chuckchi_points_fn).geometry.apply(lambda x: (x.x,x.y))
beaufort_points = gpd.read_file(beaufort_points_fn).geometry.apply(lambda x: (x.x,x.y))

# pull data from the paper tables for comparison
mark_all = {'beaufort':pd.Series({'freezeup_start':pd.Timestamp('2001-10-3'),
							'freezeup_end':pd.Timestamp('2001-11-14'),
							'breakup_start':pd.Timestamp('2001-4-24'),
							'breakup_end':pd.Timestamp('2001-8-10') }),
			'chuckchi':pd.Series({'freezeup_start':pd.Timestamp('2001-10-9'),
							'freezeup_end':pd.Timestamp('2001-12-1'),
							'breakup_start':pd.Timestamp('2001-5-2'),
							'breakup_end':pd.Timestamp('2001-8-1') })}

out = {}
for group_name, group in zip(['chuckchi', 'beaufort'],[chuckchi_points,beaufort_points]):
	colrows = [ ~a*pt for pt in group ]
	rowcols = [ (int(r),int(c)) for c,r in colrows ] # flip 'em
	# breakout rows/cols
	rows = [ r for r,c in rowcols ]
	cols = [ c for r,c in rowcols ]

	variables = ['freezeup_start','freezeup_end','breakup_end','breakup_start']
	dat = pd.Series({i:ds[i][rows,cols].mean().round().astype(int).values for i in variables})
	mike = dat.apply(lambda x: pd.Timestamp.strptime('2001{}'.format(x),'%Y%j'))

	# PAPER DATA:
	mark = mark_all[group_name]
	out[group_name] = pd.DataFrame({'mike':mike,'mark':mark})
