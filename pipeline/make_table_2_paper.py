# make Table 2 from Marks Paper

# get fubu averages
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine

fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/outputs/NetCDF/nsidc_0051_sic_nasateam_1979-2007_north_smoothed_fubu_dates_climatology.nc'
tmp_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1978-2017_north_smoothed.nc' # for affine
points_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/selection_points/chuckchi-beaufort_points.shp'

# read in the fubu dates clim
ds = xr.open_dataset(fn)

with xr.open_dataset(tmp_fn) as tmp:
	affine_transform = tmp.affine_transform

# read in the points data
points = gpd.read_file(points_fn).geometry.apply(lambda x: (x.x,x.y))

colrows = [ ~a*pt for pt in points ]
rowcols = [ (int(r),int(c)) for c,r in colrows ] # flip 'em
# breakout rows/cols
rows = [ r for r,c in rowcols ]
cols = [ c for r,c in rowcols ]

# make affine transform and use to get the data we want for comparison
a = Affine(*eval( affine_transform )[:6])

variables = ['freezeup_start','freezeup_end','breakup_end','breakup_start']
dat = pd.Series({i:ds[i][rows,cols].mean().round().astype(int).values for i in variables})
mike = dat.apply(lambda x: pd.Timestamp.strptime('2001{}'.format(x),'%Y%j'))

# PAPER DATA:
mark = pd.Series({'freezeup_start':pd.Timestamp('2001-10-16'),
		'freezeup_end':pd.Timestamp('2001-11-12'),
		'breakup_start':pd.Timestamp('2001-5-28'),
		'breakup_end':pd.Timestamp('2001-8-1') })

out = pd.DataFrame({'mike':mike,'mark':mark})