# sample point
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import ast, rasterio
from affine import Affine
import xarray as xr
import numpy as np
import datetime
import seaborn as sns

x,y = (-1915085, 937121)
pt = Point(x,y)

gdf = gpd.GeoDataFrame( {'id':[1],'geometry':[pt]}, crs={'init':'epsg:3411'}, geometry='geometry' )
gdf.to_file( '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/test_point_ak.shp' )
netcdf_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska.nc'
ds = xr.open_dataset( netcdf_fn )

a = Affine(*ast.literal_eval(ds.affine_transform)[:6])
col, row = ~a * (x, y)
row, col = np.array([row,col]).astype(int)

# rst = rasterio.open('/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/GTiff/alaska/1978/nt_19781026_n07_v1-1_n.tif')
# arr = rst.read(1)
# arr[row,col] #<- select a single profile for testing

profile = ds.sic[:,row,col]
p2 = profile.resample( time='M' ).mean( 'time' )
years = [str(i) for i in range(1979,2017)]
d = dict()
for year in years:
	p = p2.sel( time=year )
	d[year] = p.data
	p.plot()
	plt.savefig('/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/tmp/test_year_profile_monthly_{}.png'.format(year))
	plt.close()


months = [datetime.date(2000, m, 1).strftime('%B')[:3] for m in range(1, 13)]
df = pd.DataFrame( d ).T
df.columns = months

sns.heatmap( df, cmap='Blues_r' )
plt.savefig( '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/tmp/heatmap_profile_monthly.png' )
plt.close()


