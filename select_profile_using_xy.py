# sample point

import geopandas as gpd
from shapely.geometry import Point
import ast, rasterio
from affine import Affine

x,y = (-1915085, 937121)
pt = Point(x,y)

gdf = gpd.GeoDataFrame( {'id':[1],'geometry':[pt]}, crs={'init':'epsg:3411'}, geometry='geometry' )
gdf.to_file( '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/test_point_ak.shp' )

a = Affine(*ast.literal_eval(ds.affine_transform)[:6])
col, row = ~a * (x, y)
row, col = np.array([row,col]).astype(int)

# rst = rasterio.open('/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/GTiff/alaska/1978/nt_19781026_n07_v1-1_n.tif')
# arr = rst.read(1)
# arr[row,col] #<- select a single profile for testing

profile = ds.sic[:,row,col]
