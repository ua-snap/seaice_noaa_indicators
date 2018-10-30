# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#     make the figure 3 from Hajo and Marks paper.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import xarray as xr
import pandas as pd
import geopandas as gpd
from affine import Affine
import numpy as np
import rasterio, os, calendar, datetime

base_path = '/atlas_scratch/malindgren/nsidc_0051'
netcdf_fn = os.path.join( base_path, 'NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_paper_weights.nc')
ds = xr.open_dataset( netcdf_fn )
a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

# make barrow points and get their row/col locs
points_fn = '/atlas_scratch/malindgren/nsidc_0051/selection_points/barrow_points.shp'
points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
colrows = [ ~a*pt for pt in points ]
colrows = [ (int(c),int(r)) for c,r in colrows ]
cols = [c for c,r in colrows]
rows = [r for c,r in colrows]

# make a climatology
clim_fn = netcdf_fn.replace( '.nc', '_1979-2007_climatology.nc' )
if not os.path.exists( clim_fn ):
	clim_sel = ds.sel( time=slice('1979','2007') )
	clim = clim_sel.groupby('time.dayofyear').mean('time')
	clim.to_netcdf( clim_fn, format='NETCDF3_64BIT' )
else:
	clim = xr.open_dataset( clim_fn )

clim_sel = clim.sel( dayofyear=slice(121, 366) )
clim_hold = [ clim_sel.sic[:,r,c].values for c,r in colrows ]
clim_mean = pd.Series( np.mean( clim_hold, axis=0 ), index=clim_sel.dayofyear.to_index() )

plt.figure(figsize=(10, 4))
clim_mean.plot( kind='line' )

plt.tight_layout()
plt.savefig(os.path.join(base_path, 'png','barrow_avg_hann_paper_weights_fig3.png'), figsize=(20,2), dpi=300)
plt.cla()
plt.close()
