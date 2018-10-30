# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#     make the figure 7 from Hajo and Marks paper.
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


# read in the FUBU dates for all years
window_len = 4
fubu_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}_fubu_dates.nc'.format(window_len)
ds = xr.open_dataset( fubu_fn )
a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

# make barrow points and get their row/col locs
points_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/selection_points/Barrow_points_bestfit.shp'
points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
colrows = [ ~a*pt for pt in points ]
colrows = [ (int(c),int(r)) for c,r in colrows ]
cols = [c for c,r in colrows]
rows = [r for c,r in colrows]

# make a dataframe of the FUBU dates
metrics = [ 'freezeup_start','freezeup_end','breakup_start','breakup_end' ]
out = []
for metric in metrics:
	hold = [ ds[metric][:,r,c].values for c,r in colrows  ]
	out = out + [pd.Series( np.mean(hold, axis=0), ds.year.to_index() )]

df = pd.concat(out, axis=1)	
df.columns = metrics
df[ (df < 0) ] = np.nan

# plot FUBU
# df.reset_index('year')
metric_groups = [[ 'freezeup_start','freezeup_end'],['breakup_start','breakup_end' ]]
for group in metric_groups:
	start, end = group
	dat = df.loc[:,[start,end]]
	import seaborn as sns
	melted = dat.reset_index().melt(id_vars='year')
	sns.lmplot(x='year', y="value", hue="variable", data=melted)

	# ax = dat.plot(kind='line',linestyle="none", marker='o')
	# # trendlines?
	# model = sm.formula.ols(formula='{} ~ year'.format(start), data=dat.reset_index())
	# res = model.fit()
	# trend = res.fittedvalues
	# trend.index = dat.index

	# trend.plot(kind='line', ax=ax)

	plt.savefig('/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/selection_points/barrow_avg_hann_{}_{}-{}_fig7a.png'.format(window_len, start, end), figsize=(20,2), dpi=300)
	plt.cla()
	plt.close()

