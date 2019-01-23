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
import argparse

# parse some args
parser = argparse.ArgumentParser( description='stack the hourly outputs from raw WRF outputs to NetCDF files of hourlies broken up by year.' )
parser.add_argument( "-b", "--base_path", action='store', dest='base_path', type=str, help="input hourly directory containing the NSIDC_0051 data converted to GTiff" )
# parser.add_argument( "-w", "--window_len", action='store', dest='window_len', type=int, help="window length to add to the output NetCDF file name" )

# unpack args
args = parser.parse_args()
base_path = args.base_path
# window_len = args.window_len

# # TESTING
# # window_len = 4
# base_path = '/Users/malindgren/Documents/nsidc_0051'
# # END TESTING

# # read in the FUBU dates for all years
# # handle custom hann
# if window_len == 1:
# 	window_len = 'paper_weights'

fubu_fn = os.path.join( base_path,'smoothed','NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_smoothed_fubu_dates.nc' )
ds = xr.open_dataset( fubu_fn )
a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

# make barrow points and get their row/col locs
points_fn = os.path.join(base_path,'selection_points','barrow_points.shp')
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

	plt.savefig( os.path.join(base_path, 'outputs', 'png','barrow_avg_hann_smoothed_{}-{}_fig7a.png'.format(start, end)), figsize=(20,2), dpi=300)
	plt.cla()
	plt.close()

