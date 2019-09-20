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
import seaborn as sns
import argparse

# parse some args
parser = argparse.ArgumentParser( description='plot fig 7 paper' )
parser.add_argument( "-f", "--fubu_fn", action='store', dest='fubu_fn', type=str, help="input filename of the computed freeze/break-up dates (FUBU) " )
parser.add_argument( "-p", "--points_fn", action='store', dest='points_fn', type=str, help="input filename of the points shapefile to use in extraction for plot generation " )
parser.add_argument( "-o", "--out_fn", action='store', dest='out_fn', type=str, help="output filename of the plot to be generated -- PNG format" )

# unpack args
args = parser.parse_args()
fubu_fn = args.fubu_fn
points_fn = args.points_fn
out_fn = args.out_fn


# # # TESTING
# fubu_fn = '/atlas_scratch/malindgren/nsidc_0051/outputs/NetCDF/nsidc_0051_sic_nasateam_1979-2013_Alaska_hann_smoothed_fubu_dates.nc'
# points_fn = '/atlas_scratch/malindgren/nsidc_0051/selection_points/chuckchi_points.shp'
# out_fn = '/atlas_scratch/malindgren/nsidc_0051/outputs/png/chuckchi_avg_fig7.png'
# # # # #


ds = xr.open_dataset( fubu_fn )
a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

# make barrow points and get their row/col locs
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
metric_groups = [ ['freezeup_start','freezeup_end'],['breakup_start','breakup_end'] ]
for group in metric_groups:
	start, end = group
	dat = df.loc[:,[start,end]]
	
	melted = dat.reset_index().melt(id_vars='year')
	melted.columns = [ i if i != 'value' else 'day of year' for i in melted.columns ]
	melted.columns = [ i if i != 'variable' else ' ' for i in melted.columns ]
	sns.lmplot(x='year', y="day of year", hue=" ", data=melted)

	# ax = dat.plot(kind='line',linestyle="none", marker='o')
	# # trendlines?
	# model = sm.formula.ols(formula='{} ~ year'.format(start), data=dat.reset_index())
	# res = model.fit()
	# trend = res.fittedvalues
	# trend.index = dat.index

	# trend.plot(kind='line', ax=ax)
	out_fn2 = out_fn.replace('.png','_{}.png'.format(start.split('_')[0]))
	plt.savefig( out_fn2, figsize=(20,2), dpi=300)
	plt.cla()
	plt.close()

