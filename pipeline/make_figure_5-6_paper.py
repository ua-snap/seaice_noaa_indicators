# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#     make the figure 5/6 from Hajo and Marks paper.
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
parser = argparse.ArgumentParser( description='plot figs 5/6 paper' )
parser.add_argument( "-n", "--netcdf_fn", action='store', dest='netcdf_fn', type=str, help="input filename of the NSIDC_0051 smoothed NetCDF file" )
parser.add_argument( "-c", "--clim_fn", action='store', dest='clim_fn', type=str, help="input filename of the NSIDC_0051 smoothed climatology NetCDF file" )
parser.add_argument( "-f", "--fubu_fn", action='store', dest='fubu_fn', type=str, help="input filename of the computed freeze/break-up dates (FUBU) " )
parser.add_argument( "-p", "--points_fn", action='store', dest='points_fn', type=str, help="input filename of the points shapefile to use in extraction for plot generation " )
parser.add_argument( "-o", "--out_fn", action='store', dest='out_fn', type=str, help="output filename of the plot to be generated -- PNG format" )


# unpack args
args = parser.parse_args()
netcdf_fn = args.netcdf_fn
clim_fn = args.clim_fn
fubu_fn = args.fubu_fn
points_fn = args.points_fn
out_fn = args.out_fn

# # # TESTING
# netcdf_fn = '/atlas_scratch/malindgren/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_smoothed.nc'
# clim_fn = '/atlas_scratch/malindgren/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1979-2013_Alaska_hann_smoothed_climatology.nc'
# fubu_fn = '/atlas_scratch/malindgren/nsidc_0051/outputs/NetCDF/nsidc_0051_sic_nasateam_1979-2013_Alaska_hann_smoothed_fubu_dates.nc'
# points_fn = '/atlas_scratch/malindgren/nsidc_0051/selection_points/chuckchi-barrow-beaufort_points.shp'
# out_fn = '/atlas_scratch/malindgren/nsidc_0051/outputs/png/chuckchi-barrow-beaufort_avg_fig5-6.png'
# # # # # # #

ds = xr.open_dataset( netcdf_fn )
a = Affine(*eval( ds.affine_transform )[:6]) # make an affine transform for lookups

# make barrow points and get their row/col locs
points = gpd.read_file( points_fn ).geometry.apply(lambda x: (x.x, x.y)).tolist()
colrows = [ ~a*pt for pt in points ]
colrows = [ (int(c),int(r)) for c,r in colrows ]
cols = [c for c,r in colrows]
rows = [r for c,r in colrows]

# make a climatology
clim = xr.open_dataset( clim_fn )

for sl in [slice('1982-09-01','1986-09-30'), slice('2007-09-01','2012-09-30')]:
    ds_sel = ds.sel( time=sl ).copy(deep=True)
    hold = [ ds_sel.sic[:,r,c].values for c,r in colrows ]
    annual_dat = pd.Series( np.mean(hold, axis=0), ds_sel.time.to_index() )

    clim_repeat = xr.concat([ clim if calendar.isleap(i) else clim.drop(60,'dayofyear') for i in range(int(sl.start.split('-')[0]), int(sl.stop.split('-')[0])+1)], dim='dayofyear')['sic']
    clim_repeat_sel = clim_repeat.isel(dayofyear=slice((244-1)-1,-122+(30-1))) # double -1's is for 0 index and one for leapday

    clim_hold = [ clim_repeat_sel[:,r,c].values for c,r in colrows ]
    clim_mean = np.mean( clim_hold, axis=0 )
    
    fubu_ds = xr.open_dataset( fubu_fn )
    begin_year,end_year = sl.start.split('-')[0], sl.stop.split('-')[0]
    fubu_ds_sel = fubu_ds.sel(year=slice(begin_year, end_year))
    years = list(range(int(begin_year), int(end_year)+1))
    
    metrics = [ 'freezeup_start','freezeup_end','breakup_start','breakup_end' ]
    fubu = {}
    for metric in metrics:
        year_dict = {}
        for year in years:
            arr = fubu_ds[metric].sel(year=year)[rows,cols]
            arr.values[ np.where(arr.values == -9999) ] = np.nan
            year_dict[year] = np.nanmean(arr).round(0).astype(int)
        fubu[metric] = year_dict

    fubu_df = pd.DataFrame(fubu)

    # FUBU dates
    daylist = np.concatenate([ np.arange(1,367) if calendar.isleap(year) else np.arange(1,366) for year in fubu_df.index ])
    day_list = pd.Series( daylist, 
                            index=pd.date_range('{}-01-01'.format(begin_year), '{}-12-31'.format(end_year), freq='D') )
    day_list = day_list.loc[sl.start:sl.stop]
    day_list[:] = np.nan

    # make empty series to store the data to plot
    fubu_clim_fu_begin = day_list.copy(deep=True)
    fubu_clim_fu_end = day_list.copy(deep=True)
    fubu_clim_bu_begin = day_list.copy(deep=True)
    fubu_clim_bu_end = day_list.copy(deep=True)
    fubu_clim = {'freezeup_start':fubu_clim_fu_begin,'freezeup_end':fubu_clim_fu_end,
        'breakup_start':fubu_clim_bu_begin,'breakup_end':fubu_clim_bu_end,}

    for dt,df in day_list.groupby( pd.Grouper(freq='Y') ):
        year = dt.year
        for metric in metrics:
            val = fubu_df.loc[ year, metric ] - 1
            ts = pd.Timestamp( datetime.datetime.strptime(str(year)+str(val), '%Y%j') )
            if ts > day_list.index[0] and ts < day_list.index[-1] :
                # day_list.loc[ts] = annual_dat.loc[ts]
                fubu_clim[metric].loc[ts] = annual_dat.loc[ts]

    fig,ax = plt.subplots(figsize=(10, 6))
    plt.xlim(0, len(annual_dat.values)) # this is necessary for using the ticklocators in MPL

    xindex = ds_sel.time.to_index()
    months_lookup = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    xindex_months = np.array([ i.strftime('{}-%y'.format(str(months_lookup[i.month]))) for i in xindex ])

    # # # # -- ticker hacking --
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    
    xs = range(len(annual_dat.values))
    def format_fn(tick_val, tick_pos):
        if int(tick_val) in xs:
            return xindex_months[int(tick_val)]
        else:
            return ''

    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # # # # -- end ticker hacking... --

    ax.set_ylabel( 'Sea Ice Concentration' )
    
    # plot the 'annual' data
    ax.plot( annual_dat.values )

    # plot extended climatology
    ax.plot( clim_mean )
    
    # this needs to be broken out into FU-BU/begin-end and plotted accordingly
    colors = {'freezeup_start':'blue','freezeup_end':'blue',
        'breakup_start':'red','breakup_end':'red',}
    for metric in metrics:
        ax.plot( fubu_clim[metric].values, marker='s', fillstyle='none', color=colors[metric] )

    plt.tight_layout() # make it tight
    out_fn2 = out_fn.replace('.png', '_{}-{}.png'.format(sl.start.split('-')[0],sl.stop.split('-')[0]))
    plt.savefig( out_fn2, dpi=300)
    plt.cla()
    plt.close()

