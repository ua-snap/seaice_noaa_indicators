def show_2D_array_aspatial( arr, output_filename ):
    img = plt.imshow( arr, interpolation='nearest' )
    plt.colorbar( img ) 
    plt.savefig( output_filename )
    plt.close()
    return output_filename  

def convert_ordinalday_year_to_datetime( year, ordinal_day ):
    '''
    ordinal_day = [int] range(1, 365+1) or range(1, 366+1) # if leapyear
    year = [int] four digit date that can be read in datetime.date
    '''
    return datetime.date( year=year, month=1, day=1 ) + datetime.timedelta( ordinal_day - 1 )

def get_summer(month):
    return (month == 8) | (month == 9)

def get_winter(month):
    return (month == 1) | (month == 2)

def rolling_window(a, window):
    ''' borrowed from http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html '''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def freezeup_start( ds_sic, summer_mean, summer_std, year ):
    ''' find the first instance of sic exceeding the summer mean + 1 standard deviation threshold '''

    # slice 9/1 to endofyear...
    daily_vals = ds_sic.sel( time=slice(str(year)+'-09-01', str(year)+'-12-31') ).copy()
    start_ordinalday = int( daily_vals.time.to_index().min().strftime('%j') )
    
    # threshold and set minimum value to .15 or 15% conc
    mean_plus_std = (summer_mean.sel(year=int(year)).copy() + summer_std.sel(year=int(year)).copy())
    threshold = mean_plus_std.data.copy()
    threshold[ threshold < .15 ] = .15

    # make a 2D mask of np.nan
    mask = np.isnan( daily_vals.isel( time=0 ).data )
    
    # when does the daily sic exceed the threshold sic
    arr = (daily_vals >= threshold).values

    def first_freezeup( x, start_ordinalday ):
        ''' find the first instance of sic exceeding the threshold '''
        vals, = np.where( x == True )
        # get the first one only! for freezeup
        if len(vals) > 0:
            return vals.min()+start_ordinalday 
        else:
            return np.nan #-9999

    # apply through time
    ordinal_days_freezeup_start = np.apply_along_axis( first_freezeup, axis=0, arr=arr, \
                                        start_ordinalday=start_ordinalday ).astype( np.float32 )
    ordinal_days_freezeup_start[ mask ] = np.nan
    return ordinal_days_freezeup_start

def freezeup_end( ds_sic, winter_mean, freezeup_start_arr, year ):
    ''' find the first instance of sic exceeding the winter mean threshold '''
    
    # select the current year values from Sept.1 - Dec.31
    daily_vals = ds_sic.sel( time=slice(str(year)+'-09-01',str(year)+'-12-31') ).copy(deep=True)
    start_ordinalday = int( daily_vals.time.to_index().min().strftime('%j') )
    nlayers,nrows,ncols = daily_vals.shape

    # make a np.nan mask
    mask = np.isnan( daily_vals.isel( time=0 ).values )

    # remove 10% sic from winter mean values
    threshold = winter_mean.sel(year=int(year)+1).copy(deep=True).values - .10

    # threshold
    arr = daily_vals.values >= threshold

    def last_freezeup( x, y, start_ordinalday ):
        ''' determine the last freezeup day and return doy '''
        vals, = np.where( x == True )
        # only consider values that are > the freezeup_start day
        vals = vals[np.where( (vals+start_ordinalday) > y )]
        if len(vals) > 0:
            minval = vals.min()
            lastfreezeday = (minval+start_ordinalday)
            if lastfreezeday > y:
                out = lastfreezeday
            elif lastfreezeday == y:
                # if same day, take next day
                out = lastfreezeday + 1
            else: #unsolvable
                out = np.nan #-9999
        else:
            out = np.nan #-9999
        return out

    # find freezeup end date.
    ordinal_days_freezeup_end = np.zeros_like(arr[0,...].astype(np.float32))
    for i,j in np.ndindex(arr.shape[-2:]):
        ordinal_days_freezeup_end[i,j] = last_freezeup( arr[:,i,j], freezeup_start_arr[i,j], start_ordinalday )

    # mask it
    ordinal_days_freezeup_end[ mask ] = np.nan
    return ordinal_days_freezeup_end

def breakup_start( ds_sic, winter_mean, winter_std, summer_mean, year ):
    ''' find the day that breakup starts '''
    # % "last day for which previous two weeks are below threshold -2std"
    
    # get daily values for the time range in Mark's Algo (Feb.1 is Feb.14 - 14days search window)
    daily_vals = ds_sic.sel( time=slice(str(year)+'-02-01', str(year)+'-08-01') ).copy(deep=True)
    start_ordinalday = int( daily_vals.time.to_index().min().strftime('%j') ) + 13
    end_ordinalday = int(pd.Timestamp.strptime(str(year)+'-08-01', '%Y-%m-%d').strftime('%j')) # 8/1
    times,rows,cols = daily_vals.shape
    
    # make mask
    mask = np.isnan( daily_vals.isel(time=0).data )
    
    # threshold data
    threshold = (winter_mean - (2*winter_std)).sel(year=int(year)).copy(deep=True)
    arr = (daily_vals > threshold).values

    def alltrue( x ):
        return x.all() # skips np.nan

    def wherefirst( x, start_ordinalday ):
        vals, = np.where( x == True )
        if len(vals) > 0:
            return vals[~np.isnan(vals)].max() + start_ordinalday
        else:
            return np.nan #-9999

    # find where the 2week groups are all sic are below the threshold to start breakup
    twoweek_groups_alltrue = [arr[idx-14:idx,...].all(axis=0) for idx in np.arange(times)[14:]]

    # stack to a 3D array/grab a copy
    arr = np.array( twoweek_groups_alltrue ).copy()

    # ordinal_day_first_breakup_twoweeks
    ordinal_days_breakup_start = np.apply_along_axis( wherefirst, arr=arr, axis=0, start_ordinalday=start_ordinalday ).astype( np.float32 )
    ordinal_days_breakup_start[ mask ] = np.nan # mask it

    # other conditions from the old code:
    #  if summer means are greater than 40% we are making it NO BREAKUP
    summer = summer_mean.sel(year=int(year)).copy(deep=True)
    ordinal_days_breakup_start[ np.where(summer > .40) ] = np.nan

    # if the day chosen is the last day of the time window, make it NA
    ordinal_days_breakup_start[ ordinal_days_breakup_start + 1 == end_ordinalday ] = np.nan
    return ordinal_days_breakup_start

def breakup_end( ds_sic, summer_mean, summer_std, year ):
    ''' compute the breakup ending date for a give year '''
    # % find the last day where conc exceeds summer value plus 1std

    # daily_vals = ds_sic.sel( time=slice(str(year)+'-06-01', str(year)+'-09-30') ).copy(deep=True)
    daily_vals = ds_sic.sel( time=slice(str(year)+'-06-01', str(year)+'-10-01') ).copy(deep=True)
    start_ordinalday = int( daily_vals.time.to_index().min().strftime('%j') )
    # handle potential case of end_year not included in input series
    # end_ordinalday = int(pd.Timestamp.strptime(str(year)+'-09-30', '%Y-%m-%d').strftime('%j')) # 9/30
    end_ordinalday = int(pd.Timestamp.strptime(str(year)+'-10-01', '%Y-%m-%d').strftime('%j'))

    # make a mask
    mask = np.where( np.isnan( daily_vals.isel(time=0).data ) )

    summer = summer_mean.sel(year=int(year)).copy(deep=True)
    mean_plus_std = summer + summer_std.copy(deep=True).sel(year=int(year))
    threshold = mean_plus_std.data.copy()
    threshold[ threshold < .15 ] = .15

    arr = (daily_vals > threshold).values

    def last_breakup( x, start_ordinalday ):
        ''' find the last instance lessthan below the threshold along time axis '''
        vals, = np.where( x == True )

        # add a 1 to the index since it is zero based and we are using it as a way to increment days
        if vals.shape[0] > 0:
            return vals.max()+start_ordinalday # add ordinal start day to the index
        else:
            return np.nan #-9999

    ordinal_days_breakup_end = np.apply_along_axis( last_breakup, axis=0, arr=arr, start_ordinalday=start_ordinalday ).astype( np.float32 )

    # if the summer mean is greater than 25% make it NA
    ordinal_days_breakup_end[ summer.values > .25 ] = np.nan

    # if it is the last day of the time-window, (sept.30th) make it NA
    ordinal_days_breakup_end[ ordinal_days_breakup_end >= end_ordinalday ] = np.nan
    
    # mask it 
    ordinal_days_breakup_end[ mask ] = np.nan

    return ordinal_days_breakup_end

def wrap_fubu( year, ds_sic, summer_mean, summer_std, winter_mean, winter_std ):
    '''
    wrapper function to run an entire year in one process
    * this is mainly used as a wrapper function for multiprocessing.
    '''
    freezeup_start_arr = freezeup_start( ds_sic, summer_mean, summer_std, year )
    freezeup_end_arr = freezeup_end( ds_sic, winter_mean, freezeup_start_arr, year )
    breakup_start_arr = breakup_start( ds_sic, winter_mean, winter_std, summer_mean, year )
    breakup_end_arr = breakup_end( ds_sic, summer_mean, summer_std, year )
    return {'freezeup_start':freezeup_start_arr, 'freezeup_end':freezeup_end_arr, 
            'breakup_start':breakup_start_arr, 'breakup_end':breakup_end_arr }

def make_avg_ordinal( fubu, years, metric ):
    ''' 
    metrics = ['freezeup_start','freezeup_end','breakup_start','breakup_end']
    [ NOTE ]: we are masking the -9999 values for these computations...  which will 
                most likely need to change for proper use.
    '''

    # stack it
    arr = np.array([ fubu_years[year][metric] for year in years ])
    mask = np.isnan( arr[0] ).copy()
    # arr = np.ma.masked_where(arr == -9999, arr, copy=True)
    arr = np.rint( np.ma.mean(arr, axis=0) ).data
    arr[ mask ] = np.nan
    return arr.astype( np.float32 )

def make_xarray_dset_years( arr_dict, years, coords, transform ):
    ''' make a NetCDF file output computed metrics for FU/BU in a 3-D variable-based way '''

    xc,yc = (coords['xc'], coords['yc'])
    attrs = {'proj4string':'EPSG:3411', 'proj_name':'NSIDC North Pole Stereographic', 'affine_transform':transform}

    ds = xr.Dataset({ metric:(['year','yc', 'xc'], arr_dict[metric]) for metric in arr_dict.keys() },
                coords={'xc': ('xc', xc),
                        'yc': ('yc', yc),
                        'year':years }, attrs=attrs )
    return ds

def make_xarray_dset_clim( arr_dict, coords, transform ):
    ''' make a NetCDF file output computed metrics for FU/BU in a 3-D variable-based way '''

    xc,yc = (coords['xc'], coords['yc'])
    attrs = {'proj4string':'EPSG:3411', 'proj_name':'NSIDC North Pole Stereographic', 'affine_transform':transform}

    ds = xr.Dataset({ metric:(['yc', 'xc'], arr_dict[metric].squeeze()) for metric in arr_dict.keys() },
                coords={'xc': ('xc', xc),
                        'yc': ('yc', yc)}, attrs=attrs )
    return ds

def make_xarray_dset_years_levels( arr, years, coords, metrics, transform ):
    ''' make a NetCDF file output computed metrics for FU/BU in a 4-D way '''

    xc,yc = (coords['xc'], coords['yc'])
    attrs = {'proj4string':'EPSG:3411', 'proj_name':'NSIDC North Pole Stereographic', 'affine_transform': transform}

    ds = xr.Dataset({ 'ordinal_day_of_year':(['year','metrics','yc','xc'], arr) },
                coords={'xc': ('xc', xc),
                        'yc': ('yc', yc),
                        'metric':metrics,
                        'year':years }, attrs=attrs )
    return ds

def fubu_dframe_pp(fubu_years):
    ''' this only looks at a single profile and was used for testing with Mark'''
    def convert_time( x ):
        ''' time-conversion to Ordinal Day '''
        if x[1] != 'nan':
            out = pd.Timestamp.strptime(''.join(x.tolist()).split('.')[0],'%Y%j')
            out = out.strftime('%Y-%m-%d')
        else:
            out = '0000'
        return out

    metrics = ['freezeup_start','freezeup_end','breakup_start','breakup_end']
    test = pd.DataFrame({i:{j:fubu_years[i][j][0][0] for j in fubu_years[i]} for i in fubu_years}).T
    # new = test.breakup_start.reset_index().astype(str).apply(lambda x: convert_time(x), axis=1)
    new = pd.DataFrame({i:test[i].reset_index().astype(str).apply(lambda x: convert_time(x), axis=1) for i in metrics} )
    new.index=test.index
    new = new[['freezeup_start','freezeup_end','breakup_start','breakup_end']]
    return new

if __name__ == '__main__':
    import matplotlib
    matplotlib.use( 'agg' )
    from matplotlib import pyplot as plt
    import xarray as xr
    import numpy as np
    import os, dask
    from functools import partial
    import calendar, datetime
    import multiprocessing as mp
    import rasterio
    import pandas as pd
    import argparse

    # parse some args
    parser = argparse.ArgumentParser( description='compute freezeup/breakup dates from NSIDC-0051 prepped dailies' )
    parser.add_argument( "-b", "--base_path", action='store', dest='base_path', type=str, help="parent directory to store sub-dirs of NSIDC_0051 data downloaded and converted to GTiff" )
    parser.add_argument( "-f", "--fn", action='store', dest='fn', type=str, help="path to the generated NetCDF file of daily NSIDC-0051 sic." )
    parser.add_argument( "-begin", "--begin", action='store', dest='begin', type=str, help="beginning year of the climatology" )
    parser.add_argument( "-end", "--end", action='store', dest='end', type=str, help="ending year of the climatology" )
    parser.add_argument( "-n", "--ncpus", action='store', dest='ncpus', type=int, help="number of cpus to use" )
    
    # unpack the args here...  It is just cleaner to do it this way...
    args = parser.parse_args()
    base_path = args.base_path
    fn = args.fn
    begin = args.begin
    end = args.end
    ncpus = args.ncpus
    
    # # # # # # # # # # # TESTING # # # # # # # # # 
    # base_path = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051'
    # fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1978-2017_ak_smoothed.nc'
    # begin = '1979'
    # end = '2017'
    # ncpus=64
    # # # # # # # # # # END TESTING # # # # # # # 
        
    # # # # # # # # # TESTING MARK # # # # # # # # # 
    # base_path = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051'
    # fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/mark_test_data_march2019/nsidc_0051_sic_nasateam_1978-2013_Alaska_testcase_oldseries.nc'
    # begin = '1979'
    # end = '2013'
    # ncpus = 32
    # # # # # # # # END TESTING # # # # # # # 

    np.warnings.filterwarnings('ignore') # filter annoying warnings.

    # # open the NetCDF that we made...
    ds = xr.open_dataset( fn ).load() # load it so it processes a LOT faster plus it is small...
    
    # slice the data to the full years... currently this is 1979-20**
    ds_sic = ds.sel( time=slice( begin, end ) )['sic']
    suffix = '' # empty
    years = ds_sic.time.to_index().map(lambda x: x.year).unique().tolist()[:-1] # cant compute last year
    
    # # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # #   # # # # # # # # # 

    # set all nodata pixels to np.nan
    ds_sic.data[ ds_sic.data > 1 ] = np.nan

    # make a no data mask
    mask = np.isnan( ds_sic.isel(time=0).data )

    # get the summer and winter seasons that were determined in table 1 in the paper
    summer = ds_sic.sel( time=get_summer( ds_sic[ 'time.month' ] ) )
    winter = ds_sic.sel( time=get_winter( ds_sic[ 'time.month' ] ) )

    # get the means and standard deviations of these season aggregates
    summer_mean = summer.groupby( 'time.year' ).mean( dim='time' ).round(4)
    summer_std = summer.groupby( 'time.year' ).std( dim='time', ddof=1 ).round(4)
    winter_mean = winter.groupby( 'time.year' ).mean( dim='time' ).round(4)
    winter_std = winter.groupby( 'time.year' ).std( dim='time', ddof=1 ).round(4)

    # def add_inital_empty_year(seasonal):
    #     if 1978 not in seasonal.year:
    #         # add_empty_1978
    #         tmp = seasonal.isel(year=0)
    #         tmp[:] = np.nan
    #         tmp['year'] = 1978
    #         out = xr.concat([tmp.to_dataset(),seasonal.to_dataset()], dim='year')
    #     else:
    #         out = seasonal
    #     return out
 
    # # update the data to start with a full year if it is needed. this makes the subsequent code less bulky
    # seasonals = {name:add_inital_empty_year(seasonal) for name,seasonal in zip(['summer_mean','summer_std','winter_mean','winter_std'],[summer_mean, summer_std, winter_mean, winter_std])}
    # summer_mean = seasonals['summer_mean']['sic']
    # summer_std = seasonals['summer_std']['sic']
    # winter_mean = seasonals['winter_mean']['sic']
    # winter_std = seasonals['winter_std']['sic']

    # run parallel
    f = partial( wrap_fubu, ds_sic=ds_sic, summer_mean=summer_mean, summer_std=summer_std, winter_mean=winter_mean, winter_std=winter_std )
    pool = mp.Pool( ncpus )
    fubu_years = dict(zip(years, pool.map(f, years)))
    pool.close()
    pool.join()

    # # make NC file with metric outputs as variables and years as the time dimension
    # # --------- --------- --------- --------- --------- --------- --------- ---------
    # stack into arrays by metric
    transform = ds.affine_transform
    metrics = ['freezeup_start','freezeup_end','breakup_start','breakup_end']
    stacked = { metric:np.array([fubu_years[year][metric] for year in years ]) for metric in metrics }
    ds_fubu = make_xarray_dset_years( stacked, years, ds.coords, transform )
    out_fn = os.path.join( base_path, 'outputs','NetCDF','nsidc_0051_sic_nasateam_{}-{}{}_ak_smoothed_fubu_dates.nc'.format(str(begin), str(end), suffix))
    ds_fubu.to_netcdf( out_fn, format='NETCDF4')

    # make averages in ordinal day across all fu/bu
    metrics = ['freezeup_start','freezeup_end','breakup_start','breakup_end']
    averages = { metric:make_avg_ordinal( fubu_years, years, metric) for metric in metrics }

    if len(years) > 1: # this is because ds_fubu of a clim is the same as this avg...
        # write the average dates (clim) to a NetCDF
        ds_fubu_avg = make_xarray_dset_clim( averages, ds.coords, transform )
        ds_fubu_avg.to_netcdf( os.path.join(base_path,'outputs','NetCDF','nsidc_0051_sic_nasateam_{}-{}{}_ak_smoothed_fubu_dates_mean.nc'.format(begin, end, suffix)), format='NETCDF4' )


    # # # show it not spatially warped... -- for testing....
    # output_filename = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/PNG/freezup_avg_allyears_ordinal.png'
    # show_2D_array_aspatial( averages['freezeup_end'], output_filename )

    # Make a raster with the outputs
    template_fn = os.path.join(base_path,'prepped','alaska','1981','nt_19810101_n07_v1-1_n.tif')
    with rasterio.open( template_fn ) as tmp:
        meta = tmp.meta

    for metric in metrics:
        arr = averages[ metric ]

        output_filename = os.path.join(base_path,'outputs','GTiff','{}_avg_{}-{}{}_ordinal_ak_smoothed.tif'.format(metric, begin, end, suffix))
        dirname = os.path.dirname(output_filename)
        if not os.path.exists(dirname):
            _ = os.makedirs(dirname)
        meta.update( compress='lzw', count=1, nodata=np.nan )
        with rasterio.open( output_filename, 'w', **meta ) as out:
            out.write( arr, 1 )


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # SOME NOTES ABOUT TRANSLATING FROM MLAB :
    # ----------------------------------------
    # [ 1 ]: to make an ordinal date that matches the epoch used by matlab in python
    # ordinal_date = date.toordinal(date(1971,1,1)) + 366 
    # if not the number will be 366 days off due to the epoch starting January 0, 0000 whereas in Py Jan 1, 0001.
    # [ 2 ]: when computing stdev it is important to set the ddof=1 which is the matlab default.  Python leaves it at 0 default.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 