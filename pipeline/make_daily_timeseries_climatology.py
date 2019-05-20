# functionality to make a climatology from the daily file made by SNAP through interpolation

if __name__ == '__main__':
    import os
    import xarray as xr
    import numpy as np
    import argparse

    # parse some args
    parser = argparse.ArgumentParser( description='make a climatology NetCDF file of Sea Ice Concentration from the daily nsidc-0051' )
    parser.add_argument( "-f", "--fn", action='store', dest='fn', type=str, help="daily NSIDC-0051 NetCDF file" )
    # parser.add_argument( "-o", "--out_fn", action='store', dest='out_fn', type=str, help="name/path of output daily climatology file to be generated" )
    # # parser.add_argument( "-b", "--begin", action='store', dest='begin', type=str, help="beginning year of the climatology" )
    # # parser.add_argument( "-e", "--end", action='store', dest='end', type=str, help="ending year of the climatology" )


    args = parser.parse_args()
    fn = args.fn
    # out_fn = args.out_fn
    # # begin = args.begin
    # # end = args.end

    # # # # # testing
    # fn = '/atlas_scratch/malindgren/nsidc_0051/smoothed/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_smoothed.nc'
    # dirname, basename = os.path.split(fn)
    # # # # # # # # #

    dirname, basename = os.path.split(fn)
    for end_year in ['2007','2013','2017']:
        # hardwired arguments for some climatology needs.
        begin = '1979'
        end = end_year
        out_fn = os.path.join( dirname, 'nsidc_0051_sic_nasateam_{}-{}_ak_smoothed_climatology.nc'.format(begin,end) )
    
        # make climatology --> 0-366 includes leaps
        ds = xr.open_dataset( fn ).load()
        ds_sel = ds.sel( time=slice(begin,end) )
        clim = ds_sel.groupby( 'time.dayofyear' ).mean( dim='time' )
        clim['sic'].values[np.where(clim['sic'].values < 0)] = 0
        clim['sic'].values[np.where(clim['sic'].values > 1)] = 1
        clim.to_netcdf( out_fn, format='NETCDF4' )
