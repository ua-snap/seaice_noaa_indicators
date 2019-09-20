# make the timeseries mark sent able to be consumed by the application

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/mark_test_data_march2019/BarrowCoastalDailySICvalues.csv'
    out_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/mark_test_data_march2019/nsidc_0051_sic_nasateam_1978-2013_Alaska_testcase_oldseries.nc'

    df = pd.read_csv(fn, header=None, names=['year','month', 'day','sic'])
    times = pd.DatetimeIndex(df.apply(lambda x: pd.Timestamp(year=int(x['year']), month=int(x['month']), day=int(x['day'])), axis=1))
    arr = np.array(df['sic'])
    stacked_arr = np.dstack([np.vstack([arr,arr]).T, np.vstack([arr,arr]).T])

    # dummy lonlat
    xc = np.array([-2237013.7936202 , -2212136.75455998])
    yc = np.array([ 1137499.76638333,  1112792.88320338])

    # make a NetCDF with the data
    ds = xr.Dataset({'sic': (['time','xc','yc'],  stacked_arr)},
                    coords={'xc': ('xc', xc),
                            'yc': ('yc', yc),
                            'time': times.to_datetime(),
                            }, 
                    attrs=OrderedDict([('proj_name', 'NSIDC North Pole Stereographic'),
                                    ('proj4string', 'EPSG:3411'),
                                    ('affine_transform',
                                    '[24877.0390602206, 0.0, -2249452.313150307, 0.0, -24706.883179945104, 1149853.2079733, 0.0, 0.0, 1.0]')]))

    # dump this temporary file to disk
    ds.to_netcdf( out_fn )