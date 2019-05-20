# # # # [ THIS IS NOT WORKING CURRENTLY ]

# interpolate
import subprocess, os, warnings
import xarray as xr

os.chdir('/workspace/UA/malindgren/repos/seaice_noaa_indicators/pipeline')
base_path = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051'
ncpus = str(64)

# interpolate and smooth daily timeseries
print('interp/smooth')
_ = subprocess.call(['ipython','make_daily_timeseries_interp_smooth.py','--','-b', base_path, '-n', ncpus])

begin = '1979'
for end in ['2007','2013','2017']:
	with warnings.filterwarnings('ignore'): # [ remove for debugging... ]
		# make clim from smoothed timeseries
		print('clim')
		begin_full, end_full = '1978', '2017'
		fn = os.path.join(base_path,'smoothed','NetCDF','nsidc_0051_sic_nasateam_{}-{}_ak_smoothed.nc'.format(begin_full, end_full))
		clim_fn = os.path.join(base_path,'smoothed','NetCDF','nsidc_0051_sic_nasateam_{}-{}_ak_smoothed_climatology.nc'.format(begin, end))
		_ = subprocess.call(['ipython','make_daily_timeseries_climatology.py','--','-f', fn, '-o', clim_fn, '-b', begin, '-e', end])
		
		# calc FUBU
		print('fubu')
		_ = subprocess.call(['ipython','compute_fubu_dates.py','--','-b', base_path, '-f', fn, '-begin', begin, '-end', end])

		# # calc FUBU clim
		print('fubu clim')
		fubu_fn = os.path.join( base_path,'outputs','NetCDF','nsidc_0051_sic_nasateam_{}-{}_ak_smoothed_fubu_dates.nc'.format(begin, end))
		fubu_clim_fn = fn.replace('.nc', '_climatology.nc')
		with xr.open_dataset(fubu_fn) as ds:
			ds_clim = ds.sel(year=slice(1979,2007)).mean('year').round(0)
			ds_clim.to_netcdf(fubu_clim_fn)


# plots mimicking the paper figs
points_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/selection_points/chuckchi-beaufort_points.shp'
# fig3
out_fn = os.path.join(base_path,'outputs/png/chuckchi-beaufort_avg_fig3.png')
_ = subprocess.call(['ipython','make_figure_3_paper.py','--','-n',fn,'-c',clim_fn,'-p',points_fn,'-o',out_fn])
# fig 4
_ = subprocess.call(['ipython','make_figure_4_paper.py','--','-b',base_path])

_ = subprocess.call(['ipython','make_figure_5-6_paper.py','--','-b',base_path])
_ = subprocess.call(['ipython','make_figure_7_paper.py','--','-b',base_path])





