# # # # [ THIS IS NOT WORKING CURRENTLY ]

# interpolate
import subprocess, os

os.chdir('/workspace/UA/malindgren/repos/seaice_noaa_indicators/pipeline')
base_path = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051'
ncpus = str(64)

# interpolate and smooth daily timeseries
_ = subprocess.call(['ipython','make_daily_timeseries_interp_smooth.py','--','-b', base_path, '-n', ncpus])

begin = '1979'
for end in ['2007','2013','2017']:
	
	# make clim from smoothed timeseries
	begin_full, end_full = '1978', '2017'
	fn = base_path, 'smoothed','NetCDF','nsidc_0051_sic_nasateam_{}-{}_ak_smoothed.nc'.format(begin_full, end_full)
	out_fn = 'nsidc_0051_sic_nasateam_{}-{}_ak_smoothed_climatology.nc'.format(begin, end)
	_ = subprocess.call(['ipython','make_daily_timeseries_climatology.py','--','-f', fn, '-o', out_fn, '-b', begin, '-e', end])
	
	# calc FUBU
	fn = os.path.join( base_path,'smoothed','NetCDF','nsidc_0051_sic_nasateam_{}-{}_ak_smoothed.nc'.format(begin, end))
	_ = subprocess.call(['ipython','compute_fubu_dates.py','--','-b', base_path, '-f', fn, '-begin', begin, '-end', end])

	# calc FUBU clim
	fn = os.path.join( base_path,'smoothed','NetCDF','nsidc_0051_sic_nasateam_{}-{}_ak_smoothed_climatology.nc'.format(begin, end))
	# a single year needs to be passed to the FUBU computation.  
	#  this uses 2004, it is meaningless, but necessary to treat the 30year clim as a 'year'.
	cbegin, cend = '2004','2004'
	_ = subprocess.call(['ipython','compute_fubu_dates.py','--','-b',base_path,'-f',fn,'-begin',cbegin,'-end',cend,'-n',ncpus])

# plots mimicking the paper figs
_ = subprocess.call(['ipython','make_figure_3_paper.py','--','-b',base_path])
_ = subprocess.call(['ipython','make_figure_4_paper.py','--','-b',base_path])
_ = subprocess.call(['ipython','make_figure_5-6_paper.py','--','-b',base_path])
_ = subprocess.call(['ipython','make_figure_7_paper.py','--','-b',base_path])
