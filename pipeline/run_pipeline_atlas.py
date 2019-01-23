# # # # [ THIS IS NOT WORKING CURRENTLY ]

# interpolate
import subprocess, os
# os.chdir('/Users/malindgren/Documents/repos/seaice_noaa_indicators/pipeline')
os.chdir('/atlas_scratch/malindgren/repos/seaice_noaa_indicators/pipeline')

# base_path = '/atlas_scratch/malindgren/nsidc_0051'
base_path = '/atlas_scratch/malindgren/nsidc_0051'
ncpus = 32

for window_len in [1]: # ,3,4,5,6,7,8,9,10]:
	print(window_len)
	_ = subprocess.call(['ipython','make_daily_timeseries_interp.py','--','-b', base_path, '-w', str(window_len)])

	begin = '1979'
	for end in ['2007','2013','2017']:
		# make clim
		begin_full = '1978'
		end_full = '2017'
		if window_len == 1:
			fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_paper_weights.nc'.format(str(begin_full), str(end_full))
			out_fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_paper_weights_climatology.nc'.format(str(begin), str(end))
		else:
			fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_{}.nc'.format(str(begin_full), str(end_full), str(window_len),)
			out_fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_{}_climatology.nc'.format(str(begin_full), str(end_full), str(window_len),)
		
		_ = subprocess.call(['ipython','make_daily_timeseries_climatology.py','--','-f', fn, '-o', out_fn, '-b', begin, '-e', end])
		# calc FUBU
		if window_len == 1:
			fn = os.path.join( base_path, 'NetCDF','nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_paper_weights.nc'.format(str(begin), str(end),))
		else:
			fn = os.path.join( base_path, 'NetCDF','nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_{}.nc'.format(str(begin), str(end), str(window_len),))

		_ = subprocess.call(['ipython','compute_fubu_allyears_alaska_domain.py','--','-b', base_path, '-f', fn, '-begin', begin, '-end', end, '-w', str(window_len)])

		# calc FUBU clim
		if window_len == 1:
			fn = os.path.join( base_path,'NetCDF','nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_paper_weights_climatology.nc'.format(str(begin), str(end)) )
		else:
			fn = os.path.join( base_path,'NetCDF','nsidc_0051_sic_nasateam_{}-{}_Alaska_hann_{}_climatology.nc'.format(str(begin), str(end), str(window_len)) )
		
		begin = '2004'
		end = '2004'
		_ = subprocess.call(['ipython','compute_fubu_allyears_alaska_domain.py','--','-b', base_path, '-f',fn, '-begin',begin,'-end',end, '-n', str(ncpus), '-w', str(window_len)])
		# _ = subprocess.call(['ipython','compute_fubu_allyears_alaska_domain.py','--','-b', base_path, '-f', fn, '-w', str(window_len)])
		break			

	# plot this stuff.
	_ = subprocess.call(['ipython','make_figure_3_paper.py','--','-b', base_path, '-w', str(window_len) ])
	_ = subprocess.call(['ipython','make_figure_4_paper.py','--','-b', base_path, '-w', str(window_len) ])
	_ = subprocess.call(['ipython','make_figure_5-6_paper.py','--','-b', base_path, '-w', str(window_len) ])
	_ = subprocess.call(['ipython','make_figure_7_paper.py','--','-b', base_path, '-w', str(window_len) ])
