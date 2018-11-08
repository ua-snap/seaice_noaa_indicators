# # # # launcher

# interpolate
import subprocess, os
os.chdir('/atlas_scratch/malindgren/repos/seaice_noaa_indicators/pipeline')

base_path = '/atlas_scratch/malindgren/nsidc_0051'
ncpus = 64

for window_len in [1,3,4,5,6,7,8,9,10]:
	_ = subprocess.call(['ipython','make_daily_timeseries_interp.py','--','-b', base_path, '-w', str(window_len)])
	# make clim
	if window_len == 1:
		fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_paper_weights.nc'.format(str(window_len))
		out_fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1979-2017_Alaska_hann_paper_weights_climatology.nc'.format(str(window_len))
	else:
		fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}.nc'.format(str(window_len))
		out_fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1979-2017_Alaska_hann_{}_climatology.nc'.format(str(window_len))

	begin = '1979'
	end = '2017'
	_ = subprocess.call(['ipython','make_daily_timeseries_climatology.py','--','-f', fn, '-o', out_fn, '-b', begin, '-e', end])

	# calc FUBU
	if window_len == 1:
		fn = os.path.join( base_path, 'NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_paper_weights.nc')
	else:
		fn = os.path.join( base_path, 'NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_{}.nc'.format(str(window_len)))

	_ = subprocess.call(['ipython','compute_fubu_allyears_alaska_domain.py','--','-b', base_path, '-f', fn, '-begin', begin, '-end', end, '-w', str(window_len)])

	# calc FUBU clim
	if window_len == 1:
		fn = os.path.join( base_path,'NetCDF','nsidc_0051_sic_nasateam_1979-2017_Alaska_hann_paper_weights_climatology.nc' )
	else:
		fn = os.path.join( base_path,'NetCDF','nsidc_0051_sic_nasateam_1979-2017_Alaska_hann_{}_climatology.nc'.format(str(window_len)) )

	_ = subprocess.call(['ipython','compute_fubu_allyears_alaska_domain_climatology.py','--','-b', base_path, '-f', fn, '-n', str(ncpus), '-w', str(window_len)])
	# _ = subprocess.call(['ipython','compute_fubu_allyears_alaska_domain.py','--','-b', base_path, '-f', fn, '-w', str(window_len)])

	# plot this stuff.
	_ = subprocess.call(['ipython','make_figure_3_paper.py','--','-b', base_path, '-w', str(window_len) ])
	_ = subprocess.call(['ipython','make_figure_4_paper.py','--','-b', base_path, '-w', str(window_len) ])
	_ = subprocess.call(['ipython','make_figure_5-6_paper.py','--','-b', base_path, '-w', str(window_len) ])
	_ = subprocess.call(['ipython','make_figure_7_paper.py','--','-b', base_path, '-w', str(window_len) ])
