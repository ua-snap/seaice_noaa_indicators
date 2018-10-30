# # # # launcher

# interpolate
import subprocess, os
os.chdir('/atlas_scratch/malindgren/repos/seaice_noaa_indicators/pipeline')

base_path = '/atlas_scratch/malindgren/nsidc_0051'
suffix = 'paper_weights'
ncpus = 64

_ = subprocess.call(['ipython','make_daily_timeseries_interp.py','--','-b', base_path, '-s', suffix])

# make clim
fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_paper_weights.nc'
out_fn = '/atlas_scratch/malindgren/nsidc_0051/NetCDF/nsidc_0051_sic_nasateam_1979-2017_Alaska_hann_paper_weights_climatology.nc'
begin = '1979'
end = '2017'
_ = subprocess.call(['ipython','make_daily_timeseries_climatology.py','--','-f', fn, '-o', out_fn, '-b', begin, '-e', end])

# calc FUBU
fn = os.path.join( base_path, 'NetCDF','nsidc_0051_sic_nasateam_1978-2017_Alaska_hann_paper_weights.nc')
_ = subprocess.call(['ipython','compute_fubu_allyears_alaska_domain.py','--','-b', base_path, '-f', fn, '-begin', begin, '-end', end])

# calc FUBU clim
fn = os.path.join( base_path,'NetCDF','nsidc_0051_sic_nasateam_1979-2017_Alaska_hann_paper_weights_climatology.nc' )
_ = subprocess.call(['ipython','compute_fubu_allyears_alaska_domain_climatology.py','--','-b', base_path, '-f', fn, '-n', ncpus])


