# download the MASAM2 Data from NSIDC to a local drive
import os

base_url = 'ftp://sidads.colorado.edu/pub/DATASETS/NOAA/G10005/Data'
output_path = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/masam2'
years = range(2012,2017+1)

os.chdir( output_path )
for year in years:
	os.system( 'wget {}'.format(os.path.join(base_url, str(year), '*')) )
