# download the AVHRR Polar Pathfinder from NOAA to a local drive
import os

base_url = 'https://www.ncei.noaa.gov/data/avhrr-polar-pathfinder-extended/access/nhem'
output_path = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/AVHRR_PolarPathfinder'
years = range(1982, 2018+1)

for year in years:
	curpath = os.path.join(output_path, str(year))
	if not os.path.exists( curpath ):
		os.makedirs( curpath )

	os.chdir( output_path )

	os.system( 'wget -r --no-parent {}'.format(os.path.join(base_url, str(year))) )