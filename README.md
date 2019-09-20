# Sea Ice NOAA Indicators

This project aims to estimate Freeze/Break-up dates from NSIDC-0051 Version 001 Daily Time Series Data. Below are some steps involving how to run the codes therein.

## Install Python
- See the document [here](https://github.com/ua-snap/seaice_noaa_indicators/How_To_Install_and_Use_Python_on_Atlas.md) for a way to install python on the Atlas cluster at SNAP/IARC

## Install Python Packages
- To install a list of packages to an activated virtual environment (via `source venv/bin/activate`) use:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## Pre-Process NSIDC-0051
- Download and convert NSIDC-0051 Daily Sea Ice Concentration
	- Access and convert from byte flat-binary files to floating-point LZW-compressed GeoTIFF's.
	- See the `download` directory in this repository and follow the [README](https://github.com/ua-snap/seaice_noaa_indicators/blob/master/download/readme.md) file.
- Make NetCDF Time Series
	- Stack 2D GeoTIFFs along a third dimension (time) generating a 'data cube'
	- Interpolate the irregular dailies to regular dailies linearly through filling in missing time-slices.
	- Spatially smooth the data using a 3x3 moving average window
	- Temporally smooth the data using a hanning window with weights `[0.25,0.5,0.25]`. This is performed iteratively across the same dataset 3x which smooths out small <3day events.
	- Generate NetCDF format with x,y coordinates and time stamps attached to each grid point/time-slice. 

## Analyze Freeze-Up/Break-Up (FUBU) Dates
1. Compute annual summer mean and standard deviation for *August 1 - September 30*.
2. Compute annual winter mean and standard deviation for *January 1 - February 28(29)*.
3. Compute annual FUBU date estimation (in this order):
	- __FreezeUp Start__:
		- ___time-frame___: September 1 through December 31
		- ___threshold___: summer_mean + summer_stdev (minimum concentration set to 15%)
		- ___algorithm___: Find 1st instance of concentrations exceeding the threshold. return ordinal day of that instance. If the summer_mean is greater than 25%, return NAN.
	- __FreezeUp End__:
		- ___time-frame___: September 1 through December 31
		- ___threshold___: winter_mean - 10% concentration [note: winter_mean is from current year+1]
		- ___algorithm___: start lookup from FreezeUp Start output, and find the first instance of concentration exceeding the threshold. return ordinal day of that instance.
	- __BreakUp Start__:
		- ___time-frame___: February 1 through August 1
		- ___threshold___: winter_mean - (2*winter_stdev)
		- ___algorithm___: find last day for which previous two weeks are above threshold. Set to nodata any pixels where the summer_mean is greater-than 40%. If day chosen is last day of time window, set it to nodata. If day chosen is beyond August 1st, return NAN.
	- __BreakUp End__:
		- ___time-frame___: June 1 through September 30
		- ___threshold___: summer_mean + summer_stdev (with minimum concentration set to 15%)
		- ___algorithm___: find the last instance where concentration is greater than the threshold. return ordinal day of that instance. If summer_mean is greater than 25%, set pixel to nodata. If the ordinal chosen is the last day of the time-window, set it to nodata.


## Output Data
- all data are output as NetCDF files which makes it easier to implement compression and perform rigorous computation in the Python data stack. It is also a common format for Earth Science data that is commonly used across the Earth System and Climate disciplines.
- figures and tables from 2016 Elementa Paper can be reproduced for comparison against the published literature.
- rudimentary web-applications have been built that allow for collaborative assessments between the existing MATLAB script and the Python version. 
    - example: https://seaice-fubu-app-utqiagvik.herokuapp.com/
        * note that the above app is running on a Heroku free-tier and can take a minute or so to load initially. 
