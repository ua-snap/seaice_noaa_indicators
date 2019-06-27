# Sea Ice NOAA Indicators: Data Processing Steps

#### PRE-PROCESSING:
1. download and convert NSIDC-0051 Daily Sea Ice Concentration time-series from generic flat-binary files to GeoTiff.
2. stack converted GeoTiff time-series to an (irregular) daily time-series in NetCDF format providing a 3-dimensional array with x,y coordinates and time stamps attached to each grid point/slice, which allows for significant processing leverage in this sort of environment.
3. interpolate (linearly) time-series to produce a regular series at the daily time-step.
4. Hanning smooth the 1-D profiles (pixel through time) using a simple convolution technique and the following weights `[0.25,0.5,0.25]`. This is performed iteratively on each profile 3 times to smooth out small ~3day events.
5. Spatially smooth 2D array slices at each time-step using a simple mean 3x3 filter. This attempts removal of some left over land-sea layover effects from the sensor platform(s). 

#### FU/BU COMPUTATION:
1. compute the annual summer mean and standard deviation for data matching the months of August/September.
2. compute the annual winter mean and standard deviation for data matching the months of January/February.
3. for each year, perform FUBU (in this order)
	- __FreezeUp Start__:
		- ___time-frame___: September 1 through December 31
		- ___threshold___: summer_mean + summer_stdev (minimum concentration set to 15%)
		- ___algorithm___: Find 1st instance of concentrations exceeding the threshold. return ordinal day of that instance.
	- __FreezeUp End__:
		- ___time-frame___: September 1 through December 31
		- ___threshold___: winter_mean - 10% concentration [note: winter_mean is from current year+1]
		- ___algorithm___: start lookup from FreezeUp Start output, and find the first instance of concentration exceeding the threshold. return ordinal day of that instance.
	- __BreakUp Start__:
		- ___time-frame___: February 1 through August 1
		- ___threshold___: winter_mean - (2*winter_stdev)
		- ___algorithm___: find last day for which previous two weeks are above threshold. Set to nodata any pixels where the summer_mean is greater-than 40%. If day chosen is last day of time window, set it to nodata.
	- __BreakUp End__:
		- ___time-frame___: June 1 through September 30
		- ___threshold___: summer_mean + summer_stdev (with minimum concentration set to 15%)
		- ___algorithm___: find the last instance where concentration is greater than the threshold. return ordinal day of that instance. If summer_mean is greater than 25%, set pixel to nodata. If the ordinal chosen is the last day of the time-window, set it to nodata.
