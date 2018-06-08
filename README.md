#### SEA ICE NOAA INDICATORS
---
##### Preprocess the data
- stack to a 3-D daily array, and linearly interpolate over the missing days to generate a full Daily time-series
- moving window mean spatial filter 
	- footprint shape:
		array([[0, 1, 0],
		       [1, 1, 1],
		       [0, 1, 0]])
- set all out-of-bounds values to np.nan for computation cleanliness
- write stacked data to NetCDF to improve further computation of FU/BU
- [?] have I implemented the mean filter properly? 
- [?] how to implement the Hanning Smoothing Filter?

##### FU/BU Computations
- group data to summer months (august/september) and compute average/stdev by year
- group data to winter months (january/february) and compute average/stdev by year
- FREEZEUP Start:
	- for a given year, slice daily values to the dates Sept.1 through Dec.31
	- compute summer_mean + summer_stdev for given year to use as a threshold.
	-  set all threshold values less than 15% concentration to 15%
	- compute where the daily values are >= threshold values
	- find index of first daily value that exceeds threshold and use this index value as the ordinal day for the FreezeUp in that year by adding to the ordinal day of Sept.1 to that index value (store back in NetCDF).

- FREEZEUP End:
	- for a given year, slice daily values to the dates Sept.1 through Dec.31
	- compute winter_mean - 10% and use as a threshold
	- compute where the daily values are >= threshold values
	- find index of first daily value that exceeds threshold and use this index value as the ordinal day for the FreezeUp in that year by adding to the ordinal day of Sept.1 to that index value (store back in NetCDF).
	- if any values are the same as the freezeup_start days add 1 day to those values which mimicks what Mark did in the Mlab script.

- BREAKUP Start:
	- for a given year, slice daily values to Feb.14 through Aug.1
	- compute for the given year the winter_mean - 2 /* winter_stdev to use as a threshold
	- groupby 2 week intervals beyond each day in the sliced daily values series and compute where the first time all 2 week values are < threshold, which indicates breakup start

- BREAKUP End:
	- for a given year, slice daily values to the dates June.1 through Sept.30
	- compute summer_mean + summer_stdev and set all values that are less than 15% concentration to 15%
	- compute where the daily values are < threshold values
	- find index of first daily value that is less than threshold and use this index value as the ordinal day for the FreezeUp in that year by adding to the ordinal day of June.1 to that index value (store back in NetCDF).
	- [?] is the index day the first or last day of the 2 week period, in the all_true computation?


