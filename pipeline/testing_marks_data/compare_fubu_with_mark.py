# now lets get the FUBU dates by year to compare to Marks:

if __name__ == '__main__':
	import os
	import xarray as xr
	import pandas as pd
	import numpy as np

	fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/outputs/NetCDF/nsidc_0051_sic_nasateam_1979-2012_Alaska_hann_smoothed_fubu_dates.nc'
	ds = xr.open_dataset(fn)

	out = []
	metrics = ['breakup_end','freezeup_start','breakup_start','freezeup_end',]
	for metric in metrics:
		metric_series = ds[metric][:,0,0].to_pandas()
		metric_series.name = metric
		out = out + [metric_series]

	final = pd.concat(out, axis=1)
	final = final.replace(np.nan, 000)
	final = final.astype(int)

	final_out_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/mark_test_data_march2019/prepped/fubu_dates_michael.csv'
	final.to_csv( final_out_fn )

	# now read in the FUBU dates mark sent
	mark_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/mark_test_data_march2019/FUBUDatesNOSmooth.csv'
	mark_fubu = pd.read_csv(mark_fn)

	def make_ordinal(x):
		if x != '00-Jan-0000':
			out = pd.Timestamp(x).strftime('%j')
		else:
			out = '000' # not solvable?
		return out

	mark_fubu.columns = ['breakup_start', 'breakup_end', 'freezeup_start', 'freezeup_end']
	mark_fubu = mark_fubu[metrics] # sort the cols same as my version
	mark_fubu = mark_fubu.applymap(lambda x: make_ordinal(x))
	mark_fubu.index = np.arange(1979,2013)
	mark_fubu = mark_fubu.astype(int)

	mark_out_fn = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/mark_test_data_march2019/prepped/fubu_dates_mark.csv'
	mark_fubu.to_csv( mark_out_fn )

