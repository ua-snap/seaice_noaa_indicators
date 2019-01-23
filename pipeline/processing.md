#### SEAICE PROCESSING -- NSIDC-0051 -- January 2019 (malindgren@alaska.edu)
---

1. get_convert_nsidc-0051.py
	--- download and convert to GTiff

2. make_daily_timeseries_interp.py
	--- make a daily timeseries using a simple linear interpolation to fill in missing days in earlier part of the timeseries

3. make_daily_timeseries_climatology.py
	--- make a daily climatology with the linear interpolated dailies that were generated from Step 2.

4. compute_fubu_allyears_alaska_domain.py
	--- compute the freeze-up/break-up dates annually using the data produced in Step 2.

5. compute_fubu_allyears_alaska_domain.py
	--- compute the freeze-up/break-up dates climatology using the data produced in Step 3.

6. [ testing ] estimate_fubu_dates_seriesmean.py
	--- compute the average freeze-up/break-up dates using the data generated in Step 4.
	--- this will produce CSV files that are derived from annual averaging of FUBU dates.

7. [ testing ] estimate_fubu_dates_seriesmean_clim.py
	--- compute the average freeze-up/break-up dates using the data generated in Step 5.
	--- this will produce CSV files that are derived from the pre-computed climatology(ies) of FUBU dates.

8. make_figure_3_paper.py
	--- generate a plot that is similar in look and feel to the plot 3 from the paper for comparison of methods employed.

9. make_figure_4_paper.py
	--- generate plot to mimick (somewhat) the plot 4 from the paper for comparison.

10. make_figure_5-6_paper.py
	--- generate plot to mimick (somewhat) the plots 5 and 6 from the paper for comparison.

11. make_figure_7_paper.py
	--- generate plot to mimick (somewhat) the plot 7 from the paper for comparison.