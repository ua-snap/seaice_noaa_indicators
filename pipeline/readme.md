#### SEAICE PROCESSING -- NSIDC-0051 -- January 2019 (malindgren@alaska.edu)
---

1. __get_convert_nsidc-0051.py__
 - download and convert to GTiff

2. __make_daily_timeseries_interp.py__
 - make a daily timeseries using a simple linear interpolation to fill in missing days in earlier part of the timeseries

3. __make_daily_timeseries_climatology.py__
 - make a daily climatology with the linear interpolated dailies that were generated from Step 2.

4. __compute_fubu_allyears_alaska_domain.py__
 - compute the freeze-up/break-up dates annually using the data produced in Step 2.

5. __compute_fubu_allyears_alaska_domain.py__
 - compute the freeze-up/break-up dates climatology using the data produced in Step 3.

6. __[ tables ] estimate_fubu_dates_seriesmean.py__
 - compute the average freeze-up/break-up dates using the data generated in Step 4.
 - this will produce CSV files that are derived from annual averaging of FUBU dates.

7. __[ tables ] estimate_fubu_dates_seriesmean_clim.py__
 - compute the average freeze-up/break-up dates using the data generated in Step 5.
 - this will produce CSV files that are derived from the pre-computed climatology(ies) of FUBU dates.

8. __[ plotting ] make_figure_3_paper.py__
 - generate a plot that is similar in look and feel to the plot 3 from the paper for comparison of methods employed.

9. __[ plotting ] make_figure_4_paper.py__
 - generate plot to mimick (somewhat) the plot 4 from the paper for comparison.

10.__[ plotting ] make_figure_5-6_paper.py__
 - generate plot to mimick (somewhat) the plots 5 and 6 from the paper for comparison.

11.__[ plotting ] make_figure_7_paper.py__
 - generate plot to mimick (somewhat) the plot 7 from the paper for comparison.