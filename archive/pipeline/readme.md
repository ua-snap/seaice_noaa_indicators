# HOW TO RUN PIPELINE:

**NOTE** the `run_pipeline_atlas_fulldomain.py` script is not currently working but you could use that to make the production runs easier to run

## Files: 
- `make_daily_timeseries.py`: will make a timeseries of daily spatially and temporally smoothed sea ice concentration and output them as NetCDF.
- `make_daily_timeseries_climatology.py`: this will aggregate the daily outputs spatially to a climatology of sea ice concentration.
- `compute_fubu_dates.py`: this script will compute FUBU dates for the prepped daily series generated above.
- `compute_fubu_allyears_alaska_domain.py`: this is an older version of the FUBU code that was used when testing AK-only domain. It is left here so that you can take a look at it if needed.  if not, remove it.
- `compute_fubu_dates_final.py`: This version I made some changes on that _could_ be useful, but it is here for reference, not to run.
- `compute_fubu_dates_with_clim.py`: this will compute the FUBU dates for a climatology
- `make_figure_3_paper.py`: figs and tables to mimick the 2016 Elementa paper
- `make_figure_4_paper.py`: figs and tables to mimick the 2016 Elementa paper
- `make_figure_4_paper_full.py`: figs and tables to mimick the 2016 Elementa paper
- `make_figure_5-6_paper.py`: figs and tables to mimick the 2016 Elementa paper
- `make_figure_7_paper.py`: figs and tables to mimick the 2016 Elementa paper
- `make_table_2_paper.py`: figs and tables to mimick the 2016 Elementa paper
- `make_table_3_paper.py`: figs and tables to mimick the 2016 Elementa paper
- `run_pipeline_atlas.py`:non-working version of a script to run the pipeline. this could be massaged back into production.
- `run_pipeline_atlas_fulldomain.py`:non-working version of a script to run the pipeline. this could be massaged back into production.


## Script Run Order for processing the FUBU dates:
1. `make_daily_timeseries.py`
2. `make_daily_timeseries_climatology.py`
3. `compute_fubu_dates.py`
4. `compute_fubu_dates_with_clim.py`
5. `make_figure_3_paper.py`
6. `make_figure_4_paper.py`
7. `make_figure_4_paper_full.py`
8. `make_figure_5-6_paper.`
9. `make_figure_7_paper.py`
10. `make_table_2_paper.py`
11. `make_table_3_paper.py`
12. `run_pipeline_atlas.py`
