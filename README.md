# Sea Ice NOAA Indicators

This project aims to estimate Freeze/Break-up dates from NSIDC-0051 Version 001 Daily Time Series Data. This codebase consists of a main data pipeline and code for creation of manuscript content for an academic paper that describes and evaluates the sea ice indicators dataset that is produced. Below are steps for running the pipeline.

#### Background

Background on this project can be found in the `archive/` folder. The `auxiliary/` folder also has some Jupyter notebooks containing project updates, which provide some more info on tuning the algorithm.

# Running this codebase

## Install Anaconda Project

This codebase uses Anaconda Project for dependency management. You can install it in a `conda` environment by `conda install anaconda-project`. 

## Environment

#### Env vars

This project is set up to make use of the following environmental variables:

- `$BASE_DIR`: the base directory for storing project data that should be backed up.
- `$OUTPUT_DIR`: the output directory where final products are placed.
- `$SCRATCH_DIR`: scratch directory for storing project data which does not need to be backed up.

If not defined, the user will be prompted for them when any of the `anaconda-project` commands are run.

#### Earthdata username and password

The user should place their NASA Earthdata credentials in `$HOME/.netrc`. See this NASA [Earthdata Wiki page](https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget) for more information. This is necessary for execution of the `anaconda-project` commands that execute notebooks which download via the NASA earthdata platform (`pipeline.ipynb` and `manuscript_content.ipynb`). 

## `anaconda-project` commands

Executing the command `anaconda-project run` will run the entire codebase, which executes the pipeline and code for creating manuscript content. Other commands execute the individual Jupyter notebooks:  

- `anaconda-project run pipeline`: executes the pipeline notebook
- `anaconda-project run poi`: executes the points-of-interest notebook
- `anaconda-project run manuscript`: executes the manuscript content notebook

The result of any of the above is that the code in the notebook(s) is executed and the notebook is rendered to an `.html` file, written to `$OUTPUT_DIR`.

#### Running notebooks interactively

To view/run the notebooks interactively, prefix `jupyter lab` or `jupyter notebook` with `anaconda-project run`, e.g. `anaconda-project run jupyter lab`. This ensures the correct kernel is present.

## Outputs

The final output data set of freeze-up / break-up start / end dates is written to `$OUTPUT_DIR/arctic_seaice_fubu_dates_1979-2019.nc`. Content for the manuscript is written to `$OUTPUT_DIR/manuscript_content`.
