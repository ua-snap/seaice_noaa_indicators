# HOW TO DOWNLOAD AND PREPARE THE NSIDC-0051 DAILY DATA

1. use the data selection tool for these data located at: https://nsidc.org/data/nsidc-0051 (use the Download Data tab therein)
- this will download a Python script file that can be run at your command prompt.
- it will download all of the data you requested into a directory structure in the folder you instantiate it from.  I reccomend doing that in a folder structure similar to what was used prior in this project.

2. run the downloader script.  The one that is downloading all of the available data (it is up through 2018 currently) is located in this repository and has the name: `nsidc-download_NSIDC-0051.001_2019-07-11.py`
- run it like this in your shell (I suggest using `screen`):
```sh
ipython nsidc-download_NSIDC-0051.001_2019-07-11.py
```
- it will download all of the data to that directory it was instantiated in and they will just populate the `Monthly` and `Daily` data into the same folder.  

3. Move the `monthly` and `daily` to respective folders in a proper hierarchy to make things cleaner and easier to work with.  This also prepares the data in a way that is expected in the `convert_nsidc-0051_raw_to_gtiff.py` file that converts the data from the flat byte binary format to floating point GeoTiffs. The script below will organize it for you.
```sh
ipython restructure_raw_downloaded_nsidc-0051_data.py -p </path/to/download_folder/>
```

4. convert the raw flat binaries to floating-point geotiff using the script below.
```sh
ipython convert_nsidc-0051_raw_to_gtiff.py -- -i '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/raw/daily' -o '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/prepped' -n 64
```

** if for some reason you want to go back to using the old version of the downloading (meaning not using the GUI to get the Python download script from the NSIDC site) you can take a look at `get_convert_nsidc-0051.py` for the old way we did it. **
