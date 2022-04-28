"""Manuscript content functions. 

This script contains all functions used for creating the manuscript content.

Imports are done within functions for greater transparency of dependencies.
"""

def load_fubu(fubu_fp):
    """Load the FUBU data created from the pipeline
    Args:
        fubu_fp (pathlib.PosixPath): BASE_DIR path
        
    Returns:
        FUBU xarray.DataSet with some workability mods
    """
    import numpy as np
    import xarray as xr

    fubu = xr.load_dataset(fubu_fp)
    # Data for Bliss only available through 2017. Reduce J&E timespan to match
    fubu = fubu.sel(year=slice(1979, 2017))
    # set data to floats and np.nan for more straightforward processing
    for indicator in list(fubu.variables)[:4]:
        fubu[indicator] = fubu[indicator].astype(float)
        fubu[indicator].values[fubu[indicator].values == -9999.] = np.nan
        
    return fubu


def load_orac(orac_fp, fubu):
    """Load the ORAC (Bliss et al.) data and replicate the data strcutre 
    of FUBU dataset so it can be used with same functions

    Args:
        orac_fp (pathlib.PosixPath): Path to ORAC indicators 
            dataset (Bliss et al, (NSIDC-0747))
        fubu (xarray.DataSet): J&H indicators dataset
        
    Returns:
        ORAC dataset matching the structure of the FUBU indicators
            dataset
    """
    import numpy as np
    import xarray as xr
    
    orac = xr.load_dataset(orac_fp)
    
    # retain only variables that correspond to FUBU indicators
    # rename coord vars to match
    varnames = ["DOO", "DOR", "DOA", "DOC"]
    orac = orac[varnames].rename({"x": "xc", "y": "yc", "time": "year"})
    # make time variable an integer with year
    orac = orac.assign_coords(year=[dt.year for dt in orac["year"].values])
    # set non-indicator date values to np.nan
    for var in varnames:
        orac[var].values[orac[var].values < 1] = np.nan

    return orac


def get_landmask(orac_fp):
    """Get landmask from ORAC dataset
    Args:
        orac_fp (pathlib.PosixPath): path to ORAC dataset
        
    Returns:
        A landmask derived from the ORAC dataset.
    """
    import xarray as xr

    with xr.open_rasterio(f"netcdf:{orac_fp}:DOA") as da:
        landmask = da.sel(band=1).values == -4
        
    return landmask


def save_fig(fp):
    """Helper function to save matplotlib figures"""
    import time
    import datetime as dt
    import matplotlib.pyplot as plt
    
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    print(
        (f"Plot written to {fp} at "
        f"{dt.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"), 
        file=terminal_output, 
        flush=True
    )
    
    return


def make_totals_maps(ds, varnames, landmask, titles, out_fp):
    """Make the totals maps - heatmaps of
    pixel-wise counts of years where indicator was defined -
    for a supplied indicators dataset and variable names

    Args:
        ds (xarray.Dataset): indicator Dataset
        varnames (list): names of variables in ds to count and plot totals for
        landmask (np.ndarray): array where values of true correspond to pixels
            that overlap land
        landmask_poly (geopandas.GeoDataFrame): polgyon of the
            landmask
        out_fp (pathlib.PosixPath): absolute path to image write location

    Returns:
        None, writes image to out_fp
    """
    import copy
    import matplotlib.pyplot as plt
    import numpy as np

    def prep_totals(var):
        """Count totals and add landmask"""
        arr = ds[var].values.copy()
        valid = np.isnan(arr) == False
        counts = np.sum(valid, axis=0)
        plot_arr = np.ma.masked_where(landmask, counts)

        return plot_arr.astype("int32")

    plot_arrs = [prep_totals(var) for var in varnames]

    # plot data
    cmap = copy.copy(plt.cm.get_cmap("viridis"))
    cmap.set_bad(color="gray")
    cmap.set_under(color="white")
    fig, axs = plt.subplots(1, 2, figsize=(10, 7.5))

    # need to write temporary raster for plotting

    for arr, ax, title in zip(plot_arrs, axs, titles):
        # with rio.open("temp.tif", "w+", **meta) as src:
        # src.write(arr, 1)
        im = ax.imshow(arr, interpolation="none", cmap=cmap, vmin=1, vmax=39)
        # im = show((src, 1), ax=ax)
        # landmask_poly.plot(ax=ax, facecolor='none', edgecolor="gray")
        ax.set_title(title, fontdict={"fontsize": 12})
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(wspace=-0.25, right=0.95, left=0.01)
    # cbar_ax = fig.add_axes([0.85, 0.18, 0.1, 0.8])
    # cbar = fig.colorbar(im, cax=cbar_ax)
    cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax.transAxes)
    cbar = fig.colorbar(im, ax=axs, cax=cax, shrink=1.2)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Years where indicator defined", rotation=270, labelpad=14)
    # fig.tight_layout()
    # initial journal consideration: The Cryosphere, min width 8cm (3.15in)
    fig.set_size_inches(10, 7.5)
    save_fig(out_fp)
    plt.show()

    return


def make_isl_maps(fubu, orac, landmask, out_fp):
    """Create the Ice Season Length maps
    
    Args:
        fubu (xarray.DataSet): FUBU indicators dataset
        orac (xarray.DataSet): ORAC indicators dataset
        landmask (numpy.ndarray): 2D boolean array where land values are True
        out_fp (pathlib.PosixPath): absolute path to image write location
        
    Returns:
        None, writes image to out_fp
    """
    import copy
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    
    fubu_isl = (fubu["breakup_end"].values + 365) - fubu["freezeup_start"].values
    orac_isl = (orac["DOR"].values + 365) - orac["DOA"].values
    
    def prep_isl_arr(arr, landmask):
        """Compute mean ice season length array

        Args:
            arr (numpy.ndarray): 3D (year, y, x) cube of ice season length values
            landmask (numpy.ndarray): 2D boolean array where land values are True

        Returns:
            Masked array of mean ISL values taken over year dimension.
        """
        mean_arr = np.nanmean(arr, 0)
        mean_arr[np.isnan(mean_arr)] = 0
        mean_ma_arr = np.ma.masked_where(landmask, mean_arr)

        return mean_ma_arr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        isl_arrs = [prep_isl_arr(arr, landmask) for arr in [fubu_isl, orac_isl]]

    # plot data
    cmap = copy.copy(plt.cm.get_cmap("viridis"))
    cmap.set_bad(color="gray")
    cmap.set_under(color="white")
    fig, axs = plt.subplots(1, 2, figsize=(10, 7.5))

    for arr, ax in zip(isl_arrs, axs):
        im = ax.imshow(arr, interpolation="none", cmap=cmap, vmin=1, vmax=365)
        ax.set_xticks([])
        ax.set_yticks([])

    axs[0].set_title("J&E Ice Season Length", fontdict={"fontsize": 12})
    axs[1].set_title("Bliss Ice Season Length", fontdict={"fontsize": 12})

    fig.subplots_adjust(wspace=-0.25, right=0.95, left=0.01)
    # cbar_ax = fig.add_axes([0.85, 0.18, 0.1, 0.8])
    # cbar = fig.colorbar(im, cax=cbar_ax)
    cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax.transAxes)
    cbar = fig.colorbar(im, ax=axs, cax=cax, shrink=1.2)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Mean ice season length (days)", rotation=270, labelpad=14)
    # fig.tight_layout()
    # initial journal consideration: The Cryosphere, min width 8cm (3.15in)
    fig.set_size_inches(10, 7.5)

    save_fig(out_fp)

    plt.show()
    

def get_poi_scatter_args(poi_coords, raster_fp):
    """Get variables needed to plot the locations as
    scatter plots on top of the various "maps" imshow-based plots.
    """
    import rasterio as rio
    
    maps_poi_offsets = {
        "Utqiaġvik": (8, 0),
        "Prudhoe Bay": (10, 8),
        "Pevek": (-5, -12),
        "Tiksi": (0, -12),
        "Sabetta": (0, -12),
        "Mestersvig": (10, 0),
        "Clyde River": (0, 15),
        "Churchill": (-15, -12),
    }

    # use empty args to get same WGS84 coords for the coastal locations
    poi_kwargs = make_poi_kwargs(poi_coords, None, None, None)
    poi_xy = {
        place: reproject_poi(poi_kwargs[place]["map_di"]["poi_wgs84"])
        for place in poi_kwargs
        # omit places without actual points
        if poi_kwargs[place]["map_di"]["poi_name_adj"] != "none"
    }
    # open a dataset to use the .index()method for getting the row/cols for
    #  plotting the scatters
    # first, define a no-operation function to return the exact
    #  row, column decimal values for plotting scatter on
    #  imshow()
    def no_op(x):
        return x

    with rio.open(raster_fp) as src:
        poi_rc = {place: src.index(*poi_xy[place], op=no_op) for place in poi_xy}

    # make into r and c lists
    poi_r = [poi_rc[place][0] for place in poi_rc]
    poi_c = [poi_rc[place][1] for place in poi_rc]
    places = list(poi_rc.keys())
    # add points and text labels
    bbox_props = dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.75)

    return poi_r, poi_c, places, maps_poi_offsets, bbox_props


def make_fastice_maps(fast_ice_fps, landmask, titles, poi_coords, out_fp):
    """Make the maps of landfast ice

    Args:
        fast_ice_fps (list): list of file paths to landfast ice rasters
            (one median, one maximum)
        landmask (np.ndarray): array where values of true correspond to pixels
            that overlap land
        titles (list): list of titles for the subplots
        poi_coords (pandas.DataFrame): dataframe of the points-of-interest coordinates
        out_fp (pathlib.PosixPath): absolute path to image write location

    Returns:
        None, writes image to out_fp
    """
    import copy
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches
    import rasterio as rio

    # plot data
    cmap = copy.copy(plt.cm.get_cmap("viridis"))
    cmap.set_bad(color="white")
    cmap.set_under(color="gray")
    fig, axs = plt.subplots(1, 2, figsize=(10, 7.5))

    poi_r, poi_c, places, maps_poi_offsets, bbox_props = get_poi_scatter_args(
        poi_coords, fast_ice_fps[0]
    )

    for fp, ax, title in zip(fast_ice_fps, axs, titles):
        with rio.open(fp) as src:
            fast_arr = src.read(1).astype(np.float32)
        fast_arr[fast_arr == -999] = np.nan
        fast_arr[landmask] = -1
        im = ax.imshow(fast_arr, interpolation="none", cmap=cmap, vmin=0, vmax=5)
        ax.scatter(poi_c, poi_r, color="black")
        for x, y, place in zip(poi_c, poi_r, places):
            xoff, yoff = maps_poi_offsets[place]
            ax.text(x + xoff, y + yoff, place, bbox=bbox_props)
        ax.set_title(title, fontdict={"fontsize": 12})
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(wspace=-0.2, right=0.9, left=-0.05, bottom=0.1)
    patch_ax = fig.add_axes([0, 0, 0.9, 0.1])

    fast_patch = mpatches.Patch(
        color="#414487", label="Landfast ice June 1972-2007 climatology"
    )
    patch_ax.set_axis_off()
    # Add legend to bottom-right ax
    patch_ax.legend(handles=[fast_patch], loc="center", fontsize=14)
    fig.set_size_inches(10, 7.5)
    save_fig(out_fp)
    plt.show()

    return

    
def make_date_lag_maps(fubu, orac, landmask, out_fp):
    """Make maps of mean lagged date values, i.e. the 
    mean difference in days between corresponding indicators
    between J&E and Bliss datasets
    
    Args:
        fubu (xarray.DataSet): FUBU indicators dataset
        orac (xarray.DataSet): ORAC indicators dataset
        landmask (numpy.ndarray): 2D boolean array where land values are True
        out_fp (pathlib.PosixPath): absolute path to image write location
        
    Returns:
        None, writes image to out_fp
    """
    import copy
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    
    fubu_orac_di = {
        "breakup_start": "DOO",
        "breakup_end": "DOR",
        "freezeup_start": "DOA", 
        "freezeup_end": "DOC",
    }

    lag_arrs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for indicator in list(fubu_orac_di.keys()):
            lag_arr = np.nanmean(
                fubu[indicator].values - orac[fubu_orac_di[indicator]].values, 
                axis=0
            )
            lag_arr[~landmask & np.isnan(lag_arr)] = -9999
            lag_arrs.append(lag_arr)

    # plot data
    cmap = copy.copy(plt.cm.get_cmap("viridis"))
    cmap.set_bad(color="gray")
    cmap.set_under(color="white")
    fig, axs = plt.subplots(2, 2, figsize=(10, 15))

    for arr, ax in zip(lag_arrs, axs.flatten()):
        im = ax.imshow(arr, interpolation="none", cmap=cmap, vmin=-180, vmax=115)
        ax.set_xticks([])
        ax.set_yticks([])

    axs[0, 0].set_title("Break-up Start", fontdict={"fontsize": 12})
    axs[0, 1].set_title("Break-up End", fontdict={"fontsize": 12})
    axs[1, 0].set_title("Freeze-up Start", fontdict={"fontsize": 12})
    axs[1, 1].set_title("Freeze-up End", fontdict={"fontsize": 12})

    fig.subplots_adjust(wspace=-0.32, hspace=.1, right=0.95, left=0.01)

    cax = fig.add_axes([0.87, 0.25, 0.025, 0.5])
    # fig.colorbar(im, cax=cbar_ax)

    # cax = ax.inset_axes([1.04, 0, 0.05, 0.6], transform=ax.transAxes)
    cbar = fig.colorbar(im, ax=axs, cax=cax, shrink=1.2)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Mean indicator lag (days)", rotation=270, labelpad=14, fontsize=12)

    save_fig(out_fp)

    plt.show()
    

def get_polygon(row):
    """Create shapely polygon object from row of alternating
    lat/lon coords

    Args:
        pandas series object from df.iterrows() of MASIE regions
        MS Excel file

    returns
        shapely polygon derived from vertices coordinates in row

    Notes:
        designed for use with MASIE_regions_polygon_vertices.xls
    """
    import pandas as pd
    from shapely.geometry import Polygon
    
    df_di = {"lat": [], "lon": []}
    for idx, value in zip(row.index, row):

        if pd.isnull(value):
            df_di["lat"].append(df_di["lat"][0])
            df_di["lon"].append(df_di["lon"][0])
            break
        if "Lat" in idx:
            df_di["lat"].append(value)
        if "Lon" in idx:
            df_di["lon"].append(value)

    return Polygon(list(zip(df_di["lon"], df_di["lat"])))


def get_masie_polys(masie_fp):
    """Get the MASIE region polygons from the spreadsheet of 
    MASIE region vertices
    
    Args:
        masie_fp (pathlib.PosixPath): path to spreadsheet of MASIE region vertices
        
    Returns:
         GeoDataFrame of MASIE polygons in EPSG:3411
         
    Notes:
        Does some renaming for plotting as well.
    """
    import geopandas as gpd
    import pandas as pd

    masie_df = pd.read_excel(masie_fp, skiprows=1)

    masie_polys = masie_df[["Region Number Identifier", "Name"]].copy()
    for index, row in masie_df.iterrows():
        masie_polys.loc[index, "geometry"] = get_polygon(row)

    masie_polys = gpd.GeoDataFrame(masie_polys).set_crs(epsg=4326).to_crs(epsg=3411)
    masie_polys = masie_polys.replace(
        {
            "Name": {
                "Baffin Bay/Gulf of St. Lawrence": "Baffin Bay",
                "Canadian Archipelago": "Canadian Arch.",
                "East Siberian Sea": "E. Siberian Sea",
            }
        }
    )
    
    return masie_polys
    
    
def get_fubu_affine(fubu_fp):
    """Get the affine transform for the FUBU dataset

    Args:
        fubu_fp (pathlib.PosixPath): path to the FUBU dataset created from running the pipeline
        
    Returns:
        The affine transform of the FUBU dataset
    """
    import rasterio as rio
    
    with rio.open(f"netcdf:{fubu_fp}:breakup_end") as src:
        affine = src.meta["transform"]
        
    return affine


def success_rate(marr):
    """Compute the percentage of non-nan values in masked array

    Args:
        marr: Masked numpy array of raster values within the polygon

    Returns:
        Percentage of non-nan values (i.e. valid dates) in marr

    Notes:
        Intended use is for deriving indicator success rates
        for MASIE regions via rasterstats.zonal_stats function
    """
    # zonal_stats works on masked arrays,
    # so need to set values in mask to nan
    rate = round(marr[~marr.mask].shape[0] / marr.flatten().shape[0], 3) * 100
    if rate == 0:
        rate = None

    return rate


def concat_dict_lists(di_list1, di_list2):
    """Concatenate two equal-length lists of single-valued dicts
    into single list of dicts
    """
    for d1, d2 in zip(di_list1, di_list2):
        d1["mean"] = d2["mean"]
        
    return di_list1


def fix_zs_df_naming(zs_df):
    """Clean up the naming in the zonal stats dataframe to improve 
    readability as standalone table and with later plotting
    
    Args:
        zs_df (pandas.DataFrame): results of zonal statistics on both 
            indicators datasets
    """
    import pandas as pd
    
    # modify zonal stats dataframe for plotting

    # give indicators better names
    fubu_vars_lu = {
        "breakup_start": "Break-up Start",
        "breakup_end": "Break-up End",
        "freezeup_start": "Freeze-up Start", 
        "freezeup_end": "Freeze-up End", 
    }
    orac_vars_lu = {
        "DOO": "Day of Opening",
        "DOR": "Day of Retreat",
        "DOA": "Day of Advance", 
        "DOC": "Day of Closing", 
    }
    zs_df = zs_df.replace(fubu_vars_lu).replace(orac_vars_lu)

    # specify indicator type for facetting
    indicator_type_lu = {
        "Break-up Start": "breakup",
        "Break-up End": "breakup",
        "Freeze-up Start": "freezeup",
        "Freeze-up End": "freezeup",
        "Day of Opening": "breakup",
        "Day of Retreat": "breakup",
        "Day of Advance": "freezeup",
        "Day of Closing": "freezeup",
    }

    # specify indicator event for grouping by figure
    indicator_event_lu = {
        "Break-up Start": "start",
        "Break-up End": "end",
        "Freeze-up Start": "start",
        "Freeze-up End": "end",
        "Day of Opening": "start",
        "Day of Retreat": "end",
        "Day of Advance": "start",
        "Day of Closing": "end",
    }

    zs_df["indicator_type"] = zs_df["indicator"].copy().replace(indicator_type_lu)
    zs_df["indicator_event"] = zs_df["indicator"].copy().replace(indicator_event_lu)
    zs_df["fubu_indicator"] = zs_df["indicator_type"].values + "_" + zs_df["indicator_event"].values
    # make categorical for ordered plotting
    zs_df["fubu_indicator"] = pd.Categorical(zs_df["fubu_indicator"], categories=list(fubu_vars_lu.keys()), ordered=True)
    
    return zs_df


def run_zonal_stats(fubu, orac, masie_polys, affine):
    """Run the zonal stats computations and clean 
    up the results into a tidy data frame
    
    Args:
        fubu (xarray.Dataset): FUBU indicator dataset
        orac (xarray.Dataset): FUBU indicator dataset
        masie_polys (pandas.DataFrame): polygons of MASIE regions
        affine (affine.affine): affine transform for the common EPSG:3411 indicators grid
        
    Returns:
        Tidy data frame of zonal stats results by MASIE region for both 
            indicators datasets 
    """
    import numpy as np
    import pandas as pd
    from rasterstats import zonal_stats

    # take means of regions for only the cells that have 70% or higher definition rates
    # ignore the following regions: baltic, cook inlet, yellow sea, okhotsk
    masie_discard = ["Baltic Sea", "Sea of Okhotsk", "Yellow Sea", "Cook Inlet"]
    masie_polys_lm = masie_polys[~masie_polys["Name"].isin(masie_discard)]

    zs = {"J&E": {}, "Bliss": {}}
    for group, ds in zip(zs.keys(), [fubu, orac]):
        for variable in list(ds.variables)[:4]:
            # store mask of grid cells that don't meet definition rate threshold
            indicator_arr = ds[variable].values.copy()
            n = indicator_arr.shape[0]
            # array of definition rates for domain
            rate_arr = (~np.isnan(indicator_arr)).sum(axis=0) / n
            invalid_mask = rate_arr < 0.7

            zs[group][variable] = []
            for arr in indicator_arr:
                # first, determine success rates of indicators 
                #   (i.e. percentage of pixels in region where indicator was defined)
                # this is done before the mean aggregation below because that relies
                #   on filtering by definition rate
                zs_rates = zonal_stats(
                    masie_polys_lm, 
                    arr, 
                    affine=affine,
                    nodata=np.nan, 
                    # need to have a stats function specified or it 
                    #   will default to multiple stats
                    stats=["count"], 
                    add_stats={"rate": success_rate},
                )
                # remove count stat before adding
                zs_rates = [{"rate": zs_di["rate"]} for zs_di in zs_rates]

                # second, aggregate by mean for only those pixels in a region
                #   that have high enough individual success rate over the years
                # set pixels with success rates < 0.7 to np.nan
                arr[invalid_mask] = np.nan
                # then aggregate and append
                zs_means = zonal_stats(
                    masie_polys_lm, 
                    arr, 
                    affine=affine,
                    nodata=np.nan, 
                    stats=["mean"],
                )
                # concatenate with success rate list and append
                zs[group][variable].append(concat_dict_lists(zs_rates, zs_means))

            all_year_series = []
            for year_stats, year in zip(zs[group][variable], ds["year"].values):
                mean_series = pd.Series({
                    region_name: region_stats["mean"] 
                    for region_stats, region_name in zip(year_stats, masie_polys.Name)
                }, name = "mean_date")
                mean_series.index.name = "region"
                year_df = mean_series.reset_index()
                year_df["def_rate"] = [region_stats["rate"] for region_stats in year_stats]
                year_df["year"] = year
                year_df["indicator"] = variable
                year_df["group"] = group
                all_year_series.append(year_df)

            zs[group][variable] = pd.concat(all_year_series)

    zs["J&E"] = pd.concat(zs["J&E"]).reset_index().drop(columns=["level_1", "level_0"])
    zs["Bliss"] = pd.concat(zs["Bliss"]).reset_index().drop(columns=["level_1", "level_0"])
    zs_df = pd.concat(zs).reset_index().drop(columns=["level_1", "level_0"])
    
    # improve naming
    zs_df = fix_zs_df_naming(zs_df)
    
    return zs_df



def make_violin_plots(zs_df, stat_name, palette, ylab, out_fp):
    """Make violin plot from data frame output from run_zonal_stats

    Args:
        zs_df (pandas.DataFrame): output from run_zonal_stats filtered to indicator
        stat_name (str): name of statistic to plot
        palette (str): name of color palette to use
        ylab (str): Y-axis label
        out_fp (pathlib.PosixPath): absolute path to write location for image

    Returns: None, writes image to out_fp
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    g = sns.catplot(
        x="region",
        y=stat_name,
        hue="group",
        row="fubu_indicator",
        kind="violin",
        data=zs_df,
        palette=palette,
        split=True,
        inner="stick",
        linewidth=1,
        color="black",
        height=2.5, 
        aspect=3.6,
        sharey=False,
        legend=False,
    )

    axes = g.axes.flatten()
    titles = [
        "Break-up Start / Day of Opening",
        "Break-up End / Day of Retreat",
        "Freeze-up Start / Day of Advance",
        "Freeze-up End / Day of Closing",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_ylabel("")


    # offset tick label placement to help display long region names
    for tick in axes[3].xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

    axes[3].xaxis.labelpad = 15
    axes[3].xaxis.get_label().set_fontsize(13)
    plt.xlabel("MASIE Region", size=16)
    g.fig.text(-0.01, 0.51, ylab, va="center", rotation="vertical", fontsize=14)
    g.add_legend(title="", fontsize=14, bbox_to_anchor=(1, 0.49), loc="right", handletextpad=0.2)

    g.fig.set_size_inches(10, 10)

    save_fig(out_fp)
    
    return


def get_ma_data(ma):
    """Helper function for getting only valid data
    from a masked array
    
    Args:
        ma (numpy.ma.masked_array): masked array
    
    Returns:
        array where all masked values are set to np.nan
    """
    import numpy as np
    
    ma.data[ma.mask == True] = np.nan
    
    return ma.data


def get_zonal_pixel_means(ds, names_lu, polys, affine):
    """Get the means of pixels within polygons using rasterstats.zonal_stats
    
    Args:
        ds (xarray.dataset): An indicators dataset
        names_lu (dict): a names lookup structured as {<dataset varname>: <new varname>}
        polys (pandas.DataFrame): a data.frame of polygon geometries
        affine (affine.affine): affine transform for the common EPSG:3411 indicators grid
        
    Returns:
        Dict of means of within polys
    """
    import warnings
    import numpy as np
    from rasterstats import zonal_stats
    
    zonal_pixel_means = {}
    for ds_varname, new_varname in zip(names_lu.keys(), names_lu.values()):
        # for each variable (indicator), 
        # use zonal_stats to get the data for a single year
        data_zs = [zonal_stats(
            polys,
            arr, 
            affine=affine, # same affine transform
            nodata=np.nan, 
            add_stats={"data": get_ma_data}
        ) for arr in ds[ds_varname].values]
        
        # and unpack years to concatenate all data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            zonal_pixel_means[new_varname] = {
                row[1]["Name"]: np.nanmean(
                    np.stack([year_stats[row[0]]["data"] for year_stats in data_zs]),
                    axis=0
                )
                for row in polys.iterrows()
            }
    
    return zonal_pixel_means


def remove_nan(arr):
    """Helper function to remove NaNs from an arr"""
    import numpy as np
    
    return arr[~np.isnan(arr)]


def make_masie_mean_histograms(fubu, orac, masie_polys, affine, out_fp):
    """Plot histograms of means within MASIE regions and write image to out_fp
    
    Args:
        fubu (xarray.Dataset): FUBU indicator dataset
        orac (xarray.Dataset): FUBU indicator dataset
        masie_polys (pandas.DataFrame): polygons of MASIE regions
        affine (affine.affine): affine transform for the common EPSG:3411 indicators grid
        out_fp (pathlib.PosixPath): path to write image to
        
    Returns: None, writes image to out_fp
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    masie_keep_hist = ["Beaufort Sea", "Chukchi Sea", "E. Siberian Sea", "Laptev Sea"]
    masie_polys_hist = masie_polys[masie_polys["Name"].isin(masie_keep_hist)]

    fubu_names_lu = {
        "freezeup_start": "Freeze-up Start", 
        "freezeup_end": "Freeze-up End", 
        "breakup_start": "Break-up Start", 
        "breakup_end": "Break-up End",
    }
    fubu_masie_pixel_means = get_zonal_pixel_means(fubu, fubu_names_lu, masie_polys_hist, affine)

    orac_names_lu = {
        "DOA": "Day of Advance", 
        "DOC": "Day of Closing", 
        "DOO": "Day of Opening", 
        "DOR": "Day of Retreat",
    }
    orac_masie_pixel_means = get_zonal_pixel_means(orac, orac_names_lu, masie_polys_hist, affine)
    
    fubu_varnames = ["Break-up Start", "Break-up End", "Freeze-up Start", "Freeze-up End"]

    orac_varname_lu = {
        "Freeze-up Start": "Day of Advance",
        "Freeze-up End": "Day of Closing",
        "Break-up Start": "Day of Opening",
        "Break-up End": "Day of Retreat",
    }

    fig, axs = plt.subplots(4, 4, figsize=(17,16))
    for varname, i in zip(fubu_varnames, range(4)):
        for region_name, j in zip(fubu_masie_pixel_means[varname], range(4)):
            fubu_arr = remove_nan(fubu_masie_pixel_means[varname][region_name])
            orac_varname = orac_varname_lu[varname]
            orac_arr = remove_nan(orac_masie_pixel_means[orac_varname][region_name])
            counts, bins = np.histogram(np.concatenate([fubu_arr, orac_arr]))
            axs[(i,j)].hist(fubu_arr, alpha=0.5, label=varname, bins=bins)
            axs[(i,j)].hist(orac_arr, alpha=0.5, label=orac_varname, bins=bins)
            if i == 0:
                axs[(i,j)].set_title(region_name, {"fontsize": 14})

    # row labels
    row_label_x = 0.91
    fig.text(row_label_x, 0.81, "Break-up Start/\nDay of Opening", va="center", fontsize=14)
    fig.text(row_label_x, 0.6, "Break-up End/\nDay of Retreat", va="center", fontsize=14)
    fig.text(row_label_x, 0.4, "Freeze-up Start/\nDay of Advance", va="center", fontsize=14)
    fig.text(row_label_x, 0.2, "Freeze-up Start/\nDay of Closing", va="center", fontsize=14)

    fig.text(0.51, 0.085, "Day of year", ha="center", fontsize=14)
    fig.text(0.085, 0.51, "Pixel count", va="center", rotation="vertical", fontsize=14)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, ["J&E", "Bliss"], loc="right", fontsize=14)

    save_fig(out_fp)

    plt.show()
    
    return


def load_poi_coords(poi_coords_fp):
    """Read the CSV containing the point-of-interest 
    coordinates determined in points_of_interest.ipynb"""
    import pandas as pd
    
    poi_coords = pd.read_csv(poi_coords_fp)
    
    return poi_coords


def open_rasterio(fp):
    """open the seamask GeoTIFF created in points_of_interest.ipynb"""
    import rasterio as rio
    
    src = rio.open(fp)
    
    return src


def load_world_shore(gshhs_fp):
    """Load the world shore shapefile"""
    import geopandas as gpd
    
    world_shore = gpd.read_file(gshhs_fp)
    world_shore = world_shore.to_crs(epsg=3411).set_index("id")
    
    return world_shore


def make_poi_kwargs(poi_coords, seamask_src, fast_ice_gdf, world_shore):
    """Make the kwargs for working with Points-of-interest"""
    poi_kwargs = {}

    for location, df in poi_coords.groupby("location"):
        poi_kwargs[location] = {
            "poi_name": location,
            "xy_list": [
                (row[1]["x"], row[1]["y"]) 
                for row in df.iterrows()
            ],
            "scale": 2,
            "seamask_src": seamask_src,
            "fast_ice_gdf": fast_ice_gdf,
            "world_shore": world_shore,
        }

    # add aditional info as dicts for plotting
    poi_kwargs["Utqiaġvik"]["map_di"] = {
        "poi_wgs84":(71.2906, -156.7886),
        "poi_name_adj": (-1.2, -0.4),
        "text": ["Chukchi\nSea", "Beaufort\nSea"],
        "text_adj": [(-1.6, 1), (0.7, 1)]
    }
    poi_kwargs["South Chukchi Sea"]["map_di"] = {
        "poi_wgs84": (68.3478, -166.8081),
        "poi_name_adj": "none",
        "text": ["Bering\nStrait", "Chukchi Sea"],
        "text_adj": [(-1.2, -2), (-1.5, 0.8)]
    }
    poi_kwargs["St. Lawrence Island"]["map_di"] = {
        "poi_wgs84": (62.9, -169.6),
        "poi_name_adj": "none",
        "text": ["Bering Sea", "St. Lawrence\nIsland", "Bering\nStrait"],
        "text_adj": [(-1, -1.25), (0.25, 0.65), (-0.5, 1.4)]
    }
    poi_kwargs["Prudhoe Bay"]["map_di"] = {
        "poi_wgs84": (70.2, -148.2),
        "poi_name_adj": (-1.3, -0.5),
        "text": ["Beaufort Sea"],
        "text_adj": [(0.3, 1.5)]
    }
    poi_kwargs["Pevek"]["map_di"] = {
        "poi_wgs84": (69.8, 170.6),
        "poi_name_adj": (0.2, -0.2),
        "text": ["East Siberian Sea"],
        "text_adj": [(-0.75, 1.4)]
    }
    poi_kwargs["Tiksi"]["map_di"] = {
        "poi_wgs84": (71.6, 128.9),
        "poi_name_adj": (-0.6, -0.5),
        "text": ["Laptev Sea"],
        "text_adj": [(0.75, 1.25)]
    }
    poi_kwargs["Sabetta"]["map_di"] = {
        "poi_wgs84": (71.3, 72.1),
        "poi_name_adj": (-0.7, -0.5),
        "text": ["Kara Sea"],
        "text_adj": [(-1.25, 2.75)]
    }
    poi_kwargs["Mestersvig"]["map_di"] = {
        "poi_wgs84": (72.2, -23.9),
        "poi_name_adj": (-1, -0.4),
        "text": ["Greenland\nSea"],
        "text_adj": [(1.1, 0.4)]
    }
    poi_kwargs["Clyde River"]["map_di"] = {
        "poi_wgs84": (70.3, -68.3),
        "poi_name_adj": (-1.4, -0.4),
        "text": ["Baffin Bay"],
        "text_adj": [(0.8, 0.75)],
    #     "text_bg": "#DFD8E1",
    }
    poi_kwargs["Churchill"]["map_di"] = {
        "poi_wgs84": (58.8, -94.2),
        "poi_name_adj": (-1.25, -0.4),
        "text": ["Hudson Bay"],
        "text_adj": [(0.6, 0.9)]
    }
    
    return poi_kwargs


def reproject_poi(poi, in_epsg=4326, out_epsg=3411):
    """Reproject WGS84 coordinates

    Args:
        poi (list/tuple): 2-tuple or 2 item list of
            coordinates of point to reproject in (<lat>, <lon>) format
        in_epsg (int): EPSG code of coords
        out_epsg (int): EPSG to reproject to

    Returns:
        tuple of x, y coordinates in out_epsg, formatted as (<x>, <y>)
    """
    from pyproj import Transformer
    
    transformer = Transformer.from_crs(in_epsg, out_epsg)

    return transformer.transform(*poi)


def clip_shore_to_viewing_extent(world_shore, xlims, ylims):
    """Clip the shore polygon to the extent derived from the xy lims"""
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point
    
    xd = xlims[1] - xlims[0]
    yd = ylims[1] - ylims[0]
    r = np.sqrt(xd ** 2 + yd ** 2)
    # maximum viewing extent will be circle centered on middle of 
    # viewing window, with diameter equal to diagnol of view window
    view_circle = Point((xd / 2 + xlims[0], yd / 2 + ylims[0])).buffer(r/2)
    view_gdf = gpd.GeoDataFrame(geometry=[view_circle]).set_crs(3411)
    
    return gpd.overlay(view_gdf, world_shore, how="intersection")
    

def make_pixel_polygon_from_xy(transform, x, y):
    """Make polygon corresponding to raster pixels
    from given xy centerpoint and the raster's Affine transform
    """
    from shapely.geometry import Polygon
    
    res = transform[0]
    ul = (x - (res / 2), y + (res / 2))
    corner_list = [
        ul,
        (ul[0] + res, ul[1]),
        (ul[0] + res, ul[1] - res),
        (ul[0], ul[1] - res),
        ul,
    ]
    
    return Polygon([corner for corner in corner_list])


def make_pixel_poly_gdf(transform, xy_list):
    """make GeoPandas DF of pixel polygons from xy list and transform"""
    import geopandas as gpd
    
    polys = [make_pixel_polygon_from_xy(transform, *xy) for xy in xy_list]

    return gpd.GeoDataFrame(geometry=polys)


def angle_between(v1, v2):
    """Get the angle in degrees between two 2D vectors, supplied as tuples
    
    Thanks https://stackoverflow.com/a/2150111/11417211"""
    import math
    
    angle = (math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])) * 57.2958
        
    return angle


def plot_poi_pixel_polys(
    xy_list, 
    scale, 
    seamask_src,
    fast_ice_gdf,
    world_shore, 
    poi_name, 
    map_di,
    ax
):
    """Plot the polygons depicting the pixels to be used for the points of interest

    Args:
        xy_list (list): list of 2-tuples of (x, y) coordinates for pixels used
        scale (float): values for scaling viewing window
        seamask_src (rasterio.DatasetReader): seamask raster
        fast_ice_fp (file_path): path to fast ice climatology file
        world_shore (geopandas.GeoDataFrame): shapefile of world shoreline
        poi_wgs84 (tuple): lat/lon coordinates in form (lat, lon)
        poi_name (str): display name for point of interest
        ax (matplotlib.axes._subplots.AxesSubplot): axis to plot on
    """
    import geopandas as gpd
    import numpy as np
    import rasterio as rio
    from matplotlib_scalebar.scalebar import ScaleBar
    from rasterio.plot import show
    from shapely.geometry import Point
    
    # unpack map_di values
    poi_wgs84 = map_di["poi_wgs84"]
    poi_name_adj = map_di["poi_name_adj"]
    text = map_di["text"]
    text_adj = map_di["text_adj"]
    
    # get x and y limits for viewing window based on 
    #   scale and poi_xy
    # choose the centerpoint of the viewing window to be the 
    #   midpoint between a pixel selection and the poi
    poi_xy = reproject_poi(poi_wgs84)
    center_xy = [0, 0]
    for i in range(2):
        min_val = np.min((poi_xy[i], xy_list[0][i]))
        center_xy[i] = min_val + (np.max((poi_xy[i], xy_list[0][i])) - min_val) / 2
        
    xlims = (center_xy[0] - scale * 1e5, center_xy[0] + scale * 1e5)
    ylims = (center_xy[1] - scale * 1e5, center_xy[1] + scale * 1e5)
    
    # get polygon of shoreline within viewing window
    shore_poly = clip_shore_to_viewing_extent(world_shore, xlims, ylims)
    
    # make pixel polygons
    pixel_polys = make_pixel_poly_gdf(seamask_src.transform, xy_list)
    
    # rotate the shapefiles
    # two vectors we need different in angles from:
    # v1: unit arrow pointing straight up
    # v2: center of viewing window to center of raster
    v1 = (0, 1)
    domain_cp = fast_ice_gdf.unary_union.centroid.xy
    v2 = (domain_cp[0] - center_xy[0], domain_cp[1] - center_xy[1])
    north_angle = angle_between(v2, v1)
    
    #fast_ice_gdf.rotate(north_angle, origin=gdf.unary_union.centroid)
    fast_rot_gdf = fast_ice_gdf.rotate(north_angle, origin=center_xy)
    shore_rot_gdf = shore_poly.rotate(north_angle, origin=center_xy)
    pixel_rot_gdf = pixel_polys.rotate(north_angle, origin=center_xy)
    
    poi_xy_gdf = gpd.GeoDataFrame(geometry=[Point(poi_xy)])
    poi_xy_rot_gdf = poi_xy_gdf.rotate(north_angle, origin=center_xy)
    
    def fast_ice_cmapper(values):
        cmap = {
            -999: "white",
            0: "#DFD8E1", 
            1: "lightblue",
        }
        return [cmap[v] for v in values]

    fast_rot_gdf.plot(ax=ax, color=fast_ice_cmapper(fast_ice_gdf["value"]), edgecolor="none")
    # plot shorelines
    shore_rot_gdf.plot(ax=ax, facecolor="none", edgecolor="dimgray")
    # plot cells that will be used
    pixel_rot_gdf.plot(ax=ax, facecolor="none", edgecolor="red")
    # add scalebar
    ax.add_artist(ScaleBar(1, location="lower right"))
    
    # show the point of interest
    # this allows to not show a coastal point of interest as a dot,
    #  which is the case for one location -__-
    poi_rot_xy = (poi_xy_rot_gdf[0].x, poi_xy_rot_gdf[0].y)
    if poi_name_adj != "none":
        #ax.scatter(*poi_xy, color="black")
        ax.scatter(*poi_rot_xy, color="black")
        # add the adjustments to the point of interest coords
        poi_name_adj = (poi_name_adj[0] * 1e5, poi_name_adj[1] * 1e5)
        poi_name_xy = [sum(x) for x in zip(poi_rot_xy, poi_name_adj)]
        bbox_props = dict(boxstyle="round", facecolor="#DFD8E1", edgecolor="none", alpha=0.75)
        ax.text(*poi_name_xy, poi_name, fontsize=10, bbox=bbox_props)

    # display other text
    # add the adjustments for the text coords
    text_adj = [(t[0] * 1e5, t[1] * 1e5) for t in text_adj]
    text_xy = [[sum(x) for x in zip(poi_rot_xy, t_adj)] for t_adj in text_adj]
    _ = [ax.text(*t_xy, t, fontsize=10) for t, t_xy in zip(text, text_xy)]
    
    # centerpoint in normalized coords for North arrow graphic
    north_cp = (0.05, 0.9)
    yadj = 0.06
    # base of arrow
    xytext = (north_cp[0], north_cp[1] - yadj)
    xy = (north_cp[0], north_cp[1] + yadj)
    ax.annotate(
        "N", 
        xy=xy,
        xycoords=ax.transAxes,
        xytext=xytext,
        textcoords=ax.transAxes,
        fontsize=12,
        arrowprops={"facecolor": "black", "linewidth": 0, "headwidth": 10, "headlength": 8, "shrink": 0},
        ha="center",
        va="center",
    )
    
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    return


def make_poi_maps(poi_coords, world_shore, seamask_src, fast_ice_fp, out_fp):
    """Make a plot of the maps showing the cells 
    used for the points of interest. Include landfast sea ice
    climatologies for June.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import geopandas as gpd

    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    
    fast_ice_gdf = gpd.read_file(fast_ice_fp)
    poi_kwargs = make_poi_kwargs(poi_coords, seamask_src, fast_ice_gdf, world_shore)

    for ax, location in zip(axs.flatten(), poi_kwargs.keys()):
        plot_poi_pixel_polys(**poi_kwargs[location], ax=ax)

    gs = axs[2, 2].get_gridspec()
    # remove the underlying axes
    for ax in axs[2, 2:]:
        ax.remove()
    axbig = fig.add_subplot(gs[-1, 2:])

    # Clear bottom-right ax
    axbig.set_axis_off()  # removes the XY axes

    # Manually create legend handles (patches)
    land_patch = mpatches.Patch(color="#DFD8E1", label="Land mask")
    # fast_patch = mpatches.Patch(color="#7ba1c2", label="Landfast ice")
    fast_patch = mpatches.Patch(color="lightblue", label="Landfast ice")

    #pixel_patch = mpatches.Patch(color="red", fill=False, label="Pixels selected for \npoint location")
    pixel_line = Line2D(
        [0], [0], 
        marker='s', 
        color="w", 
        label='Pixels selected for \npoint location', 
        markeredgecolor="r", 
        markerfacecolor="w", 
        markersize=14
    )

    # Add legend to bottom-right ax
    axbig.legend(handles=[land_patch, fast_patch, pixel_line], loc="center", fontsize=14)

    plt.tight_layout()
    save_fig(out_fp)

    plt.show()

    
def aggregate_pixels(ds, var_dict, xy_list, group, point_name):
    """given an xy_list and variables, make a dataframe by querying an 
    xarray dataset at the given xy locations and aggregating via mean
    """
    import warnings
    import numpy as np
    import pandas as pd
    
    # empty list for holding DFs
    df_list = []
    # make a data frame for a single dataset and variable
    with warnings.catch_warnings():
        # ignore "mean of empty slice" warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for var in var_dict.keys():
            df_list.append(
                pd.DataFrame.from_dict(
                    {
                        "point_name": point_name,
                        "indicator": var_dict[var],
                        "year": ds["year"].values,
                        "mean_date": np.nanmean(
                            np.array([
                                ds[var].sel(xc=xy[0], yc=xy[1], method="nearest").values 
                                for xy in xy_list
                            ]), axis=0),
                        "group": group,
                    }
                )
            )
        
    return pd.concat(df_list)


def get_poi_data(poi_coords, fubu, orac):
    import pandas as pd
    
    # aggregate pixels according to new points of interest coordinates

    fubu_vars = {
        "breakup_start": "Break-up Start / Day of Opening",
        "breakup_end": "Break-up End / Day of Retreat",
        "freezeup_start": "Freeze-up Start / Day of Advance", 
        "freezeup_end": "Freeze-up End / Day of Closing", 
    }
    orac_vars = {
        "DOO": "Break-up Start / Day of Opening",
        "DOR": "Break-up End / Day of Retreat",
        "DOA": "Freeze-up Start / Day of Advance", 
        "DOC": "Freeze-up End / Day of Closing", 
    }

    point_data = pd.DataFrame()
    for location, df in poi_coords.groupby("location"):
        pixels_xy_list = list(df[["x", "y"]].to_records(index=False))
        point_data = point_data.append(aggregate_pixels(fubu, fubu_vars, pixels_xy_list, "J&E", location))
        point_data = point_data.append(aggregate_pixels(orac, orac_vars, pixels_xy_list, "Bliss", location))

    return point_data


def plot_poi_trends(poi_data, indicator, content_dir, output_format):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plot_data = poi_data[poi_data["indicator"] == indicator]
    lm = sns.lmplot(
        x="year", 
        y="mean_date", 
        col="point_name", 
        col_wrap=5, 
        hue="group", 
        data=plot_data, 
        sharey=False, 
        height=3, 
        legend=False
    )
    # set the limits of xaxis to avoid cutting off points
    lm.set(xlim=(1978, 2019))

    axes = lm.axes.flatten()
    for ax, name in zip(axes.flatten(), poi_data["point_name"].unique()):
        ax.set_title(name)
        ax.set_ylabel("")
        ax.set_xlabel("")

    lm.fig.text(0.485, 1, indicator, ha="center", fontsize=14)
    lm.fig.text(0.485, -0.03, "Year", ha="center", fontsize=14)
    lm.fig.text(-0.01, 0.51, "Mean day of year", va="center", rotation="vertical", fontsize=14)

    lm.fig.subplots_adjust(hspace=0.25)
    plt.subplots_adjust(right=0.88, top=0.92)

    lm.add_legend(title="", fontsize=14, bbox_to_anchor=(1, 0.49), loc="right", handletextpad=0.1)
    
    suffix = indicator.replace(" ", "_").split("_/_")[0].lower().replace("-", "")
    out_fp = content_dir.joinpath(f"point_trends_plots_{suffix}.{output_format}")
    save_fig(out_fp)

    plt.show()
    
    
def make_poi_results_table(poi_df, out_fp):
    """Get the results of fitting linear models to the POI data"""
    import numpy as np
    import pandas as pd
    from scipy import stats
    
    # Add column to differentiate indicators between groups
    # copy poi df but modifies slightly so need to copy
    tmp_poi_df = poi_df.copy()
    tmp_poi_df["indicator_name"] = ""
    combined_indicators = [
        "Break-up Start / Day of Opening", 
        "Break-up End / Day of Retreat", 
        "Freeze-up Start / Day of Advance", 
        "Freeze-up End / Day of Closing"
    ]
    for indicator in combined_indicators:
        je_ind, bliss_ind = indicator.split(" / ")
        tmp_poi_df.loc[(tmp_poi_df["group"] == "J&E") & (tmp_poi_df["indicator"] == indicator), "indicator_name"] = je_ind
        tmp_poi_df.loc[(tmp_poi_df["group"] == "Bliss") & (tmp_poi_df["indicator"] == indicator), "indicator_name"] = bliss_ind
        
    rows = []
    for groups, df in tmp_poi_df.groupby(["point_name", "group", "indicator_name"]):
        df = df[~np.isnan(df["mean_date"])]
        slope, intercept, r_value, p_value, std_err = stats.linregress(df["year"], df["mean_date"])
        if p_value < 0.01:
            p_value = "< 0.01**"
        elif p_value <= 0.05:
            p_value = str(round(p_value, 2)) + "*"
        else:
            p_value = str(round(p_value, 2))

        rows.append([*groups, round(slope, 1), round(r_value, 2), p_value])

    poi_lm_results = pd.DataFrame(rows, columns=["Location", "Indicator Group", "Indicator", "Slope", "r2", "p"])

    # do some ordering and save
    indicator_names = [i for l in [[ind.split(" / ")[i] for ind in combined_indicators] for i in (0, 1)] for i in l]
    poi_lm_results["Indicator"] = pd.Categorical(poi_lm_results["Indicator"], categories=indicator_names, ordered=True)
    poi_lm_results.sort_values(["Location", "Indicator Group", "Indicator"])

    poi_lm_results.to_csv(out_fp, index=False)
    print(f"Point trends results written to {out_fp}")
    
    return poi_lm_results


def concat_masie_poi_data(zs_df, poi_data):
    
    # make a lookup table for region to poi
    poi_region_lu = {
        "Churchill": "Hudson Bay",
        "Clyde River": "Baffin Bay",
        "Mestersvig": "Greenland Sea",
        "Pevek": "E. Siberian Sea",
        "Sabetta": "Kara Sea", 
        "St. Lawrence Island": "Bering Sea", 
        "Tiksi": "Laptev Sea",
        "Prudhoe Bay": "Beaufort Sea",
        "Utqiaġvik": "Chukchi Sea",
        "South Chukchi Sea": "Chukchi Sea", 
    }

    # wrangle the data frames to be concat-able

    # subset data frames to J&E before joining
    poi_data_je = poi_data[poi_data["group"] == "J&E"].drop(columns="group")
    zs_df_je = zs_df[zs_df["group"] == "J&E"].drop(
        columns=["group", "indicator_event", "indicator_type", "fubu_indicator", "def_rate"]
    )

    # replace inclusive indicator names with just FUBU names
    poi_data_je["indicator"] = [
        name.split(" / ")[0] for name in poi_data_je["indicator"]
    ]

    # add region info using lookup
    poi_data_je["region"] = poi_data_je["point_name"].map(poi_region_lu)

    # add location info to region df using reverse lu
    zs_df_je["point_name"] = zs_df_je["region"].map(
        dict((v, k) for k, v in poi_region_lu.items()), na_action="ignore"
    )
    # drop data without point name of interest
    zs_df_je = zs_df_je[~zs_df_je["point_name"].isnull()]

    # # add variable for coloring by location/region
    poi_data_je["type"] = "Location"
    zs_df_je["type"] = "MASIE\nregion"

    # reorder cols and append
    poi_masie_df = poi_data_je.append(zs_df_je[poi_data_je.columns])

    # need to copy Chukchi data and re-apply with the South Chukchi Sea location
    # since we have two locations in the Chukchi
    chukchi_temp = poi_masie_df[
        (poi_masie_df["region"] == "Chukchi Sea")
        & (poi_masie_df["type"] == "MASIE\nregion")
    ].copy()
    chukchi_temp["point_name"] = chukchi_temp["point_name"].replace(
        {"South Chukchi Sea": "Utqiaġvik"}
    )
    poi_masie_df = poi_masie_df.append(chukchi_temp)

    # create a column for displaying both location and region names
    poi_masie_df["region_location"] = poi_masie_df["point_name"] + " /\n" + poi_masie_df["region"]
    
    return poi_masie_df


def make_masie_poi_trends_plots(poi_masie_df, indicator, content_dir, output_format):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plot_data = poi_masie_df[poi_masie_df["indicator"] == indicator]
    
    p = sns.color_palette()
    lm = sns.lmplot(
        x="year", 
        y="mean_date", 
        col="region_location", 
        col_wrap=5, 
        hue="type", 
        data=plot_data, 
        sharey=False, 
        height=3, 
        legend=False,
        palette=sns.color_palette([p[0], p[6]])
    )
    # set the limits of xaxis to avoid cutting off points
    lm.set(xlim=(1978, 2019))

    axes = lm.axes.flatten()
    for ax, name in zip(axes.flatten(), plot_data["region_location"].unique()):
        ax.set_title(name)
        ax.set_ylabel("")
        ax.set_xlabel("")

    lm.fig.text(0.47, 1.025, indicator, ha="center", fontsize=14)
    lm.fig.text(0.485, -0.03, "Year", ha="center", fontsize=14)
    lm.fig.text(-0.01, 0.51, "Mean day of year", va="center", rotation="vertical", fontsize=14)

    lm.fig.subplots_adjust(hspace=0.25)
    plt.subplots_adjust(right=0.88, top=0.92)

    lm.add_legend(title="", fontsize=14, bbox_to_anchor=(1, 0.49), loc="right", handletextpad=0.1)
    
    suffix = indicator.replace(" ", "_").lower().replace("-", "")
    save_fig(content_dir.joinpath(f"masie_point_trends_plots_{suffix}.{output_format}"))

    plt.show()
    
    return

    
    
def make_masie_trends_plots(zs_df, indicator, out_fp):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    masie_discard = ["Baltic Sea", "Sea of Okhotsk", "Yellow Sea", "Cook Inlet"]
    plot_df = zs_df[~zs_df["region"].isin(masie_discard)]
    
    plot_df = plot_df[plot_df["fubu_indicator"] == indicator]
    lm = sns.lmplot(x="year", y="mean_date", col_wrap=4, col="region", hue="group", data=plot_df, sharey=False, height=3, legend=False)
    # set the limits of xaxis to avoid cutting off points
    lm.set(xlim=(1978, 2019))
    
    axes = lm.axes.flatten()
    for ax in axes:
        ax.set_title(ax.get_title().split(" = ")[1])
        ax.set_ylabel("")
        ax.set_xlabel("")

    lm.fig.text(0.46, -0.01, "Year", ha="center", fontsize=14)
    lm.fig.text(-0.01, 0.5, "Mean day of year", va="center", rotation="vertical", fontsize=14)

    # lm._legend.set_title("")
    # plt.setp(lm._legend.get_texts(), fontsize=16)
    lm.fig.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(right=0.9, top=0.92)

    lm.add_legend(title="", fontsize=14, frameon=True, borderaxespad=0.25)
    
    save_fig(out_fp)

    return