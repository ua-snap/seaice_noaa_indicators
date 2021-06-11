"""Create the tables and figures for the manuscript summarizing the
indicator trends and comprison with the NSIDC-0747 dataset

Usage:
    pipenv run python manuscript_content.py -n <number of CPUs> [-b] [-e]
    Script #X of data pipeline

Returns:
    Figures are written to $OUTPUT_DIR/manuscript_content/

Notes:
    J&E referes to the indicator dates that were developed within this
    project, i.e. that were produced from an algorithm derived
    from that used in the 2016 paper by Hajo Eicken and
    Mark Johnson. 
"""

import argparse
import copy
import os
import numpy as np
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio as rio
import seaborn as sns
import xarray as xr
from rasterio.crs import CRS
from rasterio.plot import show
from rasterio.features import shapes
from rasterstats import zonal_stats
from shapely.geometry import Polygon


def assimilate_orac(orac):
    """Replicate the data strcutre of FUBU dataset 
    in the ORAC dataset so they can be used with same functions

    Args:
        orac: Steele (NSIDC-0747) indicators xarray.dataset
        fubu: J&H indicators xarray.dataset
        
    Returns:
        ORAC dataset matching the structure of the FUBU indicators
            dataset
    """
    # retain only variables that correspond to FUBU indicators
    # rename coord vars to match
    varnames = ["DOO", "DOR", "DOA", "DOC"]
    orac = orac[varnames].rename({"x": "xc", "y": "yc", "time": "year"})
    # make time variable an integer with year
    orac = orac.assign_coords(time=[dt.year for dt in orac["year"].values])
    # set non-indicator date values to np.nan
    for var in varnames:
        orac[var].values[orac[var].values < 1] = np.nan

    return orac


def make_totals_maps(ds, varnames, landmask, titles, out_fp):
    """Make the totals maps - density plots /heatmaps of
    pixel-wise counts of years where indicator was defined -
    for a supplied indicators dataset and variable names

    Args:
        ds (xarray.Dataset): indicator Dataset
        varnames (list): names of variables in ds to count and plot totals for
        landmask (np.ndarray): array where values of true correspond to pixels
            that overlap land
        landmask_poly (geopandas.GeoDataFrame): polgyon of the
            landmask
        out_fp (path-like): absolute path to image write location

    Returns:
        None, writes image to out_fp
    """

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
    fig, axs = plt.subplots(1, 2)

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
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    print(f"\nTotals maps for {varnames[0]}, {varnames[1]} written to {out_fp}\n")

    return


def get_polygon(row):
    """Create shapely polygon object from row of alternating
    lat/lon coords

    Args:
        pandas series object from df.iterrows() of MAISE regions
        MS Excel file

    returns
        shapely polygon derived from vertices coordinates in row

    Notes:
        designed for use with MASIE_regions_polygon_vertices.xls
    """
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


def success_rate(marr):
    """Compute the percentage of non-nan values in masked array

    Args:
        marr: Masked numpy array of raster values within the polygon

    Returns:
        Percentage of non-nan values (i.e. valid dates) in marr

    Notes:
        Intended use is for deriving indicator success rates
        for MAISE regions via rasterstats.zonal_stats function
    """
    # zonal_stats works on masked arrays,
    # so need to set values in mask to nan
    rate = round(marr[~marr.mask].shape[0] / marr.flatten().shape[0], 3) * 100
    if rate == 0:
        rate = None

    return rate


def run_zonal_stats(
    ds, varnames, maise_polys, affine, add_stats={"rate": success_rate}
):
    """Wrapper for the rasterstats.zonal_stats function to 
    compute stats for the polygons in maise_polys and 
    output as tidy data frame

    Args:
        ds (xarray.Dataset): indicator Dataset
        varnames (list): names of variables in ds to summarize
        maise_polys (pandas.DataFrame): polygons of MAISE regions
        affine (array-like): affine transformation for raster
        add_stats (dict): dict of <stat name>: <stat function> for additional
            statistics

    Returns:
        Tidy data frame of stats of "rasters" in ds evaluated over polygons in
            maise_polys
    """
    zs_di = {}
    for varname in varnames:
        zs_di[varname] = [
            zonal_stats(
                maise_polys,
                arr,
                affine=affine,
                nodata=np.nan,
                stats=["mean"],
                add_stats=add_stats,
            )
            for arr in ds[varname].values
        ]

    # unpack values from zs_di into tidy data frame by repeating/tiling
    stats_df = pd.DataFrame(
        {
            "region": np.tile(
                maise_polys["Name"].values, len(varnames) * len(zs_di[varnames[0]])
            ),
            "indicator": np.repeat(
                varnames, len(zs_di[varnames[0]]) * len(zs_di[varnames[0]][0])
            ),
        }
    )

    stat_names = list(zs_di[varnames[0]][0][0].keys())
    for stat in stat_names:
        stats_df[stat] = [
            region_stats[stat]
            for varname in varnames
            for year_stats in zs_di[varname]
            for region_stats in year_stats
        ]

    return stats_df


def make_violin_plots(zs_df, stat_name, varname_di, palette, legend_loc, ylab, out_fp):
    """Make violin plot from data frame output from run_zonal_stats

    Args:
        zs_df (pandas.DataFrame): output from run_zonal_stats
        stat_name (str): name of statistic to plot
        varname_di (dict): dict for giving better indicator names for display
        palette (str): name of color palette to use
        legend_loc (str): argument to plt.legend for where to place it
        ylab (str): Y-axis label
        out_fp (path-like): absolute path to write location for image

    Returns: None, writes image to out_fp
    """
    # filter df to varnames in varname_di
    zs_df = zs_df[zs_df["indicator"].isin(list(varname_di.keys()))]
    # change varnames as specified for display
    fig = plt.figure(figsize=(18, 6))
    ax = sns.violinplot(
        x="region",
        y=stat_name,
        hue="indicator",
        data=zs_df.replace({"indicator": varname_di}),
        palette=palette,
        split=True,
        inner="stick",
    )

    # offset tick label placement to help display long region names
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

    ax.xaxis.labelpad = 15
    # plt.title(title, size=16)
    plt.xlabel("MASIE Region", size=16)
    plt.ylabel(ylab, size=16)
    plt.legend(loc=legend_loc, prop={"size": 14})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # fig.set_size_inches(10, 5)
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    print(f"\nViolin plots for written to {out_fp}\n")

    return


# import warnings
# import geopandas as gpd
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#

if __name__ == "__main__":
    # parse some args
    parser = argparse.ArgumentParser(
        description="Create the tables and figures for the trends manuscript",
    )
    parser.add_argument(
        "-ckw",
        "--content-keywords",
        action="store",
        dest="content_kws",
        default="all",
        help="Specify what content to create with ' '-separated \
            list of values (omit to create all tables and figures): \
            ** tm-je: J&E totals maps * tm-steele: Steele totals maps Steele * \
            ",
    )
    parser.add_argument(
        "-n",
        "--ncpus",
        action="store",
        dest="ncpus",
        type=int,
        default=8,
        help="Number of cpus to use",
    )
    parser.add_argument(
        "-of",
        "--output-format",
        action="store",
        dest="output_format",
        type=str,
        default="eps",
        help="Output format of figures",
    )

    # unpack the args
    args = parser.parse_args()
    content_kws = set(args.content_kws.split(" "))
    ncpus = args.ncpus
    output_format = args.output_format

    out_dir = Path(os.getenv("OUTPUT_DIR")).joinpath("manuscript_content")
    out_dir.mkdir(exist_ok=True)
    base_dir = Path(os.getenv("BASE_DIR"))

    # use sets of keyword args to determine what data
    # to read, process, and plot:
    # totals maps keywords
    tm_kws = {"tm"}
    # MAISE region keywords
    maise_kws = {"isr", "mdfu", "mdbu", "mrt"}
    # zonal stats keywords
    zs_kws = {"isr", "mdfu", "mdbu"}
    # kws for all plot types
    all_type_kws = tm_kws | maise_kws | zs_kws
    # Johnson & Eicken keywords
    je_kws = {kw + "-je" for kw in all_type_kws}
    # Steele keyword
    steele_kws = {kw + "-steele" for kw in all_type_kws}

    # ignore anything else if "all" is specified, set
    # content_kws to all possible content
    if "all" in content_kws:
        content_kws = je_kws | steele_kws

    # break content keywords apart to indicator group and plot type
    content_type_kws = {kw.split("-")[0] for kw in content_kws}
    content_group_kws = {kw.split("-")[1] for kw in content_kws}

    if "je" in content_group_kws:
        fubu_fp = base_dir.joinpath("nsidc_0051/outputs/nsidc_0051_1979-2019_fubu.nc")
        fubu = xr.load_dataset(fubu_fp)
        # Data for Steele only available through 2017. Reduce J&E timespan to match
        fubu = fubu.sel(year=slice(1979, 2017))

    if "steele" in content_group_kws:
        orac_fp = base_dir.joinpath(
            "nsidc_0747/arctic_seaice_climate_indicators_nh_v01r01_1979-2017.nc"
        )
        orac = xr.load_dataset(orac_fp)

        # Since a number of plots involve both Steele and J&H,
        # give them a similar data structure for consistent use
        # plotting functions
        orac = assimilate_orac(orac)

    if (content_type_kws & maise_kws) != set():
        # read MAISE region vertices and create polygons
        masie_fp = base_dir.joinpath("ancillary/MASIE_regions_polygon_vertices.xls")
        maise_df = pd.read_excel(masie_fp, skiprows=1)

        maise_polys = maise_df[["Region Number Identifier", "Name"]].copy()
        for index, row in maise_df.iterrows():
            maise_polys.loc[index, "geometry"] = get_polygon(row)

        maise_polys = gpd.GeoDataFrame(maise_polys).set_crs(epsg=4326).to_crs(epsg=3411)
        maise_polys = maise_polys.replace(
            {
                "region": {
                    "Baffin Bay/Gulf of St. Lawrence": "Baffin Bay",
                    "Canadian Archipelago": "Canadian Arch.",
                    "East Siberian Sea": "E. Siberian Sea",
                }
            }
        )

    # totals maps
    if (content_type_kws & tm_kws) != set():
        orac_fp = base_dir.joinpath(
            "nsidc_0747/arctic_seaice_climate_indicators_nh_v01r01_1979-2017.nc"
        )
        with xr.open_rasterio(f"netcdf:{orac_fp}:DOA") as da:
            landmask = da.sel(band=1).values == -4

        #         with rio.open(f"netcdf:{orac_fp}:DOA") as src:
        #             meta = src.meta
        #         meta["crs"] = CRS.from_epsg(3411)
        #         # in addition to landmask, make polygon of landmask outline
        #         polys = (
        #             {'properties': {'raster_val': v}, 'geometry': s}
        #             for i, (s, v) in enumerate(shapes(
        #                 landmask.astype(np.int16), landmask, transform=meta["transform"]
        #             ))
        #         )
        #         landmask_poly = gpd.GeoDataFrame.from_features(list(polys)).dissolve("raster_val")

        # make the totals plots
        if "tm-je" in content_kws:
            make_totals_maps(
                fubu,
                ["breakup_start", "freezeup_start"],
                landmask,
                # landmask_poly,
                # meta,
                ["Break-up Start/End", "Freeze-up Start/End"],
                out_dir.joinpath(f"Johnson_Eicken_totals_maps.{output_format}"),
            )

        if "tm-steele" in content_kws:
            make_totals_maps(
                orac,
                ["DOR", "DOC"],
                landmask,
                # landmask_poly,
                # meta,
                ["Day of Retreat", "Day of Closing"],
                out_dir.joinpath(f"Steele_totals_maps.{output_format}"),
            )

    # make violin plots
    if (content_type_kws & zs_kws) != set():
        # need affine transform for zonal_stats()
        with rio.open(f"netcdf:{fubu_fp}:breakup_end") as src:
            affine = src.meta["transform"]

        if "je" in content_group_kws:
            fubu_zs = run_zonal_stats(
                fubu,
                ["breakup_start", "breakup_end", "freezeup_start", "freezeup_end"],
                polys,
                affine,
            )

            if "isr-je" in content_kws:
                make_violin_plots(
                    fubu_zs,
                    "rate",
                    {
                        "breakup_start": "Break-up Start/End",
                        "freezeup_start": "Freeze-up Start/End",
                    },
                    "PiYG",
                    "upper right",
                    "Definition rate (%)",
                    out_dir.joinpath(
                        f"Johnson_Eicken_definition_rate_violin_plots.{output_format}"
                    ),
                )

            if "mdfu-je" in content_kws:
                make_violin_plots(
                    fubu_zs,
                    "mean",
                    {
                        "freezeup_start": "Freeze-up Start",
                        "freezeup_end": "Freeze-up End",
                    },
                    "Set3",
                    "upper left",
                    "Day of year",
                    out_dir.joinpath(
                        f"Johnson_Eicken_mean_freezeup_date_violin_plots.{output_format}"
                    ),
                )

            if "mdfu-je" in content_kws:
                make_violin_plots(
                    fubu_zs,
                    "mean",
                    {"breakup_start": "Break-up Start", "breakup_end": "Break-up End",},
                    "Set3",
                    "upper left",
                    "Day of year",
                    out_dir.joinpath(
                        f"Johnson_Eicken_mean_breakup_date_violin_plots.{output_format}"
                    ),
                )

        if "steele" in content_group_kws:
            orac_zs = run_zonal_stats(orac, ["DOO", "DOR", "DOA", "DOC"], polys, affine)

            if "isr-steele" in content_kws:
                make_violin_plots(
                    orac_zs,
                    "rate",
                    {"DOO": "Day of Opening", "DOC": "Day of Closing",},
                    "PiYG",
                    "upper right",
                    "Definition rate (%)",
                    out_dir.joinpath(
                        f"Steele_definition_rate_violin_plots.{output_format}"
                    ),
                )

            if "mdfu-steele" in content_kws:
                make_violin_plots(
                    orac_zs,
                    "mean",
                    {"DOA": "Day of Advance", "DOC": "Day of Closing",},
                    "Set3",
                    "upper left",
                    "Day of year",
                    out_dir.joinpath(
                        f"Steele_mean_freezeup_date_violin_plots.{output_format}"
                    ),
                )

            if "mdfu-steele" in content_kws:
                make_violin_plots(
                    orac_zs,
                    "mean",
                    {"DOO": "Day of Opening", "DOR": "Day of Retreat",},
                    "Set3",
                    "upper left",
                    "Day of year",
                    out_dir.joinpath(
                        f"Steele_mean_breakup_date_violin_plots.{output_format}"
                    ),
                )

    # make point locations trends plots
