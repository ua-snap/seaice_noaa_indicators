"""Functions for running the points-of-interest analysis"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer
from rasterio.plot import show
from shapely.geometry import Polygon

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
    transformer = Transformer.from_crs(in_epsg, out_epsg)

    return transformer.transform(*poi)


def get_offsets_xy(src, xy, offsets):
    """
    returns list of center points of pixels corresponding
    to the offsets
    """
    row, col = src.index(*xy)

    return [src.xy(row + offset[0], col + offset[1]) for offset in offsets]


def get_xy_lims(x, y, scale):
    """Get the x, y limits for a point given the scale specified in the dict"""
    xlims = (x - scale * 1e5, x + scale * 1e5)
    ylims = (y - scale * 1e5, y + scale * 1e5)

    return xlims, ylims


def clip_shore_to_viewing_extent(world_shore, xlims, ylims):
    """Clip the shore polygon to the extent derived from the xy lims"""
    corner_list = [
        (xlims[0], ylims[0]),
        (xlims[0], ylims[1]),
        (xlims[1], ylims[1]),
        (xlims[1], ylims[0]),
        (xlims[0], ylims[0]),
    ]
    bb = Polygon([corner for corner in corner_list])
    bb_df = gpd.GeoDataFrame(geometry=[bb]).set_crs(3411)
    
    return gpd.overlay(bb_df, world_shore, how="intersection")
    

def make_pixel_polygon_from_xy(transform, x, y):
    """Make polygon corresponding to raster pixels
    from given xy centerpoint and the raster's Affine transform
    """
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
    polys = [make_pixel_polygon_from_xy(transform, *xy) for xy in xy_list]

    return gpd.GeoDataFrame(geometry=polys)


def plot_poi_pixel_polys(poi, offsets, scale, landmask_src, world_shore, coast_src=None):
    """Plot the polygons depicting the pixels to be used for the points of interest

    Args:
        poi (tuple): lat/lon coordinates in form (lat, lon)
        offsets (list): list of 2-tuples with offset values for pixels to inclue in analysis
        scale (float): scalar multiplier to determine viewing window
        landmask_src (rasterio.io.DatasetReader): Open GeoTIFF of landmask for basemap
        coast_src (rasterio.io.DatasetReader): Open GeoTIFF of coastline mask to show coastal pixels
        world_shore (GeoPandas.GeoDataFrame): polyons for shoreline boundaries
        

    Returns:
        list of coordinates of pixels corresponding to offsets
    """
    # coordinates of point of interest
    poi_xy = reproject_poi(poi)
    # list of coordinates of pixels corresponding to offsets from 
    #   pixel overlapping POI
    xy_list = get_offsets_xy(landmask_src, poi_xy, offsets)
    # get x and y limits for viewing window based on 
    #   scale and poi_xy
    xlims = (poi_xy[0] - scale * 1e5, poi_xy[0] + scale * 1e5)
    ylims = (poi_xy[1] - scale * 1e5, poi_xy[1] + scale * 1e5)
    
    # get polygon of shoreline within viewing window
    shore_poly = clip_shore_to_viewing_extent(world_shore, xlims, ylims)
    
    # make pixel polygons
    pixel_polys = make_pixel_poly_gdf(landmask_src.transform, xy_list)

    # create plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # plot basemap
    show(landmask_src, ax=ax, vmax=10000, cmap="twilight")
    # plot coastline pixels
    if coast_src is not None:
        show(coast_src, ax=ax, interpolation="none")
    # plot shorelines
    shore_poly.plot(ax=ax, facecolor="none", edgecolor="gray")
    # plot cells that will be used
    pixel_polys.plot(ax=ax, facecolor="none", edgecolor="red")
    # show the point of interest
    ax.scatter(*poi_xy, color="red")
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

    return xy_list


def plot_date_histograms(fubu, xy_list, poi_name):
    """Summarize the counts of years for which the indicators were defined
    in a bar chart
    """
    # initialize index for DF
    index = []
    
    # make names for each of the offset pixels
    offset_names = [f"p{i}" for i in range(len(xy_list))]
    # initialize dict for DF
    df_di = {name: [] for name in offset_names}
    df_di["indicator"] = []
    
    for indicator in list(fubu.variables)[:4]:
        for xy, name in zip(xy_list, offset_names):
            dates = fubu[indicator].sel(xc=xy[0], yc=xy[1], method="nearest").values.astype(np.float32)
            dates[dates == -9999] = np.nan
            df_di[name].extend(dates)
        df_di["indicator"].extend(np.repeat(indicator, len(dates)))
        
    df = pd.DataFrame(df_di)
    df.hist(by="indicator", figsize=(15, 4), layout=(1, 4))
    plt.tight_layout()
    
    # compute mean indicator dates for point extraction
    df["mean_date"] = np.round(df.drop(columns="indicator").mean(axis=1), 1)
    df["location"] = poi_name
    df["year"] = np.tile(fubu.year.values, 4)
    
    return df[["year", "location", "indicator", "mean_date"]]