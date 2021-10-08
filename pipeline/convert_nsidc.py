"""Convert the raw NSIDC-0051 data from flat binary to GeoTIFFs

Usage:
    Functions for step #2 of data pipeline, which creates GeoTIFFs
    of raw NSIDC-0051 data.
"""

import os
import subprocess
import numpy as np
import rasterio as rio


def rescale_values(fp, band=1):
    """
    Rescale the values of a converted GeoTIFF
    """
    with rio.open(fp, "r") as rst:
        arr = rst.read(band).astype(np.float32)
        ind = np.where(arr <= 250)
        arr[ind] = arr[ind] / 250.0
        meta = rst.meta.copy()
        meta.update(compress="lzw", dtype="float32")

    with rio.open(fp, "w", **meta) as out:
        out.write(arr.astype(np.float32), band)

    return fp


def convert_bin_to_gtiff(fp, out_dir):
    """
    Convert binary file to GeoTIFF

    see here: https://nsidc.org/support/how/how-do-i-convert-nsidc-0051-sea-ice-concentration-data-binary-geotiff
    """
    fn = fp.name
    header = [
        u"ENVI",
        u"description = {}".format(fn),
        u"samples = 304",
        u"lines   = 448",
        u"bands   = 1",
        u"header offset = 300",
        u"file type = ENVI Standard",
        u"data type = 1",
        u"interleave = bsq",
        u"byte order = 0",
        u'map info = {Polar Stereographic, 1, 1, -3850000, 5850000, 25000, 25000} projection info = {31, 6378273, 6356889.449, 70, -45, 0, 0, Polar Stereographic} coordinate system string = {PROJCS["Stereographic_North_Pole",GEOGCS["GCS_unnamed ellipse",DATUM["D_unknown",SPHEROID["Unknown",6378273,298.279411123064]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Stereographic_North_Pole"],PARAMETER["standard_parallel_1",70],PARAMETER["central_meridian",-45],PARAMETER["faluse_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]}',
        u"​band names = {Band 1}".strip(u"\u200b"),
    ]

    hdr_fp = str(fp).replace(".bin", ".bin.hdr")
    with open(hdr_fp, "w") as f:
        f.write("\n".join(header))

    out_fp = out_dir.joinpath(fn.replace(".bin", ".tif").replace("v1.1", "v1-1"))

    print(fp)
    print(out_fp)

    command = [
        "gdal_translate",
        "-q",
        "-of",
        "GTiff",
        "-a_srs",
        "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs",
        "-a_nodata",
        "255",
        "-a_ullr",
        "-3850000.0", 
        "5850000.0", 
        "3750000.0", 
        "-5350000.0",
        str(fp),
        str(out_fp),
    ]
    subprocess.call(command)

    # cleanup
    _ = os.remove(hdr_fp)
    print(out_fp)  # for the fun of it!
    _ = rescale_values(out_fp, band=1)

    return out_fp
