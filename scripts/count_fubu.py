"""Create maps of pixel-wise counts of years where metric defined"""

import os, copy
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import xarray as xr


def plot_counts(fubu_ds, metric, samp_arr, plot_dir, meta):
    """Count valid dates by metric and save map plot + GeoTIFF"""
    metric_arr = fubu_ds[metric].values
    valid = np.isnan(metric_arr) == False
    counts = np.sum(valid, axis=0)
    counts[samp_arr == 254.0] = -9999
    plot_arr = np.ma.masked_where(samp_arr == 254.0, counts)
    plot_arr = plot_arr.astype("int32")
    plot_fp = os.path.join(plot_dir, f"valid_{metric}_counts.png")
    cmap = copy.copy(plt.cm.get_cmap("viridis"))
    cmap.set_bad(color="black")
    plt.imshow(plot_arr, interpolation="none", cmap=cmap)
    plt.title("Years where {} defined".format(metric.replace("_", " ")))
    plt.colorbar()
    plt.axis("off")
    plt.savefig(plot_fp)
    plt.close()
    gtiff_fp = os.path.join(
        plot_dir.replace("png", "geotiff"), f"valid_{metric}_counts.tif"
    )
    with rio.open(gtiff_fp, "w", **meta) as out:
        out.write(plot_arr, 1)
    print(f"{metric} counts map plot saved to {plot_fp}")
    print(f"{metric} counts GeoTIFF saved to {gtiff_fp}")


if __name__ == "__main__":
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    fubu_fn = "nsidc_0051_sic_nasateam_1979-2018_north_smoothed_fubu_dates.nc"
    # computed fubu dates fp
    fubu_fp = os.path.join(base_dir, "outputs", "NetCDF", fubu_fn)
    # sample fp for mask
    samp_fp = os.path.join(scratch_dir, "nsidc_sample_20181231.tif")
    # load fubu dates netCDF
    with xr.open_dataset(fubu_fp) as ds:
        fubu_ds = ds.load().copy()
    # set up meta data for saving GeoTIFFs
    with rio.open(samp_fp) as src:
        samp_arr = src.read(1)
        meta = src.meta
    meta.update(dtype="int32", compress="lzw", count=1)
    plot_dir = os.path.join(scratch_dir, "fubu_counts", "png")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(plot_dir.replace("png", "geotiff")):
        os.makedirs(plot_dir.replace("png", "geotiff"))
    for metric in fubu_ds:
        plot_counts(fubu_ds, metric, samp_arr, plot_dir, meta)
