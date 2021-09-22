"""Download other ancillary data needed for analyses:
1. the MASIE region bounadries from NSIDC
2. world shoreline polygons from NCEI

Returns:
    above shapefiles are written to $BASE_DIR/ancillary
"""

import os, shutil
import urllib.request as request
from contextlib import closing
from pathlib import Path


def run_download(url, out_fp):
    """Run the download using urllib
    
    Args:
        url (string): URL of data file to be downloaded
        out_dir (pathlib.PosixPath): directory to download the target file to

    Returns:
        None, file at url is downloaded to out_dir
    """
    with closing(request.urlopen(url)) as r:
        with open(out_fp, "wb") as f:
            shutil.copyfileobj(r, f)
        
    return


# if __name__ == "__main__":
#     base_dir = Path(os.getenv("BASE_DIR"))
#     out_dir = base_dir.joinpath("ancillary")
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # download MASIE region vertices
#     masie_url = "ftp://sidads.colorado.edu/DATASETS/NOAA/G02186/ancillary/MASIE_regions_polygon_vertices.xls"
#     masie_fp = out_dir.joinpath(masie_url.split("/")[-1])
#     run_download(masie_url, masie_fp)
#     print(f"MASIE Region vertices saved to {masie_fp}")
    
#     # download world shoreline polygons
#     shore_url = "https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/gshhg-shp-2.3.7.zip"
#     shore_fp = out_dir.joinpath(shore_url.split("/")[-1])
#     run_download(shore_url, shore_fp)

#     print(f"World shoreline  saved to {shore_fp}")
