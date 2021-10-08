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
