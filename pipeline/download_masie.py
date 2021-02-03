"""Download and transform the MASIE region bounadries from NSIDC

The MASIE region boundaries are saved to BASE_DIR/ancillary
"""

import os, shutil
import urllib.request as request
from contextlib import closing
from pathlib import Path


def main():    
    base_dir = Path(os.getenv("BASE_DIR"))
    out_dir = base_dir.joinpath("ancillary")
    out_dir.mkdir(parents=True, exist_ok=True)

    masie_url = "ftp://sidads.colorado.edu/DATASETS/NOAA/G02186/ancillary/MASIE_regions_polygon_vertices.xls"
    masie_fp = out_dir.joinpath(masie_url.split("/")[-1])

    with closing(request.urlopen(masie_url)) as r:
        with open(masie_fp, "wb") as f:
            shutil.copyfileobj(r, f)

    print(f"MASIE Region boundaries saved to {masie_fp}")


if __name__ == "__main__":
    main()
