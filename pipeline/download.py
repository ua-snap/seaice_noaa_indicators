"""Download data sets from NSIDC

Usage:
    pipenv run python download.py [-c]
    Script #1 of data pipeline

Returns:
    NSIDC-0051 daily data files written to $BASE_DIR/nsidc_0051/raw/daily
    NSIDC-0747 data file written $BASE_DIR/nsidc_0747 

Notes:
    The script will first search Earthdata for all matching files.
    The user needs to store Earthdata credentials in $HOME/.netrc
    The .netrc file should have the following format:
    machine urs.earthdata.nasa.gov login myusername password mypassword

    This script was adapted from the nsidc-download script.
    It was originally tested to work with Python 2 as well
    which is why it is longer than needed, plan to re-implement in
    python 3 exclusively if time allows.
"""

import argparse
import base64
import itertools
import json
import netrc
import os
import ssl
import sys
from getpass import getpass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor

version = "1"
time_start = "1978-10-26T00:00:00Z"
time_end = "2019-12-31T23:59:59Z"
polygon = ""
filename_filter = "*"

CMR_URL = "https://cmr.earthdata.nasa.gov"
URS_URL = "https://urs.earthdata.nasa.gov"
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = (
    "{0}/search/granules.json?provider=NSIDC_ECS"
    "&sort_key[]=start_date&sort_key[]=producer_granule_id"
    "&scroll=true&page_size={1}".format(CMR_URL, CMR_PAGE_SIZE)
)


def get_username():
    username = ""

    # For Python 2/3 compatibility:
    try:
        do_input = raw_input  # noqa
    except NameError:
        do_input = input

    while not username:
        try:
            username = do_input("Earthdata username: ")
        except KeyboardInterrupt:
            quit()
    return username


def get_password():
    password = ""
    while not password:
        try:
            password = getpass("password: ")
        except KeyboardInterrupt:
            quit()
    return password


def get_credentials(url):
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(urlparse(URS_URL).hostname)

    except Exception:
        try:
            username, account, password = info.authenticators(
                urlparse(CMR_URL).hostname
            )
        except Exception:
            username = None
            password = None

    while not credentials:
        if not username:
            username = get_username()
            password = get_password()
        credentials = "{0}:{1}".format(username, password)
        credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")

        if url:
            try:
                req = Request(url)
                req.add_header("Authorization", "Basic {0}".format(credentials))
                opener = build_opener(HTTPCookieProcessor())
                opener.open(req)
            except HTTPError:
                print("Incorrect username or password")
                credentials = None
                username = None
                password = None

    return credentials


def build_version_query_params(version):
    desired_pad_length = 3
    if len(version) > desired_pad_length:
        print('Version string too long: "{0}"'.format(version))
        quit()

    version = str(int(version))  # Strip off any leading zeros
    query_params = ""

    while len(version) <= desired_pad_length:
        padded_version = version.zfill(desired_pad_length)
        query_params += "&version={0}".format(padded_version)
        desired_pad_length -= 1
    return query_params


def build_cmr_query_url(
    short_name, version, time_start, time_end, polygon=None, filename_filter=None
):
    params = "&short_name={0}".format(short_name)
    params += build_version_query_params(version)
    params += "&temporal[]={0},{1}".format(time_start, time_end)
    if polygon:
        params += "&polygon={0}".format(polygon)
    if filename_filter:
        params += "&producer_granule_id[]={0}&options[producer_granule_id][pattern]=true".format(
            filename_filter
        )
    return CMR_FILE_URL + params


def cmr_download(urls, out_dir):
    """Download files from list of urls."""
    if not urls:
        return

    url_count = len(urls)
    print("Downloading {0} files...".format(url_count))
    credentials = None

    for index, url in enumerate(urls, start=1):
        if not credentials and urlparse(url).scheme == "https":
            credentials = get_credentials(url)

        fp = out_dir.joinpath(Path(url).name)
        print(
            "{0}/{1}: {2}".format(
                str(index).zfill(len(str(url_count))), url_count, fp
            )
        )

        try:
            # In Python 3 we could eliminate the opener and just do 2 lines:
            # resp = requests.get(url, auth=(username, password))
            # open(fp, 'wb').write(resp.content)
            print(url)
            req = Request(url)
            if credentials:
                req.add_header("Authorization", "Basic {0}".format(credentials))
            opener = build_opener(HTTPCookieProcessor())
            data = opener.open(req).read()
            open(fp, "wb").write(data)
        except HTTPError as e:
            print("HTTP error {0}, {1}".format(e.code, e.reason))
        except URLError as e:
            print("URL error: {0}".format(e.reason))
        except IOError:
            raise
        except KeyboardInterrupt:
            quit()

    return fp


def cmr_filter_urls(search_results):
    """Select only the desired data files from CMR response."""
    if "feed" not in search_results or "entry" not in search_results["feed"]:
        return []

    entries = [e["links"] for e in search_results["feed"]["entry"] if "links" in e]
    # Flatten "entries" to a simple list of links
    links = list(itertools.chain(*entries))

    urls = []
    unique_filenames = set()
    for link in links:
        if "href" not in link:
            # Exclude links with nothing to download
            continue
        if "inherited" in link and link["inherited"] is True:
            # Why are we excluding these links?
            continue
        if "rel" in link and "data#" not in link["rel"]:
            # Exclude links which are not classified by CMR as "data" or "metadata"
            continue

        if "title" in link and "opendap" in link["title"].lower():
            # Exclude OPeNDAP links--they are responsible for many duplicates
            # This is a hack; when the metadata is updated to properly identify
            # non-datapool links, we should be able to do this in a non-hack way
            continue

        filename = link["href"].split("/")[-1]
        if filename in unique_filenames:
            # Exclude links with duplicate filenames (they would overwrite)
            continue
        unique_filenames.add(filename)

        urls.append(link["href"])

    return urls


def cmr_search(
    short_name, version, time_start, time_end, polygon="", filename_filter=""
):
    """Perform a scrolling CMR query for files matching input criteria."""
    cmr_query_url = build_cmr_query_url(
        short_name=short_name,
        version=version,
        time_start=time_start,
        time_end=time_end,
        polygon=polygon,
        filename_filter=filename_filter,
    )
    print("Querying for data:\n\t{0}\n".format(cmr_query_url))

    cmr_scroll_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        urls = []
        while True:
            req = Request(cmr_query_url)
            if cmr_scroll_id:
                req.add_header("cmr-scroll-id", cmr_scroll_id)
            response = urlopen(req, context=ctx)
            if not cmr_scroll_id:
                # Python 2 and 3 have different case for the http headers
                headers = {k.lower(): v for k, v in dict(response.info()).items()}
                cmr_scroll_id = headers["cmr-scroll-id"]
                hits = int(headers["cmr-hits"])
                if hits > 0:
                    print("Found {0} matches.".format(hits))
                else:
                    print("Found no matches.")
            search_page = response.read()
            search_page = json.loads(search_page.decode("utf-8"))
            url_scroll_results = cmr_filter_urls(search_page)
            if not url_scroll_results:
                break
            if hits > CMR_PAGE_SIZE:
                print(".", end="")
                sys.stdout.flush()
            urls += url_scroll_results

        if hits > CMR_PAGE_SIZE:
            print()
        return urls
    except KeyboardInterrupt:
        quit()
    

    
if __name__ == "__main__":
    # download NSIDC-0051 first, NSIDC-0747 second
    # check what is already present locally before downloading
    
    # Supply some default search parameters, just for testing purposes.
    # These are only used if the parameters aren't filled in up above.
    global version, time_start, time_end, polygon, filename_filter

    parser = argparse.ArgumentParser(description="Download NSIDC data")
    parser.add_argument(
        "-c",
        "--clobber",
        action="store_true",
        dest="clobber",
        default=False,
        help=("Flag to download all data, overwriting any local data"),
    )
    
    # unpack args
    args = parser.parse_args()
    clobber = args.clobber
    
    base_dir = Path(os.getenv("BASE_DIR"))
    out_0051_dir = base_dir.joinpath("nsidc_0051/raw/daily")
    out_0051_dir.mkdir(exist_ok=True, parents=True)
    
    urls = cmr_search(
        "NSIDC-0051",
        version,
        time_start,
        time_end,
        polygon=polygon,
        filename_filter=filename_filter,
    )
    
    # dunno what these new files are yet, but remove them
    urls = [url for url in urls if "s.bin" not in Path(url).name]
    # also filter out monthly data urls
    urls = [url for url in urls if len(Path(url).name.split("_")[1]) == 8]  

    if not clobber:
        # filter out already downloaded files
        local_fps = [fp.name for fp in list(out_0051_dir.glob("*"))]
        urls = [url for url in urls if Path(url).name not in local_fps]

    _ = cmr_download(urls, out_0051_dir)
    print(f"NSIDC-0051 data written to {out_0051_dir}")

    # next, download NSIDC-0747
    out_0747_dir = base_dir.joinpath("nsidc_0747")
    out_0747_dir.mkdir(exist_ok=True)
    url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0747_seaice_melt_indicators_v1/arctic_seaice_climate_indicators_nh_v01r01_1979-2017.nc"
    if not clobber:
        if not out_0747_dir.joinpath(Path(url).name).exists():
            out_0747_fp = cmr_download([url], out_0747_dir)
            print(f"NSIDC-0747 written to {out_0747_fp}")
        else:
            print("NSIDC-0747 already preent in $BASE_DIR")
    else:
        out_0747_fp = cmr_download([url], out_0747_dir)
        print(f"NSIDC-0747 written to {out_0747_fp}")
