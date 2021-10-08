"""Download data sets 0051 and 0747 from NSIDC

Usage:
    Functions for step #1 of the main data pipeline and for use in 
    manuscript content creation. Use to download NSIDC-0051 and NSIDC-0747
    data sets from NSIDC.

Notes:
    The functions will first search Earthdata for all matching files.
    The user needs to store Earthdata credentials in $HOME/.netrc
    The .netrc file should have the following format:
    machine urs.earthdata.nasa.gov login myusername password mypassword

    These functions were adapted from the NSIDC-provided download script.
    It was originally tested to work with Python 2 as well
    which is why it is probably longer than needed.
"""

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
        username, account, password = info.authenticators(urlparse(urs_url).hostname)

    except Exception:
        try:
            username, account, password = info.authenticators(
                urlparse(cmr_url).hostname
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
    return cmr_file_url + params


def cmr_download(urls, out_dir):
    """Download files from list of urls."""
    if not urls:
        return

    url_count = len(urls)
    print("Downloading {0} files...".format(url_count))
    credentials = None
    
    out_di = {}
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
            out_di[url] = fp
        except HTTPError as e:
            print("HTTP error {0}, {1}".format(e.code, e.reason))
            out_di[url] = e.reason
        except URLError as e:
            print("URL error: {0}".format(e.reason))
            out_di[url] = e.reason
        except IOError:
            raise
        except KeyboardInterrupt:
            quit()

    return out_di


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
            if hits > cmr_page_size:
                print(".", end="")
                sys.stdout.flush()
            urls += url_scroll_results

        if hits > cmr_page_size:
            print()
        return urls
    except KeyboardInterrupt:
        quit()
    