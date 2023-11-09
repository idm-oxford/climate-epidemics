import os
import re
import shutil
import urllib.request
from urllib.parse import urlparse

from siphon.catalog import TDSCatalog


def _get_catalog_name(var_name):
    if var_name == "TS":
        return (
            "https://tds.ucar.edu/thredds/catalog/"
            + "esgcet/459/ucar.cgd.cesm2le.atm.proc.monthly_ave."
            + var_name
            + ".v1.xml"
        )
    elif var_name == "PRECT":
        return (
            "https://tds.ucar.edu/thredds/catalog/"
            + "esgcet/457/ucar.cgd.cesm2le.atm.proc.monthly_ave."
            + var_name
            + ".v1.xml"
        )


def _filter_time_range(siphon_datasets_in, start_year, end_year):
    regex = re.compile(
        r"""(?P<startyear>\d{4})(?P<startmonth>[01]\d)-(?P<endyear>\d{4})
        (?P<endmonth>[01]\d)"""
    )
    siphon_datasets_out = []
    for i in range(len(siphon_datasets_in)):  # pylint: disable=consider-using-enumerate
        dataset = siphon_datasets_in[i]
        match = regex.search(dataset.id)
        if not match:
            raise ValueError("Dataset name does not match expected pattern")
        match_dict = match.groupdict()
        start_year_data = int(match_dict["startyear"])
        end_year_data = int(match_dict["endyear"])
        if start_year_data > end_year or end_year_data < start_year:
            continue
        siphon_datasets_out.append(dataset)
    return siphon_datasets_out


def _get_url(siphon_dataset):
    return (
        "https://tds.ucar.edu/thredds/fileServer"
        + siphon_dataset.access_urls["GRIDFTPatNCAR"]
    )


def _get_file_name(url):
    return os.path.basename(urlparse(url).path)


def _download(opener, url, file_name):
    print(f"Downloading File: {url}")

    try:
        with opener.open(url) as response, open(file_name, "ab") as out_file:
            shutil.copyfileobj(response, out_file)
    except urllib.error.HTTPError as e:
        # Return code error (e.g. 404, 501, ...)
        print(f"HTTPError: {e.code}")
    except urllib.error.URLError as e:
        # Not an HTTP-specific error (e.g. connection refused)
        print(f"URLError: {e.reason}")
    else:
        print("Success")


def _main():
    var_names = ["TS", "PRECT"]
    start_year = 1850
    end_year = 1851

    opener = urllib.request.build_opener()

    for var_name in var_names:
        cat = TDSCatalog(_get_catalog_name(var_name))
        datasets = _filter_time_range(cat.datasets, start_year, end_year)

        urls = [_get_url(dataset) for dataset in datasets]
        file_names = [_get_file_name(url) for url in urls]

        _download(opener, urls[0], file_names[0])


if __name__ == "__main__":
    _main()
