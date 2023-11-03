import pathlib

import xcdat
import xclim.ensembles

import climepi  # noqa


def load_example_data(**kwargs):
    # Import ensemble data via xclim

    file_dir_path = str(pathlib.Path(__file__).parent)
    paths = [file_dir_path + "/data/sim" + str(i) + ".nc" for i in range(1, 101)]

    ds = xclim.ensembles.create_ensemble(paths, multifile=True, **kwargs)

    # Use xcdat to add time bounds and center times

    ds1 = xcdat.open_mfdataset(paths[0], center_times=True)
    ds1["time_bnds"].load()

    for var in ["lon", "lon_bnds", "lat", "lat_bnds", "time", "time_bnds"]:
        ds[var] = ds1[var]
        ds[var].attrs.update(**ds1[var].attrs)

    # Final formatting

    ds = xcdat.swap_lon_axis(ds, to=(-180, 180))
    ds = ds.rename_vars({"TS": "temperature", "PRECT": "precipitation"})

    ds.temperature.attrs.update(long_name="Temperature")
    ds.temperature.attrs.update(units="°C")
    ds.precipitation.attrs.update(long_name="Precipitation")

    ds.climepi.modes = {
        "spatial": "global",
        "temporal": "monthly",
        "ensemble": "ensemble",
    }

    return ds
