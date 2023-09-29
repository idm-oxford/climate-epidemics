import pathlib
import xclim.ensembles
import xclim.core.calendar
import xarray as xr

def import_data(**kwargs):
    file_dir_path = str(pathlib.Path(__file__).parent)
    paths = [file_dir_path+"/data/sim"+str(i)+".nc" for i in range(1,101)]
    ds = xclim.ensembles.create_ensemble(paths, multifile=True, **kwargs)
    ds = ds.rename_vars({'TS':'temperature', 'PRECT':'precipitation'})
    return ds

if __name__ == "__main__":
    ds = import_data()
    ds