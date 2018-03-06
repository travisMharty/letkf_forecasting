import os
import xarray as xr
import glob


def return_day(year, month, day, run_name):
    path = os.path.expanduser('~')
    path = os.path.join(
        path,
        f'results/{year:04}/{month:02}/{day:02}/' + run_name)
    paths = glob.glob(path + '*')
    paths.sort()
    path = paths[-1]
    path = os.path.join(path, '*.nc')
    full_day = xr.open_mfdataset(path, concat_dim='init',
                                 preprocess=add_init,
                                 decode_cf=False)
    full_day.init.attrs['units'] = full_day.time.units
    full_day = xr.decode_cf(full_day)
    return full_day


def add_init(ds):
    ds.coords['init'] = ds.time.init
    return ds
