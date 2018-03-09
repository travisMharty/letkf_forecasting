import os
import xarray as xr
import glob
import numpy as np


def return_day(year, month, day, run_name):
    path = os.path.expanduser('~')
    path = os.path.join(
        path,
        f'results/{year:04}/{month:02}/{day:02}/' + run_name)
    paths = glob.glob(path + '*')
    paths.sort()
    path = paths[-1]
    path = os.path.join(path, '*.nc')
    full_day = xr.open_mfdataset(path,
                                 preprocess=add_init,
                                 decode_cf=False)
    full_day.horizon.attrs['units'] = 'minutes'
    full_day = xr.decode_cf(full_day)
    return full_day


def add_init(ds):
    ds.coords['horizon'] = (ds.time - ds.time[0])/60
    return ds


def get_horizon(df, horizon):
    if type(horizon) != np.timedelta64:
        horizon = np.timedelta64(horizon, 'm')
    to_return = df.where(df.horizon == horizon, drop=True)
    to_return = to_return.drop('horizon')
    return to_return


def return_error_domain(ds):
    we_er_min = ds.we_er_min
    we_er_max = ds.we_er_max
    sn_er_min = ds.sn_er_min
    sn_er_max = ds.sn_er_max
    to_return = ds['ci'].sel(
        west_east=slice(we_er_min, we_er_max),
        south_north=slice(sn_er_min, sn_er_max))
    return to_return


def return_ens_mean(ds):
    return ds.mean(dim='ensemble_number', keep_attrs=True)
