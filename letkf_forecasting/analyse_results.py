import os
import glob
import xarray as xr
import numpy as np
import pandas as pd


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
                                 preprocess=add_horizon,
                                 decode_cf=False)
    full_day.horizon.attrs['units'] = 'minutes'
    full_day = xr.decode_cf(full_day)
    return full_day


def add_horizon(ds):
    ds.coords['horizon'] = (ds.time - ds.time[0])/60
    return ds


def return_horizon(df, horizon):
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


def return_ens_var(ds):
    return ds.var(dim='ensemble_number')


def add_crop_attributes(ds):
    ds.attrs['we_er_min'] = 240
    ds.attrs['we_er_max'] = 280
    ds.attrs['sn_er_min'] = 32
    ds.attrs['sn_er_max'] = 88
    return ds


def return_average_error(truth, full_day, horizon):
    rmse = (return_horizon(full_day, horizon) - truth)**2
    rmse = rmse.mean(dim=['south_north', 'west_east'])
    rmse = rmse.to_pandas()
    return rmse


def error_compare(year, month, day, runs):
    truth = xr.open_dataset(
        f'/home2/travis/data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = add_crop_attributes(truth)
    truth = return_error_domain(truth)
    error_dfs = []
    for run in runs:
        full_day = return_day(year, month, day, run)
        full_day = add_crop_attributes(full_day)
        full_day = return_error_domain(full_day)
        full_day = return_ens_mean(full_day)
        fore15 = return_average_error(truth, full_day, 15)
        fore30 = return_average_error(truth, full_day, 30)
        fore45 = return_average_error(truth, full_day, 45)
        fore60 = return_average_error(truth, full_day, 60)
        error_dfs.append(pd.concat([fore15, fore30, fore45, fore60],
                                   axis=1, keys=[15, 30, 45, 60]))
    return error_dfs
