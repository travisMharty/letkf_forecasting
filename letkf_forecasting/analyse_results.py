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


# should't assume ci
def return_error_domain(ds):
    we_er_min = ds.we_er_min
    we_er_max = ds.we_er_max
    sn_er_min = ds.sn_er_min
    sn_er_max = ds.sn_er_max
    we_slice = slice(we_er_min, we_er_max)
    sn_slice = slice(sn_er_min, sn_er_max)
    test = ('west_east_stag' in ds.coords.variables and
            'south_north_stag' in ds.coords.variables)
    if test:
        to_return = ds.sel(
            west_east=we_slice,
            south_north=sn_slice,
            west_east_stag=we_slice,
            south_north_stag=sn_slice)
    else:
        to_return = ds.sel(
            west_east=we_slice,
            south_north=sn_slice)
    return to_return


def return_ens_mean(ds):
    return ds.mean(dim='ensemble_number', keep_attrs=True)


def return_ens_var(ds):
    return ds.var(dim='ensemble_number', keep_attrs=True)


def add_crop_attributes(ds):
    ds.attrs['we_er_min'] = 240
    ds.attrs['we_er_max'] = 280
    ds.attrs['sn_er_min'] = 32
    ds.attrs['sn_er_max'] = 88
    return ds


def return_average_error(truth, full_day, horizon):
    rmse = (return_horizon(full_day, horizon) - truth)**2
    rmse = np.sqrt(rmse.mean(dim=['south_north', 'west_east']))
    rmse = rmse.to_pandas()
    return rmse


def return_sd(full_day, horizon):
    sd = return_horizon(full_day, horizon)
    sd = sd.var(dim=['south_north', 'west_east'])
    sd = np.sqrt(sd)
    sd = sd.to_pandas()
    return sd


def return_spread(da, horizon):
    spread = return_horizon(da, horizon)
    spread = return_ens_var(spread)
    if da.name == 'ci':
        spread = spread.mean(dim=['south_north', 'west_east'])
    elif da.name == 'U':
        spread = spread.mean(dim=['south_north', 'west_east_stag'])
    elif da.name == 'V':
        spread = spread.mean(dim=['south_north_stag', 'west_east'])
    spread = np.sqrt(spread)
    spread = spread.to_pandas()
    return spread


def error_compare(year, month, day, runs):
    truth = xr.open_dataset(
        f'/home2/travis/data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = truth['ci']
    truth = add_crop_attributes(truth)
    truth = return_error_domain(truth)
    error_dfs = []
    for run in runs:
        full_day = return_day(year, month, day, run)
        full_day = add_crop_attributes(full_day)
        full_day = full_day['ci']
        full_day = return_error_domain(full_day)
        full_day = return_ens_mean(full_day)
        fore15 = return_average_error(truth, full_day, 15)
        fore30 = return_average_error(truth, full_day, 30)
        fore45 = return_average_error(truth, full_day, 45)
        fore60 = return_average_error(truth, full_day, 60)
        error_dfs.append(pd.concat([fore15, fore30, fore45, fore60],
                                   axis=1, keys=[15, 30, 45, 60]))
    return error_dfs


def error_spread_compare(year, month, day, runs):
    truth = xr.open_dataset(
        f'/home2/travis/data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = truth['ci']
    truth = add_crop_attributes(truth)
    truth = return_error_domain(truth)
    error_dfs = []
    spread_wind = []
    spread_ci = []
    for run in runs:
        full_day = return_day(year, month, day, run)
        full_day = add_crop_attributes(full_day)
        full_day = return_error_domain(full_day)
        u_spread = return_spread(full_day['U'], 0)
        v_spread = return_spread(full_day['V'], 0)
        spread_wind.append(pd.concat([u_spread, v_spread],
                                     axis=1, keys=['U', 'V']))
        ci_spread_15 = return_spread(full_day['ci'], 15)
        ci_spread_30 = return_spread(full_day['ci'], 30)
        ci_spread_45 = return_spread(full_day['ci'], 45)
        ci_spread_60 = return_spread(full_day['ci'], 60)
        spread_ci.append(pd.concat([ci_spread_15, ci_spread_30,
                                    ci_spread_45, ci_spread_60],
                                   axis=1, keys=[15, 30, 45, 60]))
        full_day = full_day['ci']
        full_day = return_ens_mean(full_day)
        fore15 = return_average_error(truth, full_day, 15)
        fore30 = return_average_error(truth, full_day, 30)
        fore45 = return_average_error(truth, full_day, 45)
        fore60 = return_average_error(truth, full_day, 60)
        error_dfs.append(pd.concat([fore15, fore30, fore45, fore60],
                                   axis=1, keys=[15, 30, 45, 60]))
    return error_dfs, spread_ci, spread_wind


def error_stats(year, month, day, runs):
    truth = xr.open_dataset(
        f'/home2/travis/data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = truth['ci']
    truth = add_crop_attributes(truth)
    truth = return_error_domain(truth)
    error_dfs = []
    mean_sd_dfs = []
    spread_wind = []
    spread_ci = []
    for run in runs:
        print(run)
        full_day = return_day(year, month, day, run)
        full_day = add_crop_attributes(full_day)
        full_day = return_error_domain(full_day)
        u_spread = return_spread(full_day['U'], 0)
        v_spread = return_spread(full_day['V'], 0)
        spread_wind.append(pd.concat([u_spread, v_spread],
                                     axis=1, keys=['U', 'V']))
        ci_spread_15 = return_spread(full_day['ci'], 15)
        ci_spread_30 = return_spread(full_day['ci'], 30)
        ci_spread_45 = return_spread(full_day['ci'], 45)
        ci_spread_60 = return_spread(full_day['ci'], 60)
        spread_ci.append(pd.concat([ci_spread_15, ci_spread_30,
                                    ci_spread_45, ci_spread_60],
                                   axis=1, keys=[15, 30, 45, 60]))
        full_day = full_day['ci']
        full_day = return_ens_mean(full_day)
        fore15 = return_average_error(truth, full_day, 15)
        fore30 = return_average_error(truth, full_day, 30)
        fore45 = return_average_error(truth, full_day, 45)
        fore60 = return_average_error(truth, full_day, 60)
        error_dfs.append(pd.concat([fore15, fore30, fore45, fore60],
                                   axis=1, keys=[15, 30, 45, 60]))
        fore15_sd = return_sd(full_day, 15)
        fore30_sd = return_sd(full_day, 30)
        fore45_sd = return_sd(full_day, 45)
        fore60_sd = return_sd(full_day, 60)
        mean_sd_dfs.append(pd.concat([fore15_sd, fore30_sd,
                                      fore45_sd, fore60_sd],
                                     axis=1, keys=[15, 30, 45, 60]))
    return error_dfs, spread_ci, mean_sd_dfs, spread_wind
