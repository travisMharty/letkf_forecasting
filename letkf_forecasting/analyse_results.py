import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import letkf_forecasting.letkf_io as io


def return_horizon(df, horizon):
    if type(horizon) != np.timedelta64:
        horizon = np.timedelta64(int(horizon), 'm')
    to_return = df.where(df.horizon == horizon, drop=True)
    to_return = to_return.drop('horizon')
    return to_return


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


def return_rmse(truth, full_day):
    rmse = (full_day - truth)**2
    rmse = np.sqrt(rmse.mean(dim=['south_north', 'west_east']))
    rmse = rmse.to_pandas()
    return rmse


def return_rmse_one_day(truth, full_day):
    rmse_df = pd.DataFrame(columns=['rmse'])
    for horizon in [15, 30, 45, 60]:
        rmse = (return_horizon(full_day, horizon) - truth)**2
        rmse = np.sqrt(rmse.mean(
            dim=['south_north', 'west_east', 'time']).item())
        rmse_df.loc[horizon] = rmse
    return rmse_df


def return_bias(truth, full_day):
    bias = truth - full_day
    bias = bias.mean(dim=['south_north', 'west_east'])
    bias = bias.to_pandas()
    return bias


def return_bias_one_day(truth, full_day):
    bias_df = pd.DataFrame(columns=['bias'])
    for horizon in [15, 30, 45, 60]:
        bias = return_horizon(full_day, horizon)
        bias = bias - truth
        bias = bias.mean(dim=['south_north', 'west_east', 'time'])
        bias = bias.item()
        bias_df.loc[horizon] = bias
    return bias_df


def return_correlation(truth, full_day):
    fore_df = full_day
    fore_df = fore_df.to_dataframe(name='ci')
    fore_df = fore_df.unstack(0)
    truth_df = truth.to_dataframe(name='ci')
    truth_df = truth_df.unstack(0)
    R = fore_df.corrwith(truth_df)
    R = R['ci']
    return R


def return_correlation_one_day(truth, full_day):
    R_df = pd.DataFrame(columns=['correlation'])
    truth_df = truth.to_dataframe(name='ci')
    # print(truth_df)
    # print(type(truth_df))
    for horizon in [15, 30, 45, 60]:
        fore_df = return_horizon(full_day, horizon)
        fore_df = fore_df.to_dataframe(name='ci')
        # print(fore_df)
        # print(type(truth_df))
        R = fore_df.corrwith(truth_df['ci']).item()
        R_df.loc[horizon] = R
    return R_df


def return_sd(truth, full_day):
    sd = full_day
    sd = sd.var(dim=['south_north', 'west_east'])
    sd = np.sqrt(sd)
    sd = sd.to_pandas()
    return sd


def return_sd_one_day(truth, full_day):
    sd_df = pd.DataFrame(columns=['sd'])
    sd_truth_df = pd.DataFrame(columns=['true_sd'])
    for horizon in [15, 30, 45, 60]:
        sd = return_horizon(full_day, horizon)
        times = np.intersect1d(
            truth.time.to_pandas(),
            sd.time.to_pandas())
        sd = sd.sel(time=times)
        sd_truth = truth.sel(time=times)
        sd = sd.var(dim=['south_north', 'west_east', 'time'])
        sd = np.sqrt(sd).item()
        sd_df.loc[horizon] = sd
        sd_truth = sd_truth.var(dim=['south_north', 'west_east', 'time'])
        sd_truth = np.sqrt(sd_truth).item()
        sd_truth_df.loc[horizon] = sd_truth
    return sd_df, sd_truth_df


def return_spread(truth, da):
    spread = return_ens_var(da)
    if 'west_east_stag' in da.dims:
        spread = spread.mean(dim=['south_north', 'west_east_stag'])
    elif 'south_north_stag' in da.dims:
        spread = spread.mean(dim=['south_north_stag', 'west_east'])
    else:
        spread = spread.mean(dim=['south_north', 'west_east'])
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
        full_day = io.return_day(year, month, day, run)
        full_day = add_crop_attributes(full_day)
        full_day = full_day['ci']
        full_day = return_error_domain(full_day)
        full_day = return_ens_mean(full_day)
        fore15 = return_rmse(truth, full_day, 15)
        fore30 = return_rmse(truth, full_day, 30)
        fore45 = return_rmse(truth, full_day, 45)
        fore60 = return_rmse(truth, full_day, 60)
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
        full_day = io.return_day(year, month, day, run)
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
        fore15 = return_rmse(truth, full_day, 15)
        fore30 = return_rmse(truth, full_day, 30)
        fore45 = return_rmse(truth, full_day, 45)
        fore60 = return_rmse(truth, full_day, 60)
        error_dfs.append(pd.concat([fore15, fore30, fore45, fore60],
                                   axis=1, keys=[15, 30, 45, 60]))
    return error_dfs, spread_ci, spread_wind


def return_stat_df(truth, full_day, stat_function):
    horizons = full_day.horizon.to_pandas().unique()
    horizons = (horizons/(60*1e9)).astype(np.int16)
    stats = []
    for horizon in horizons:
        this_stat = stat_function(
            truth,
            return_horizon(full_day, horizon))
        stats.append(this_stat)
    stats = pd.concat(stats, axis=1, keys=horizons)
    return stats


def error_stats(year, month, day, runs):
    truth = xr.open_dataset(
        f'/home2/travis/data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = truth['ci']
    truth = add_crop_attributes(truth)
    truth = return_error_domain(truth)
    to_return = []
    truth_sd = np.sqrt(truth.var(dim=['south_north', 'west_east']))
    truth_sd = truth_sd.to_pandas()
    for run in runs:
        print(run)
        adict = {'name': run, 'truth_sd': truth_sd}
        full_day = io.return_day(year, month, day, run)
        full_day = add_crop_attributes(full_day)
        full_day = return_error_domain(full_day)
        adict['u_spread'] = return_stat_df(
            truth, full_day['U'], return_spread)
        adict['v_spread'] = return_stat_df(
            truth, full_day['V'], return_spread)
        adict['spread_ci'] = return_stat_df(
            truth, full_day['ci'], return_spread)

        full_day = full_day['ci']
        full_day = return_ens_mean(full_day)
        adict['rmse'] = return_stat_df(
            truth, full_day, return_rmse)
        adict['forecast_sd'] = return_stat_df(
            truth, full_day, return_sd)
        adict['bias'] = return_stat_df(
            truth, full_day, return_bias)
        adict['correlation'] = return_stat_df(
            truth, full_day, return_correlation)

        to_return.append(adict)
    return to_return


def error_stats_one_day(year, month, day, runs):
    truth = xr.open_dataset(
        f'/home2/travis/data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = truth['ci']
    truth = add_crop_attributes(truth)
    truth = return_error_domain(truth)
    to_return = []
    # truth_sd = np.sqrt(truth.var()).item()
    for run in runs:
        print(run)
        # adict = {'name': run, 'truth_sd': truth_sd}
        adict = {'name': run}
        if run == 'persistence':
            adict = return_persistence_dict_one_day(
                adict, truth, [15, 30, 45, 60])
            to_return.append(adict)
            continue
        full_day = io.return_day(year, month, day, run)
        full_day = add_crop_attributes(full_day)
        full_day = return_error_domain(full_day)
        full_day = full_day['ci']
        full_day = return_ens_mean(full_day)

        rmse = return_rmse_one_day(truth, full_day)
        forecast_sd, truth_sd = return_sd_one_day(truth, full_day)
        bias = return_bias_one_day(truth, full_day)
        corr = return_correlation_one_day(truth, full_day)

        adict['rmse'] = rmse
        adict['forecast_sd'] = forecast_sd
        adict['truth_sd'] = truth_sd
        adict['bias'] = bias
        adict['correlation'] = corr

        to_return.append(adict)
    return to_return


def return_persistence_dict_one_day(adict, truth, horizons):
    rmse_df = pd.DataFrame(columns=['rmse'])
    sd_df = pd.DataFrame(columns=['sd'])
    sd_truth_df = pd.DataFrame(columns=['true_sd'])
    bias_df = pd.DataFrame(columns=['bias'])
    corr_df = pd.DataFrame(columns=['correlation'])
    for horizon in horizons:
        forecast = truth.copy()
        forecast['time'] = forecast.time + pd.Timedelta(horizon, 'm')

        rmse = (forecast - truth)**2
        rmse = rmse.mean(
            dim=['south_north', 'west_east', 'time']).item()
        rmse = np.sqrt(rmse)
        rmse_df.loc[horizon] = rmse

        sd_times = np.intersect1d(
            truth.time.to_pandas(),
            forecast.time.to_pandas())
        sd = forecast.sel(time=sd_times).var(
            dim=['south_north', 'west_east', 'time']).item()
        sd = np.sqrt(sd)
        sd_df.loc[horizon] = sd

        sd_truth = truth.sel(time=sd_times).var(
            dim=['south_north', 'west_east', 'time']).item()
        sd_truth = np.sqrt(sd_truth)
        sd_truth_df.loc[horizon] = sd_truth

        t_mean = truth.mean(
            dim=['south_north', 'west_east', 'time']).item()
        f_mean = forecast.mean(
            dim=['south_north', 'west_east', 'time']).item()
        bias = f_mean - t_mean
        bias_df.loc[horizon] = bias

        truth_df = truth.to_dataframe(name='ci')
        fore_df = forecast.to_dataframe(name='ci')
        corr = fore_df.corrwith(truth_df['ci']).item()
        corr_df.loc[horizon] = corr
    adict['rmse'] = rmse_df
    adict['forecast_sd'] = sd_df
    adict['bias'] = bias_df
    adict['correlation'] = corr_df
    return adict


def generate_plots(year, month, day, run_name):
    truth = xr.open_dataset(
        f'/home2/travis/data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = truth['ci']
    truth = add_crop_attributes(truth)
    plot_folder = io.return_results_folder(year, month, day, run_name)
    plot_folder = os.join(plot_folder, 'plots/')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    full_day = io.return_day(year, month, day, run_name)
    fore15 = return_horizon(full_day, 15)
    fore30 = return_horizon(full_day, 30)
    fore45 = return_horizon(full_day, 45)
    fore60 = return_horizon(full_day, 60)
    for forecast in [fore15, fore30, fore45, fore60]:
        for time in forecast.time:
            return None
