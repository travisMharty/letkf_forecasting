import os
import xarray as xr
import numpy as np
import pandas as pd
import letkf_forecasting.letkf_io as letkf_io
import properscoring as ps
import sklearn.calibration as skcal


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
            dim=['south_north', 'west_east', 'time']).values.item())
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
        bias = bias.values.item()
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
        sd = np.sqrt(sd.values.item())
        sd_df.loc[horizon] = sd
        sd_truth = sd_truth.var(dim=['south_north', 'west_east', 'time'])
        sd_truth = np.sqrt(sd_truth.values.item())
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
    truth = letkf_io.add_crop_attributes(truth)
    truth = return_error_domain(truth)
    error_dfs = []
    for run in runs:
        full_day = letkf_io.return_day(year, month, day, run)
        full_day = letkf_io.add_crop_attributes(full_day)
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
    truth = letkf_io.add_crop_attributes(truth)
    truth = return_error_domain(truth)
    error_dfs = []
    spread_wind = []
    spread_ci = []
    for run in runs:
        full_day = letkf_io.return_day(year, month, day, run)
        full_day = letkf_io.add_crop_attributes(full_day)
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


def find_error_stats(year, month, day,
                     runs, base_folder):
    to_return = []
    for run in runs:
        results_file_path = os.path.join(
            base_folder,
            f'results/{year:04}/{month:02}/{day:02}/',
            run)
        results_file_path = letkf_io.find_latest_run(
            results_file_path)
        print(results_file_path)
        truth_sd = pd.read_hdf(
            os.path.join(results_file_path, 'truth_sd.h5'))
        adict = {'name': run, 'truth_sd': truth_sd}
        u_spread_file = os.path.join(results_file_path, 'u_spread.h5')
        if os.path.exists(u_spread_file):
            adict['u_spread'] = pd.read_hdf(
                u_spread_file)
            adict['v_spread'] = pd.read_hdf(
                os.path.join(results_file_path, 'v_spread.h5'))
            adict['spread_ci'] = pd.read_hdf(
                os.path.join(results_file_path, 'spread_ci.h5'))
        adict['rmse'] = pd.read_hdf(
            os.path.join(results_file_path, 'rmse.h5'))
        adict['forecast_sd'] = pd.read_hdf(
            os.path.join(results_file_path, 'forecast_sd.h5'))
        adict['bias'] = pd.read_hdf(
            os.path.join(results_file_path, 'bias.h5'))
        adict['correlation'] = pd.read_hdf(
            os.path.join(results_file_path, 'correlation.h5'))
        to_return.append(adict)
    return to_return


def error_stats(year, month, day, runs, base_folder, optimize_folder=None):
    truth = os.path.join(base_folder,
                         f'data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = xr.open_dataset(truth)
    truth = truth['ci']
    truth = letkf_io.add_crop_attributes(truth)
    truth = return_error_domain(truth)
    to_return = []
    truth_sd = np.sqrt(truth.var(dim=['south_north', 'west_east']))
    truth_sd = truth_sd.to_pandas()
    for run in runs:
        print(run)
        adict = {'name': run, 'truth_sd': truth_sd}
        if run == 'persistence':
            adict = return_persistence_dict(
                adict, truth, [15, 30, 45, 60])
            to_return.append(adict)
            continue
        full_day = letkf_io.return_day(year, month, day, run, base_folder,
                                       optimize_folder)
        full_day = letkf_io.add_crop_attributes(full_day)
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


def return_persistence_dict(adict, truth, horizons):
    rmse_df = []
    sd_df = []
    bias_df = []
    corr_df = []
    for horizon in horizons:
        forecast = truth.copy()
        forecast['time'] = forecast.time + pd.Timedelta(horizon, 'm')

        rmse = return_rmse(truth, forecast)
        rmse_df.append(rmse)

        sd = return_sd(truth, forecast)
        sd_df.append(sd)

        bias = return_bias(truth, forecast)
        bias_df.append(bias)

        corr = return_correlation(truth, forecast)
        corr_df.append(corr)
    adict['rmse'] = pd.concat(rmse_df, axis=1, keys=horizons)
    adict['forecast_sd'] = pd.concat(sd_df, axis=1, keys=horizons)
    adict['bias'] = pd.concat(bias_df, axis=1, keys=horizons)
    adict['correlation'] = pd.concat(corr_df, axis=1, keys=horizons)
    return adict


def error_stats_one_day(year, month, day, runs, base_folder):
    truth = os.path.join(base_folder,
                         f'data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = xr.open_dataset(truth)
    truth = truth['ci']
    truth = letkf_io.add_crop_attributes(truth)
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
        full_day = letkf_io.return_day(year, month, day, run, base_folder)
        full_day = letkf_io.add_crop_attributes(full_day)
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
            dim=['south_north', 'west_east', 'time']).values.item()
        rmse = np.sqrt(rmse)
        rmse_df.loc[horizon] = rmse

        sd_times = np.intersect1d(
            truth.time.to_pandas(),
            forecast.time.to_pandas())
        sd = forecast.sel(time=sd_times).var(
            dim=['south_north', 'west_east', 'time']).values.item()
        sd = np.sqrt(sd)
        sd_df.loc[horizon] = sd

        sd_truth = truth.sel(time=sd_times).var(
            dim=['south_north', 'west_east', 'time']).values.item()
        sd_truth = np.sqrt(sd_truth)
        sd_truth_df.loc[horizon] = sd_truth

        bias = forecast - truth
        bias_df.loc[horizon] = bias.mean(
            dim=['south_north', 'west_east', 'time']).values.item()

        truth_df = truth.to_dataframe(name='ci')
        fore_df = forecast.to_dataframe(name='ci')
        corr = fore_df.corrwith(truth_df['ci']).item()
        corr_df.loc[horizon] = corr
    adict['rmse'] = rmse_df
    adict['forecast_sd'] = sd_df
    adict['truth_sd'] = sd_truth_df
    adict['bias'] = bias_df
    adict['correlation'] = corr_df
    return adict


def error_stats_many_days(dates, runs, base_folder):
    truth = letkf_io.return_many_truths(dates, base_folder)
    truth = truth['ci']
    truth = letkf_io.add_crop_attributes(truth)
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
        all_days = letkf_io.return_many_days(dates, run, base_folder)
        all_days = all_days['ci']
        all_days = return_ens_mean(all_days)

        rmse = return_rmse_one_day(truth, all_days)
        forecast_sd, truth_sd = return_sd_one_day(truth, all_days)
        bias = return_bias_one_day(truth, all_days)
        corr = return_correlation_one_day(truth, all_days)

        adict['rmse'] = rmse
        adict['forecast_sd'] = forecast_sd
        adict['truth_sd'] = truth_sd
        adict['bias'] = bias
        adict['correlation'] = corr

        to_return.append(adict)
    return to_return
