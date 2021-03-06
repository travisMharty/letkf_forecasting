import os
import datetime
import numba
import xarray as xr
import numpy as np
import pandas as pd
import letkf_forecasting.letkf_io as letkf_io
import properscoring as ps
import sklearn.calibration as calibration


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


def return_rmse_one_day(truth, full_day, horizons,
                        cloudy_times=None):
    if cloudy_times is not None:
        error_times = cloudy_times
    else:
        error_times = truth.time
    total_error_times = pd.DataFrame(
        data=False,
        columns=horizons,
        index=error_times)
    if 'ensemble_number' in full_day.coords:
        rmse_df = pd.DataFrame(columns=full_day.ensemble_number.values)
    else:
        rmse_df = pd.DataFrame(columns=['rmse'])
    for horizon in horizons:
        forecast = return_horizon(full_day, horizon)
        these_error_times = np.intersect1d(
            truth.sel(time=error_times).time.to_pandas(),
            forecast.time.to_pandas())
        total_error_times[horizon].loc[these_error_times] = True
        rmse = (forecast - truth)**2
        rmse = np.sqrt(
            rmse.mean(dim=['south_north', 'west_east', 'time']).values)
        rmse = np.array(rmse)
        if rmse.size > 1:
            rmse_df.loc[horizon] = rmse
        else:
            rmse = rmse.item()
            rmse_df.loc[horizon] = rmse
    return rmse_df, total_error_times


def return_bias(truth, full_day):
    bias = truth - full_day
    bias = bias.mean(dim=['south_north', 'west_east'])
    bias = bias.to_pandas()
    return bias


def return_bias_one_day(truth, full_day, horizons,
                        total_error_times):
    if 'ensemble_number' in full_day.coords:
        bias_df = pd.DataFrame(columns=full_day.ensemble_number.values)
    else:
        bias_df = pd.DataFrame(columns=['bias'])
    for horizon in horizons:
        these_error_times = total_error_times.index[
            total_error_times[horizon]]
        bias = return_horizon(full_day, horizon)
        bias = (bias - truth).sel(time=these_error_times)
        bias = bias.mean(dim=['south_north', 'west_east', 'time'])
        bias = bias.values
        bias = np.array(bias)
        if bias.size > 1:
            bias_df.loc[horizon] = bias
        else:
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


def return_correlation_one_day(truth, full_day, horizons,
                               total_error_times):
    if 'ensemble_number' in full_day.coords:
        R_df = pd.DataFrame(columns=full_day.ensemble_number.values)
        ens_flag = True
    else:
        R_df = pd.DataFrame(columns=['correlation'])
        ens_flag = False
    truth_df = truth.to_dataframe(name='ci')
    for horizon in horizons:
        these_error_times = total_error_times.index[
                total_error_times[horizon]]
        fore_df = return_horizon(full_day, horizon)
        fore_df = fore_df.sel(time=these_error_times)
        if not ens_flag:
            fore_df = fore_df.to_dataframe(name='ci')
            R = fore_df.corrwith(truth_df['ci']).item()
            R_df.loc[horizon] = R
        else:
            for ens_num in R_df.columns:
                this_fore_df = fore_df.sel(ensemble_number=ens_num)
                this_fore_df = this_fore_df.drop('ensemble_number')
                this_fore_df = this_fore_df.to_dataframe(name='ci')
                R = this_fore_df.corrwith(truth_df['ci'])
                R = R.item()
                R_df[ens_num].loc[horizon] = R
    return R_df


def return_sd(truth, full_day):
    sd = full_day
    sd = sd.var(dim=['south_north', 'west_east'])
    sd = np.sqrt(sd)
    sd = sd.to_pandas()
    return sd


def return_sd_one_day(truth, full_day, horizons,
                      total_error_times):
    if 'ensemble_number' in full_day.coords:
        sd_df = pd.DataFrame(columns=full_day.ensemble_number.values)
    else:
        sd_df = pd.DataFrame(columns=['sd'])
    sd_truth_df = pd.DataFrame(columns=['true_sd'])
    for horizon in horizons:
        these_error_times = total_error_times.index[
            total_error_times[horizon]]
        sd = return_horizon(full_day, horizon)
        sd = sd.sel(time=these_error_times)
        sd_truth = truth.sel(time=these_error_times)
        sd = sd.var(dim=['south_north', 'west_east', 'time'])
        sd = np.sqrt(sd.values)
        sd = np.array(sd)
        if sd.size > 1:
            sd_df.loc[horizon] = sd
        else:
            sd = sd.item()
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
    for run in runs:
        print(run)
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


def return_persistence_dict_one_day(adict, truth, horizons,
                                    cloudy_times=None):
    if cloudy_times is not None:
        error_times = cloudy_times
    else:
        error_times = truth.time
    rmse_df = pd.DataFrame(columns=['rmse'])
    sd_df = pd.DataFrame(columns=['sd'])
    sd_truth_df = pd.DataFrame(columns=['true_sd'])
    bias_df = pd.DataFrame(columns=['bias'])
    corr_df = pd.DataFrame(columns=['correlation'])
    total_error_times = pd.DataFrame(
        data=False,
        columns=horizons,
        index=error_times)
    for horizon in horizons:
        forecast = truth.copy()
        forecast['time'] = forecast.time + pd.Timedelta(horizon, 'm')
        these_error_times = np.intersect1d(
            truth.sel(time=error_times).time.to_pandas(),
            forecast.time.to_pandas())
        total_error_times[horizon].loc[these_error_times] = True
        forecast = forecast.sel(time=these_error_times)
        rmse = (forecast - truth.sel(time=these_error_times))**2
        rmse = rmse.mean(
            dim=['south_north', 'west_east', 'time']).values.item()
        rmse = np.sqrt(rmse)
        rmse_df.loc[horizon] = rmse

        sd = forecast.sel(time=these_error_times).var(
            dim=['south_north', 'west_east', 'time']).values.item()
        sd = np.sqrt(sd)
        sd_df.loc[horizon] = sd

        sd_truth = truth.sel(time=these_error_times).var(
            dim=['south_north', 'west_east', 'time']).values.item()
        sd_truth = np.sqrt(sd_truth)
        sd_truth_df.loc[horizon] = sd_truth

        bias = forecast - truth.sel(time=these_error_times)
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
    adict['stat_times'] = total_error_times
    return adict


def down_sample_coord(coord, dx):
    coord_ds = np.arange(coord[0],
                         coord[-1] + dx,
                         dx)
    return coord_ds


def error_stats_many_days(dates, runs, horizons, base_folder,
                          only_cloudy=False, only_of_times=True,
                          mean_win_size=None, one_km_err=False):
    truth = letkf_io.return_many_truths(dates, base_folder)
    truth = truth['ci']
    truth = letkf_io.add_crop_attributes(truth)
    truth = return_error_domain(truth)
    if only_of_times:
        truth_times = truth.time.to_pandas()
        these_dates = np.unique(truth_times.index.date)
        keep_times = pd.Series()
        for this_date in these_dates:
            keep_times = pd.concat(
                [keep_times, truth_times.loc[str(this_date)].iloc[1:]])
        truth = truth.sel(time=keep_times.index)
    if only_cloudy:
        print('only_cloudy')
        truth_max = truth.max(dim=['south_north', 'west_east'])
        truth_mean = truth.mean(dim=['south_north', 'west_east'])
        bool_max = truth_max > 0.2
        bool_mean = truth_mean > 0.1
        cloudy_bool = xr.ufuncs.logical_or(
            bool_max, bool_mean)
        cloudy_times = truth.time[cloudy_bool]
    else:
        cloudy_times = None
    if one_km_err:
        west_east_err = down_sample_coord(truth.west_east, 1)
        south_north_err = down_sample_coord(truth.south_north, 1)
        truth = truth.sel(west_east=west_east_err,
                          south_north=south_north_err)
    to_return = []
    for run in runs:
        print(run)
        adict = {'name': run}
        if run == 'persistence':
            adict = return_persistence_dict_one_day(
                adict, truth, horizons,
                cloudy_times=cloudy_times)
            to_return.append(adict)
            continue
        ens_flag = False
        analysis_fore_flag = False
        if run[0] == 'ensemble':
            ens_flag = True
            run = run[1]
        elif run[0] == 'anly_fore':
            run = run[1]
            analysis_fore_flag = True
        all_days = letkf_io.return_many_days(
            dates, run, base_folder,
            only_of_times=only_of_times,
            mean_win_size=mean_win_size,
            analysis_fore_flag=analysis_fore_flag)
        if mean_win_size is None:
            all_days = all_days['ci']
        if one_km_err:
            all_days = all_days.sel(west_east=west_east_err,
                                    south_north=south_north_err)
        if not ens_flag:
            all_days = return_ens_mean(all_days)
        rmse, total_error_times = return_rmse_one_day(
            truth, all_days, horizons,
            cloudy_times=cloudy_times)
        forecast_sd, truth_sd = return_sd_one_day(
            truth, all_days, horizons,
            total_error_times=total_error_times)
        bias = return_bias_one_day(
            truth, all_days, horizons,
            total_error_times=total_error_times)
        corr = return_correlation_one_day(
            truth, all_days, horizons,
            total_error_times=total_error_times)

        adict['rmse'] = rmse
        adict['forecast_sd'] = forecast_sd
        adict['truth_sd'] = truth_sd
        adict['bias'] = bias
        adict['correlation'] = corr
        adict['stat_times'] = total_error_times

        to_return.append(adict)
    return to_return


def prob_analysis_baselines(month_day, horizons, file_path,
                            base_folder='/a2/uaren/travis', ):
    ens_members = 50
    climatology = pd.read_hdf(file_path)
    climatology = climatology.values.ravel()
    climatology = climatology[~np.isnan(climatology)]
    climatology = climatology.clip(min=0, max=1)
    for this_month_day in month_day:
        print(this_month_day)
        year = 2014
        month = this_month_day[0]
        day = this_month_day[1]
        truth = os.path.join(
            base_folder,
            f'data/{year:04}/{month:02}/{day:02}/data.nc')
        truth = xr.open_dataset(truth)
        truth = truth['ci']
        truth = letkf_io.add_crop_attributes(truth)
        truth = return_error_domain(truth)

        full_index = truth.time.to_pandas().index

        crps_persistence = pd.DataFrame(
            index=full_index,
            columns=horizons)
        crps_persistent_dist = pd.DataFrame(
            index=full_index,
            columns=horizons)
        crps_climatology = pd.DataFrame(
            index=full_index,
            columns=horizons)
        crps_dates_climatology = pd.DataFrame(
            index=full_index,
            columns=horizons)
        for horizon in horizons:
            persistence = truth.copy()
            time_step = pd.Timedelta(str(horizon) + 'min')
            persistence['time'] = persistence.time + time_step
            dates_error_times = np.intersect1d(
                truth.time.to_pandas(),
                persistence.time.to_pandas())
            this_truth = truth.sel(time=dates_error_times)
            this_index = this_truth.time.to_pandas().index
            persistence = persistence.sel(time=dates_error_times)

            # for persistence
            this_crps = ps.crps_ensemble(
                this_truth.values, persistence.values)
            this_crps = pd.Series(this_crps.mean(axis=(1, 2)),
                                  index=this_index)
            crps_persistence[horizon] = this_crps

            persis_array = persistence.values
            persis_shape = persis_array.shape
            persis_array = persis_array.reshape(
                persis_shape[0], persis_shape[1] * persis_shape[2])
            persis_weights = np.ones(
                [dates_error_times.size, ens_members])
            for ii in range(dates_error_times.size):
                persis_weights[ii], bin_edges = np.histogram(
                    persis_array[ii], bins=ens_members,
                    range=(0, 1), normed=True)
            persis_ens = (bin_edges[:-1] + bin_edges[1:])/2
            persis_ens = np.repeat(persis_ens[None, :],
                                   persis_shape[2], axis=0)
            persis_ens = np.repeat(persis_ens[None, :],
                                   persis_shape[1], axis=0)
            persis_ens = np.repeat(persis_ens[None, :],
                                   persis_shape[0], axis=0)

            persis_weights = np.repeat(persis_weights[:, None, :],
                                       persis_shape[2], axis=1)
            persis_weights = np.repeat(persis_weights[:, None, :, :],
                                       persis_shape[1], axis=1)

            # for persistent distribution
            this_crps = ps.crps_ensemble(
                this_truth.sel(time=dates_error_times).values,
                persis_ens, weights=persis_weights)

            this_crps = pd.Series(this_crps.mean(axis=(1, 2)),
                                  index=this_index)
            crps_persistent_dist[horizon] = this_crps

            clim_weights, bin_edges = np.histogram(
                climatology, bins=ens_members, range=(0, 1), normed=True)
            clim_ens = (bin_edges[:-1] + bin_edges[1:])/2
            clim_ens = np.repeat(clim_ens[None, :], persis_shape[2], axis=0)
            clim_ens = np.repeat(clim_ens[None, :], persis_shape[1], axis=0)
            clim_ens = np.repeat(clim_ens[None, :], persis_shape[0], axis=0)

            clim_weights = np.repeat(
                clim_weights[None, :], persis_shape[2], axis=0)
            clim_weights = np.repeat(
                clim_weights[None, :], persis_shape[1], axis=0)
            clim_weights = np.repeat(
                clim_weights[None, :], persis_shape[0], axis=0)

            # for climatology
            this_crps = ps.crps_ensemble(
                this_truth.sel(time=dates_error_times).values,
                clim_ens, weights=clim_weights)
            this_crps = pd.Series(this_crps.mean(axis=(1, 2)),
                                  index=this_index)
            crps_climatology[horizon] = this_crps

            climatology_dates = this_truth.values.ravel()
            climatology_dates = climatology_dates[~np.isnan(climatology_dates)]
            climatology_dates = climatology_dates.clip(min=0, max=1)
            clim_dates_weights, bin_edges = np.histogram(
                climatology_dates, bins=ens_members, range=(0, 1), normed=True)
            clim_dates_ens = (bin_edges[:-1] + bin_edges[1:])/2
            clim_dates_ens = np.repeat(clim_dates_ens[None, :],
                                       persis_shape[2], axis=0)
            clim_dates_ens = np.repeat(clim_dates_ens[None, :],
                                       persis_shape[1], axis=0)
            clim_dates_ens = np.repeat(clim_dates_ens[None, :],
                                       persis_shape[0], axis=0)
            clim_dates_weights = np.repeat(clim_dates_weights[None, :],
                                           persis_shape[2], axis=0)
            clim_dates_weights = np.repeat(clim_dates_weights[None, :],
                                           persis_shape[1], axis=0)
            clim_dates_weights = np.repeat(clim_dates_weights[None, :],
                                           persis_shape[0], axis=0)

            # for dates climatology
            this_crps = ps.crps_ensemble(
                this_truth.sel(time=dates_error_times).values,
                clim_dates_ens, weights=clim_dates_weights)
            this_crps = pd.Series(this_crps.mean(axis=(1, 2)),
                                  index=this_index)
            crps_dates_climatology[horizon] = this_crps

        adict = {'persistence': crps_persistence,
                 'persistent_dist': crps_persistent_dist,
                 'climatology': crps_climatology,
                 'dates_climatology': crps_dates_climatology}

        for key, value in adict.items():
            save_directory = (
                '/a2/uaren/travis/'
                + f'results/2014/{month:02}/{day:02}/{key}_000')
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            save_file = os.path.join(save_directory, 'crps.h5')
            value.to_hdf(save_file, 'crps')


def prob_analysis_runs(month_day, runs, horizons,
                       base_folder='/a2/uaren/travis', ):
    for this_month_day in month_day:
        print(this_month_day)
        year = 2014
        month = this_month_day[0]
        day = this_month_day[1]
        truth = os.path.join(
            base_folder,
            f'data/{year:04}/{month:02}/{day:02}/data.nc')
        truth = xr.open_dataset(truth)
        truth = truth['ci']
        truth = letkf_io.add_crop_attributes(truth)
        truth = return_error_domain(truth)
        truth = truth.load()

        full_index = truth.time.to_pandas().index
        for run in runs:
            crps_df = pd.DataFrame(
                index=full_index,
                columns=horizons)
            print(run)
            full_day = letkf_io.return_day(
                year, month, day, run, base_folder)
            full_day = letkf_io.add_crop_attributes(full_day)
            full_day = return_error_domain(full_day)
            full_day = full_day['ci']
            full_day = full_day.load()
            for horizon in horizons:
                this_full_day = return_horizon(full_day, horizon)
                these_error_times = np.intersect1d(
                    full_index, this_full_day.time.to_pandas().index)
                this_full_day = this_full_day.sel(time=these_error_times)
                this_truth = truth.sel(time=these_error_times)
                this_crps = ps.crps_ensemble(
                    this_truth.values,
                    this_full_day.values.transpose([0, 2, 3, 1]))
                this_crps = pd.Series(this_crps.mean(axis=(1, 2)),
                                      index=these_error_times)
                crps_df[horizon] = this_crps
            file_path = os.path.join(
                base_folder,
                'results',
                f'{year:04}',
                f'{month:02}',
                f'{day:02}',
                run)
            file_path = letkf_io.find_latest_run(file_path)
            file_path = os.path.join(file_path, 'crps.h5')
            crps_df.to_hdf(file_path, 'crps')


def fraction_of_positives_runs(month_day, runs, horizons, bounds_dict, N_bins,
                               base_folder='/a2/uaren/travis', ):
    bins = np.arange(N_bins)
    multi_column = [np.repeat(horizons, bins.size),
                    np.tile(bins, len(horizons))]
    multi_column = list(zip(*multi_column))
    multi_column = pd.MultiIndex.from_tuples(
        multi_column, names=['horizon', 'bin'])

    for this_month_day in month_day:
        print(this_month_day)
        year = 2014
        month = this_month_day[0]
        day = this_month_day[1]
        truth = os.path.join(
            base_folder,
            f'data/{year:04}/{month:02}/{day:02}/data.nc')
        truth = xr.open_dataset(truth)
        truth = truth['ci']
        truth = letkf_io.add_crop_attributes(truth)
        truth = return_error_domain(truth)
        truth = truth.load()

        full_index = truth.time.to_pandas().index
        for run in runs:
            print(run)
            full_day = letkf_io.return_day(
                year, month, day, run, base_folder)
            full_day = letkf_io.add_crop_attributes(full_day)
            full_day = return_error_domain(full_day)
            full_day = full_day['ci']
            full_day = full_day.load()
            for bound_name, bounds in bounds_dict.items():
                print(bound_name)
                if bounds[0] == 0:
                    truth_bounded = (truth < bounds[1]).astype('float')
                    full_day_bounded = (full_day < bounds[1]).astype('float')
                elif bounds[1] == 1:
                    truth_bounded = (truth >= bounds[0]).astype('float')
                    full_day_bounded = (full_day >= bounds[0]).astype('float')
                else:
                    truth_bounded = np.logical_and(
                        truth >= bounds[0],
                        truth < bounds[1]).astype('float')
                    full_day_bounded = np.logical_and(
                        full_day >= bounds[0],
                        full_day < bounds[1]).astype('float')
                brier_score = pd.DataFrame(
                    index=full_index,
                    columns=horizons)
                fraction_of_positives = pd.DataFrame(
                    index=full_index,
                    columns=multi_column)
                mean_predicted_prob = fraction_of_positives.copy()
                forecast_hist = fraction_of_positives.copy()
                truth_hist = pd.DataFrame(
                    index=full_index,
                    columns=bins)
                for tt in range(truth_bounded.shape[0]):
                    hist, temp = np.histogram(
                        truth_bounded.values[tt],
                        bins=N_bins,
                        range=(0, 1))
                    truth_hist.iloc[tt] = hist
                for horizon in horizons:
                    this_full_day = return_horizon(full_day_bounded, horizon)
                    these_error_times = np.intersect1d(
                        full_index, this_full_day.time.to_pandas().index)
                    this_full_day = this_full_day.sel(time=these_error_times)
                    this_full_day = this_full_day.mean(dim='ensemble_number')
                    # account for boundary cases
                    this_full_day = (this_full_day - 1e-8).clip(0, 1)
                    this_truth = truth_bounded.sel(time=these_error_times)

                    this_brier_score = ps.brier_score(
                        this_truth.values.ravel(),
                        this_full_day.values.ravel())
                    this_brier_score = this_brier_score.reshape(
                        this_truth.shape).mean(axis=(1, 2))
                    this_brier_score = pd.Series(this_brier_score,
                                                 index=these_error_times)
                    brier_score[horizon]  = this_brier_score

                    this_fraction_of_positives = np.ones(
                        [this_truth.shape[0], N_bins]) * np.nan
                    this_mean_predicted_prob = np.ones(
                        [this_truth.shape[0], N_bins]) * np.nan
                    this_forecast_hist = this_fraction_of_positives.copy()

                    for tt in range(this_truth.shape[0]):
                        this_forecast_hist[tt], temp = np.histogram(
                            this_full_day.values[tt],
                            bins=N_bins,
                            range=(0, 1))
                        fop, mpp = calibration.calibration_curve(
                            this_truth.values[tt].ravel(),
                            this_full_day.values[tt].ravel(),
                            n_bins=N_bins)
                        if fop.size < N_bins:
                            correct_bins = np.floor(mpp*N_bins).astype('int')
                            indexes = np.setdiff1d(bins, correct_bins)
                            indexes -= np.arange(indexes.size)
                            fop = np.insert(fop, indexes, 0)
                            mpp = np.insert(mpp, indexes, 0)
                        this_fraction_of_positives[tt] = fop
                        this_mean_predicted_prob[tt] = mpp
                    this_forcast_hist = pd.DataFrame(
                        this_forecast_hist,
                        index=these_error_times,
                        columns=bins)
                    forecast_hist[horizon] = this_forcast_hist
                    this_fraction_of_positives = pd.DataFrame(
                        this_fraction_of_positives,
                        index=these_error_times,
                        columns=bins)
                    fraction_of_positives[horizon] = this_fraction_of_positives
                    this_mean_predicted_prob = pd.DataFrame(
                        this_mean_predicted_prob,
                        index=these_error_times,
                        columns=bins)
                    mean_predicted_prob[horizon] = this_mean_predicted_prob
                file_path = os.path.join(
                    base_folder,
                    'results',
                    f'{year:04}',
                    f'{month:02}',
                    f'{day:02}',
                    run)
                file_path = letkf_io.find_latest_run(file_path)
                this_folder = (bound_name
                               + '_' + str(bounds[0]).replace('.', 'p')
                               + '_' + str(bounds[1]).replace('.', 'p'))
                file_path = os.path.join(
                    file_path, this_folder)
                if not os.path.exists(file_path):
                    os.mkdir(file_path)

                this_file_path = os.path.join(file_path, 'brier_score.h5')
                brier_score.to_hdf(this_file_path, 'brier_score')
                this_file_path = os.path.join(file_path, 'truth_hist.h5')
                truth_hist.to_hdf(this_file_path, 'truth_hist')
                this_file_path = os.path.join(file_path, 'forecast_hist.h5')
                forecast_hist.to_hdf(this_file_path, 'forecast_hist')
                this_file_path = os.path.join(file_path,
                                              'fraction_of_positives.h5')
                fraction_of_positives.to_hdf(this_file_path,
                                             'fraction_of_positives')
                this_file_path = os.path.join(file_path,
                                              'mean_predicted_prob.h5')
                mean_predicted_prob.to_hdf(this_file_path,
                                           'mean_predicted_prob')


def read_prob_stats(month_day, runs, stats,
                    base_folder='/a2/uaren/travis'):
    to_return = []
    year = 2014
    baselines = ['climatology', 'dates_climatology',
                 'persistence', 'persistent_dist']
    for run in runs:
        print(run)
        adict = {'name': run}
        for stat in stats:
            print(stat)
            this_stat = pd.DataFrame()
            for this_month_day in month_day:
                month = this_month_day[0]
                day = this_month_day[1]
                file_path = os.path.join(
                    base_folder,
                    'results',
                    f'{year:04}',
                    f'{month:02}',
                    f'{day:02}',
                    run)
                if run not in baselines:
                    file_path = letkf_io.find_latest_run(file_path)
                file_path = os.path.join(
                    file_path, stat + '.h5')
                temp_stat = pd.read_hdf(file_path)
                this_stat = pd.concat([this_stat, temp_stat])
            adict[stat] = this_stat
        to_return.append(adict)
    return to_return
