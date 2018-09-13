import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from letkf_forecasting import (
    analyse_results,
    letkf_io)


def return_smoothing_data():
    runs = ['owp_opt']
    directory_name = 'third_set'
    error_stats = []
    for run in runs:
        load_directory = ('/a2/uaren/travis/'
                          + 'results/multi_day_error/'
                          + f'{directory_name}/{run}')
        adict = {'name': run}
        for stat_file in os.listdir(load_directory):
            stat_name = stat_file.split('.')[0]
            stat_file = os.path.join(load_directory,
                                     stat_file)
            adict[stat_name] = pd.read_hdf(stat_file)
        error_stats.append(adict)
    for this_stat in error_stats:
        if this_stat['name'] == 'owp_opt':
            owp_rmse = this_stat['rmse']
            owp_corr = this_stat['correlation']
            owp_sd = this_stat['forecast_sd']
    smoothing_params = np.array(
        [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63])
    smoothing_runs = ['opt_flow', 'wrf_no_div']
    these_smoothing_runs = []
    for run in smoothing_runs:
        for this_param in smoothing_params:
            these_smoothing_runs.append(
                run + '_' + str(this_param))
    directory_name = 'smoothing_runs'
    smoothing_stats = []
    for run in these_smoothing_runs:
        load_directory = ('/a2/uaren/travis/'
                          + 'results/multi_day_error/'
                          + f'{directory_name}/{run}')
        adict = {'name': run}
        for stat_file in os.listdir(load_directory):
            stat_name = stat_file.split('.')[0]
            stat_file = os.path.join(load_directory,
                                     stat_file)
            adict[stat_name] = pd.read_hdf(stat_file)
        smoothing_stats.append(adict)
    of_rmse = pd.DataFrame(
        index=[15, 30, 45, 60],
        columns=smoothing_params)
    of_corr = of_rmse.copy()
    of_sd = of_rmse.copy()
    wrf_rmse = of_rmse.copy()
    wrf_corr = of_rmse.copy()
    wrf_sd = of_rmse.copy()
    for this_stat in smoothing_stats:
        split_name = this_stat['name'].split('_')
        smoothing_param = int(split_name[-1])
        run_name = split_name[0]
        if run_name == 'wrf':
            wrf_rmse[smoothing_param] = this_stat['rmse']
            wrf_corr[smoothing_param] = this_stat['correlation']
            wrf_sd[smoothing_param] = this_stat['forecast_sd']
        elif run_name == 'opt':
            of_rmse[smoothing_param] = this_stat['rmse']
            of_corr[smoothing_param] = this_stat['correlation']
            of_sd[smoothing_param] = this_stat['forecast_sd']
    one_of_rmse = pd.DataFrame(
        index=[15, 30, 45, 60],
        columns=['rmse'])
    one_of_corr = pd.DataFrame(
        index=[15, 30, 45, 60],
        columns=['correlation'])
    one_of_sd = pd.DataFrame(
        index=[15, 30, 45, 60],
        columns=['forecast_sd'])
    one_wrf_rmse = one_of_rmse.copy()
    one_wrf_corr = one_of_corr.copy()
    one_wrf_sd = one_of_sd.copy()
    for hor in one_of_rmse.index:
        this_sd = owp_sd.loc[hor].values.item()
        to_minimize = np.abs(of_sd.loc[hor].values - this_sd)
        optimal_index = to_minimize.argmin()
        one_of_rmse.loc[hor] = of_rmse.loc[hor].iloc[optimal_index]
        one_of_corr.loc[hor] = of_corr.loc[hor].iloc[optimal_index]
        one_of_sd.loc[hor] = of_sd.loc[hor].iloc[optimal_index]

        to_minimize = np.abs(wrf_sd.loc[hor].values - this_sd)
        optimal_index = to_minimize.argmin()
        one_wrf_rmse.loc[hor] = wrf_rmse.loc[hor].iloc[optimal_index]
        one_wrf_corr.loc[hor] = wrf_corr.loc[hor].iloc[optimal_index]
        one_wrf_sd.loc[hor] = wrf_sd.loc[hor].iloc[optimal_index]
    to_return = {
        'owp_rmse': owp_rmse,
        'owp_corr': owp_corr,
        'owp_sd': owp_sd,
        'of_rmse': of_rmse,
        'of_corr': of_corr,
        'of_sd': of_sd,
        'wrf_rmse': wrf_rmse,
        'wrf_corr': wrf_corr,
        'wrf_sd': wrf_sd,
        'one_of_rmse': one_of_rmse,
        'one_of_corr': one_of_corr,
        'one_of_sd': one_of_sd,
        'one_wrf_rmse': one_wrf_rmse,
        'one_wrf_corr': one_wrf_corr,
        'one_wrf_sd': one_wrf_sd,
    }
    return to_return


def plot_smoothing_data(*, cpal_dict, marker_dict, legend_dict, format, dpi):
    to_plot = return_smoothing_data()
    save_directory = ('/home/travis/python_code/'
                      'letkf_forecasting/figures/'
                      'smoothing_plots')
    # RMSE
    plt.figure()
    plt.plot(to_plot['one_of_rmse'],
             color=cpal_dict['opt_flow'],
             marker=marker_dict['opt_flow'])
    plt.plot(to_plot['one_wrf_rmse'],
             color=cpal_dict['wrf_no_div'],
             marker=marker_dict['wrf_no_div'])
    plt.plot(to_plot['owp_rmse'].loc[slice(15, None)],
             color=cpal_dict['owp_opt'],
             marker=marker_dict['owp_opt'])
    plt.legend([legend_dict['opt_flow'],
                legend_dict['wrf_no_div'],
                legend_dict['owp_opt']])
    plt.title('RMSE for equal SD')
    plt.xlabel('Forecast horizon (min)')
    plt.ylabel('RMSE (CI)')
    plt.xlim([15, 60])
    plt.savefig(fname=os.path.join(save_directory,
                                   f'rmse.{format}'),
                format=format, dpi=dpi)

    # correlation
    plt.figure()
    plt.plot(to_plot['one_of_corr'],
             color=cpal_dict['opt_flow'],
             marker=marker_dict['opt_flow'])
    plt.plot(to_plot['one_wrf_corr'],
             color=cpal_dict['wrf_no_div'],
             marker=marker_dict['wrf_no_div'])
    plt.plot(to_plot['owp_corr'].loc[slice(15, None)],
             color=cpal_dict['owp_opt'],
             marker=marker_dict['owp_opt'])
    plt.legend([legend_dict['opt_flow'],
                legend_dict['wrf_no_div'],
                legend_dict['owp_opt']])
    plt.title('Correlation for equal SD')
    plt.xlabel('Forecast horizon (min)')
    plt.ylabel('Correlation')
    plt.xlim([15, 60])
    plt.savefig(fname=os.path.join(save_directory,
                                   f'correlation.{format}'),
                format=format, dpi=dpi)

    for hor in [15, 30, 45, 60]:
        # sd vs rmse
        plt.figure()
        plt.plot(
            to_plot['of_sd'].loc[hor],
            to_plot['of_rmse'].loc[hor],
            color=cpal_dict['opt_flow'],
            marker=marker_dict['opt_flow'])
        plt.plot(
            to_plot['wrf_sd'].loc[hor],
            to_plot['wrf_rmse'].loc[hor],
            color=cpal_dict['wrf_no_div'],
            marker=marker_dict['wrf_no_div'])
        plt.plot(
            to_plot['owp_sd'].loc[hor],
            to_plot['owp_rmse'].loc[hor].values,
            color=cpal_dict['owp_opt'],
            marker=marker_dict['owp_opt'])
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt']])
        plt.xlabel('Standard deviation (CI)')
        plt.ylabel('RMSE (CI)')
        plt.title(f'Standard deviation vs RMSE; Horizon: {hor}')
        plt.savefig(fname=os.path.join(save_directory,
                                       f'sd_vs_rmse_{hor}.{format}'),
                    format=format, dpi=dpi)

        # rmse
        plt.figure()
        plt.plot(
            to_plot['of_rmse'].loc[hor],
            color=cpal_dict['opt_flow'],
            marker=marker_dict['opt_flow'])
        plt.plot(
            to_plot['wrf_rmse'].loc[hor],
            color=cpal_dict['wrf_no_div'],
            marker=marker_dict['wrf_no_div'])
        plt.plot(
            to_plot['of_rmse'].columns,
            to_plot['owp_rmse'].loc[hor].values
            * np.ones(to_plot['of_rmse'].columns.size),
            '--',
            color=cpal_dict['owp_opt'],
            marker=marker_dict['owp_opt'])
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt']])
        plt.xlabel('Smoothing parameter')
        plt.ylabel('RMSE (CI)')
        plt.title(f'Smoothing vs RMSE; Horizon: {hor}')
        plt.savefig(fname=os.path.join(save_directory,
                                       f'rmse_{hor}.{format}'),
                    format=format, dpi=dpi)

        # correlation
        plt.figure()
        plt.plot(
            to_plot['of_corr'].loc[hor],
            color=cpal_dict['opt_flow'],
            marker=marker_dict['opt_flow'])
        plt.plot(
            to_plot['wrf_corr'].loc[hor],
            color=cpal_dict['wrf_no_div'],
            marker=marker_dict['wrf_no_div'])
        plt.plot(
            to_plot['of_corr'].columns,
            to_plot['owp_corr'].loc[hor].values
            * np.ones(to_plot['of_corr'].columns.size),
            '--',
            color=cpal_dict['owp_opt'],
            marker=marker_dict['owp_opt'])
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt']])
        plt.xlabel('Smoothing parameter')
        plt.ylabel('Correlation')
        plt.title(f'Smoothing vs Correlation; Horizon: {hor}')
        plt.savefig(fname=os.path.join(save_directory,
                                       f'corr_{hor}.{format}'),
                    format=format, dpi=dpi)

        # standard deviation
        plt.figure()
        plt.plot(
            to_plot['of_sd'].loc[hor],
            color=cpal_dict['opt_flow'],
            marker=marker_dict['opt_flow'])
        plt.plot(
            to_plot['wrf_sd'].loc[hor],
            color=cpal_dict['wrf_no_div'],
            marker=marker_dict['wrf_no_div'])
        plt.plot(
            to_plot['of_sd'].columns,
            to_plot['owp_sd'].loc[hor].values
            * np.ones(to_plot['of_sd'].columns.size),
            '--',
            color=cpal_dict['owp_opt'],
            marker=marker_dict['owp_opt'])
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt']])
        plt.xlabel('Smoothing parameter')
        plt.ylabel('Standard deviation (CI)')
        plt.title(f'Smoothing vs Standard deviation; Horizon: {hor}')
        plt.savefig(fname=os.path.join(save_directory,
                                       f'sd_{hor}.{format}'),
                    format=format, dpi=dpi)


def return_original_stats():
    runs = ['opt_flow', 'wrf_no_div', 'owp_opt', 'persistence']
    directory_name = 'third_set_only_cloudy'
    error_stats = {}
    for run in runs:
        load_directory = ('/a2/uaren/travis/'
                          + 'results/multi_day_error/'
                          + f'{directory_name}/{run}')
        adict = {'name': run}
        for stat_file in os.listdir(load_directory):
            stat_name = stat_file.split('.')[0]
            stat_file = os.path.join(load_directory,
                                     stat_file)
            adict[stat_name] = pd.read_hdf(stat_file)
        error_stats[run] = adict
    return error_stats


def plot_original_error(*, cpal_dict, marker_dict, legend_dict, format, dpi):
    to_plot = return_original_stats()
    save_directory = ('/home/travis/python_code/'
                      'letkf_forecasting/figures/'
                      'error_plots')
    # RMSE
    plt.figure()
    this_to_plot = to_plot[
        'persistence']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['persistence'],
             marker=marker_dict['persistence'])
    this_to_plot = to_plot[
        'opt_flow']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['opt_flow'],
             marker=marker_dict['opt_flow'])
    this_to_plot = to_plot[
        'wrf_no_div']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['wrf_no_div'],
             marker=marker_dict['wrf_no_div'])
    this_to_plot = to_plot[
        'owp_opt']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['owp_opt'],
             marker=marker_dict['owp_opt'])
    plt.legend([legend_dict['persistence'],
                legend_dict['opt_flow'],
                legend_dict['wrf_no_div'],
                legend_dict['owp_opt']])
    plt.title('RMSE')
    plt.xlabel('Forecast horizon (min)')
    plt.ylabel('RMSE (CI)')
    plt.xlim([15, 60])
    plt.savefig(fname=os.path.join(save_directory,
                                   f'rmse.{format}'),
                format=format, dpi=dpi)

    # correlation
    plt.figure()
    this_to_plot = to_plot[
        'persistence']['correlation'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['persistence'],
             marker=marker_dict['persistence'])
    this_to_plot = to_plot[
        'opt_flow']['correlation'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['opt_flow'],
             marker=marker_dict['opt_flow'])
    this_to_plot = to_plot[
        'wrf_no_div']['correlation'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['wrf_no_div'],
             marker=marker_dict['wrf_no_div'])
    this_to_plot = to_plot[
        'owp_opt']['correlation'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['owp_opt'],
             marker=marker_dict['owp_opt'])
    plt.legend([legend_dict['persistence'],
                legend_dict['opt_flow'],
                legend_dict['wrf_no_div'],
                legend_dict['owp_opt']])
    plt.title('Correlation')
    plt.xlabel('Forecast horizon (min)')
    plt.ylabel('Correlation')
    plt.xlim([15, 60])
    plt.savefig(fname=os.path.join(save_directory,
                                   f'correlation.{format}'),
                format=format, dpi=dpi)

    # forecast sd
    plt.figure()
    this_to_plot = to_plot[
        'persistence']['forecast_sd'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['persistence'],
             marker=marker_dict['persistence'])
    this_to_plot = to_plot[
        'opt_flow']['forecast_sd'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['opt_flow'],
             marker=marker_dict['opt_flow'])
    this_to_plot = to_plot[
        'wrf_no_div']['forecast_sd'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['wrf_no_div'],
             marker=marker_dict['wrf_no_div'])
    this_to_plot = to_plot[
        'owp_opt']['forecast_sd'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['owp_opt'],
             marker=marker_dict['owp_opt'])
    plt.legend([legend_dict['persistence'],
                legend_dict['opt_flow'],
                legend_dict['wrf_no_div'],
                legend_dict['owp_opt']])
    plt.title('Forecast standard deviation')
    plt.xlabel('Forecast horizon (min)')
    plt.ylabel('Standard deviation (CI)')
    plt.xlim([15, 60])
    plt.savefig(fname=os.path.join(save_directory,
                                   f'sd.{format}'),
                format=format, dpi=dpi)


def return_daily_error():
    runs = ['persistence', 'opt_flow', 'wrf_no_div', 'owp_opt']
    base_folder = '/a2/uaren/travis/'
    year = 2014
    month_day = [[4, 9],  [4, 15], [4, 18],
                 [5, 6],  [5, 9],  [5, 29],
                 [6, 11], [6, 12],
                 [4, 2],  [4, 5],  [4, 19],
                 [5, 7],  [5, 8],  [5, 19],
                 [6, 3],  [6, 10], [6, 14],
                 [6, 15],
                 [4, 10], [4, 11], [4, 12],
                 [4, 20], [4, 21], [4, 22],
                 [4, 25], [4, 26], [5, 5],
                 [5, 20], [5, 21], [5, 22],
                 [5, 23], [5, 24], [5, 25],
                 [5, 30], [6, 16], [6, 17],
                 [6, 18], [6, 19], [6, 22]]
    persistence = pd.DataFrame(columns=[15, 30, 45, 60])
    opt_flow = persistence.copy()
    wrf_no_div = persistence.copy()
    owp_opt = persistence.copy()
    for this_month_day in month_day:
        month = this_month_day[0]
        day = this_month_day[1]
        returned = analyse_results.find_error_stats(
            year, month, day, runs, base_folder)
        for this_stat in returned:
            name = this_stat['name']
            this_date = this_stat['rmse'].index[0].date()
            exec(
                name
                + '.loc[this_date] '
                + '= np.sqrt((this_stat[\'rmse\']**2)'
                + '.mean())')
    persistence = persistence.sort_index()
    opt_flow = opt_flow.sort_index()
    wrf_no_div = wrf_no_div.sort_index()
    owp_opt = owp_opt.sort_index()
    adict = {'persistence': persistence,
             'opt_flow': opt_flow,
             'wrf_no_div': wrf_no_div,
             'owp_opt': owp_opt}
    return adict


def plot_daily_error(*, cpal_dict, marker_dict, legend_dict, format, dpi):
    save_directory = ('/home/travis/python_code/'
                      'letkf_forecasting/figures/'
                      'daily_error')
    error_dict = return_daily_error()
    persistence = error_dict['persistence']
    opt_flow = error_dict['opt_flow']
    wrf_no_div = error_dict['wrf_no_div']
    owp_opt = error_dict['owp_opt']

    xticks = [str(index.month) + ' ' + str(index.day)
              for index in persistence.index]
    xarange = np.arange(len(xticks))
    figsize = plt.figaspect(0.3)
    width = 0.28

    for hor in [15, 30, 45, 60]:
        of_skill = 1 - opt_flow[hor]/persistence[hor]
        wrf_skill = 1 - wrf_no_div[hor]/persistence[hor]
        owp_skill = 1 - owp_opt[hor]/persistence[hor]

        plt.figure(figsize=figsize)
        plt.bar(xarange,
                of_skill, width,
                color=cpal_dict['opt_flow'])
        plt.bar(xarange + width,
                wrf_skill, width,
                color=cpal_dict['wrf_no_div'])
        plt.bar(xarange + 2*width,
                owp_skill, width,
                color=cpal_dict['owp_opt'])
        plt.xticks(xarange + width, xticks, rotation=90)
        plt.title(f'RMSE skill score; Horizon: {hor}')
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt']])
        plt.xlabel('Date')
        plt.ylabel('RMSE skill score')
        plt.ylim([0, None])
        plt.tight_layout()
        plt.savefig(fname=os.path.join(save_directory,
                                       f'rmse_ss_{hor}.{format}'),
                    format=format, dpi=dpi)

        plt.figure(figsize=figsize)
        plt.bar(xarange,
                opt_flow[hor], width,
                color=cpal_dict['opt_flow'])
        plt.bar(xarange + width,
                wrf_no_div[hor], width,
                color=cpal_dict['wrf_no_div'])
        plt.bar(xarange + 2*width,
                owp_opt[hor], width,
                color=cpal_dict['owp_opt'])
        plt.xticks(xarange + width, xticks, rotation=90)
        plt.title(f'RMSE; Horizon: {hor}')
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt']])
        plt.xlabel('Date')
        plt.ylabel('RMSE (CI)')
        plt.ylim([0, None])
        plt.tight_layout()
        plt.savefig(fname=os.path.join(save_directory,
                                       f'rmse_{hor}.{format}'),
                    format=format, dpi=dpi)


def return_spaghetti_error(*, month, day):
    year = 2014
    horizons = [15, 30, 45, 60]
    base_folder = '/a2/uaren/travis/'
    truth = xr.open_dataset(
        f'/a2/uaren/travis/data/{year:04}/{month:02}/{day:02}/data.nc')
    truth = truth['ci']
    truth = letkf_io.add_crop_attributes(truth)
    truth_full = truth.copy()
    truth = analyse_results.return_error_domain(truth)

    truth = truth.load()

    full_day = letkf_io.return_day(year,
                                   month,
                                   day,
                                   'owp_opt',
                                   base_folder)

    full_day = letkf_io.add_crop_attributes(full_day)
    full_day = analyse_results.return_error_domain(full_day)
    full_day = full_day['ci'].load()

    full_day_mean = analyse_results.return_ens_mean(full_day)

    wrf = letkf_io.return_day(year,
                              month,
                              day,
                              'wrf_no_div',
                              base_folder)

    wrf = letkf_io.add_crop_attributes(wrf)
    wrf = analyse_results.return_error_domain(wrf)
    wrf = wrf['ci'].load()

    opt_flow = letkf_io.return_day(year,
                                   month,
                                   day,
                                   'opt_flow',
                                   base_folder)

    opt_flow = letkf_io.add_crop_attributes(opt_flow)
    opt_flow = analyse_results.return_error_domain(opt_flow)
    opt_flow = opt_flow['ci'].load()

    full_day.ensemble_number.size

    ens_num = full_day.ensemble_number.size

    ensemble_rmse = np.ones([len(horizons), ens_num]) * np.nan
    ensemble_bias = np.ones([len(horizons), ens_num]) * np.nan

    mean_rmse = np.ones([len(horizons)]) * np.nan
    mean_bias = np.ones([len(horizons)]) * np.nan

    wrf_rmse = np.ones([len(horizons)]) * np.nan
    wrf_bias = np.ones([len(horizons)]) * np.nan

    opt_flow_rmse = np.ones([len(horizons)]) * np.nan
    opt_flow_bias = np.ones([len(horizons)]) * np.nan
    for ii, hor in enumerate(horizons):
        ensemble_rmse_temp = analyse_results.return_horizon(full_day, int(hor))
        ensemble_rmse_temp = ensemble_rmse_temp - truth
        ensemble_rmse[ii] = np.sqrt((ensemble_rmse_temp ** 2).mean(
            dim=['south_north', 'west_east', 'time'])).values
        ensemble_bias_temp = analyse_results.return_horizon(full_day, hor)
        ensemble_bias_temp = ensemble_bias_temp - truth
        ensemble_bias[ii] = ensemble_bias_temp.mean(
            dim=['south_north', 'west_east', 'time']).values

        mean_rmse_temp = analyse_results.return_horizon(full_day_mean, int(hor))
        mean_rmse_temp = mean_rmse_temp - truth
        mean_rmse[ii] = np.sqrt((mean_rmse_temp ** 2).mean(
            dim=['south_north', 'west_east', 'time'])).values
        mean_bias_temp = analyse_results.return_horizon(full_day_mean, hor)
        mean_bias_temp = mean_bias_temp - truth
        mean_bias[ii] = mean_bias_temp.mean(
            dim=['south_north', 'west_east', 'time']).values

        wrf_rmse_temp = analyse_results.return_horizon(wrf, int(hor))
        wrf_rmse_temp = wrf_rmse_temp - truth
        wrf_rmse[ii] = np.sqrt((wrf_rmse_temp ** 2).mean(
            dim=['south_north', 'west_east', 'time'])).values
        wrf_bias_temp = analyse_results.return_horizon(wrf, hor)
        wrf_bias_temp = wrf_bias_temp - truth
        wrf_bias[ii] = wrf_bias_temp.mean(
            dim=['south_north', 'west_east', 'time']).values

        opt_flow_rmse_temp = analyse_results.return_horizon(opt_flow, int(hor))
        opt_flow_rmse_temp = opt_flow_rmse_temp - truth
        opt_flow_rmse[ii] = np.sqrt((opt_flow_rmse_temp ** 2).mean(
            dim=['south_north', 'west_east', 'time'])).values
        opt_flow_bias_temp = analyse_results.return_horizon(opt_flow, hor)
        opt_flow_bias_temp = opt_flow_bias_temp - truth
        opt_flow_bias[ii] = opt_flow_bias_temp.mean(
            dim=['south_north', 'west_east', 'time']).values
    return_dict = {'horizons': horizons,
                   'ensemble_rmse': ensemble_rmse,
                   'ensemble_bias': ensemble_bias,
                   'mean_rmse': mean_rmse,
                   'mean_bias': mean_bias,
                   'wrf_rmse': wrf_rmse,
                   'wrf_bias': wrf_bias,
                   'opt_flow_rmse': opt_flow_rmse,
                   'opt_flow_bias': opt_flow_bias}
    return return_dict


def plot_spaghetti(*, cpal_dict, legend_dict, marker_dict,
                   format, dpi, dates_dict):
    for day_type, month_day in dates_dict.items():
        print(month_day)
        month = month_day[0]
        day = month_day[1]
        save_directory = ('/home/travis/python_code/'
                          'letkf_forecasting/figures/'
                          + day_type)
        returned_dict = return_spaghetti_error(month=month,
                                               day=day)

        horizons = returned_dict['horizons']
        ensemble_rmse = returned_dict['ensemble_rmse']
        ensemble_bias = returned_dict['ensemble_bias']
        mean_rmse = returned_dict['mean_rmse']
        mean_bias = returned_dict['mean_bias']
        wrf_rmse = returned_dict['wrf_rmse']
        wrf_bias = returned_dict['wrf_bias']
        opt_flow_rmse = returned_dict['opt_flow_rmse']
        opt_flow_bias = returned_dict['opt_flow_bias']

        # RMSE
        plt.figure()
        plt.plot(horizons, opt_flow_rmse,
                 marker=marker_dict['opt_flow'],
                 color=cpal_dict['opt_flow'])
        plt.plot(horizons, wrf_rmse,
                 marker=marker_dict['wrf_no_div'],
                 color=cpal_dict['wrf_no_div'])
        plt.plot(horizons, mean_rmse,
                 marker=marker_dict['owp_opt'],
                 color=cpal_dict['owp_opt'])
        plt.plot(horizons, ensemble_rmse, alpha=0.5, linestyle=':',
                 color=cpal_dict['ens_member'])
        plt.xlim([min(horizons), max(horizons)])
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt'],
                    legend_dict['ens_member']])
        plt.xlabel('Horizon')
        plt.ylabel('RMSE (CI)')
        plt.title(f'RMSE: {month}/{day}')
        plt.savefig(fname=os.path.join(save_directory,
                                       f'rmse.{format}'),
                    format=format, dpi=dpi)

        # Bias
        plt.figure()
        plt.plot(horizons, opt_flow_bias,
                 marker=marker_dict['opt_flow'],
                 color=cpal_dict['opt_flow'])
        plt.plot(horizons, wrf_bias,
                 marker=marker_dict['wrf_no_div'],
                 color=cpal_dict['wrf_no_div'])
        plt.plot(horizons, mean_bias,
                 marker=marker_dict['owp_opt'],
                 color=cpal_dict['owp_opt'])
        plt.plot(horizons, ensemble_bias, alpha=0.5, linestyle=':',
                 color=cpal_dict['ens_member'])
        plt.xlim([min(horizons), max(horizons)])
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt'],
                    legend_dict['ens_member']])
        plt.xlabel('Horizon')
        plt.ylabel('Bias (CI)')
        plt.title(f'Bias: {month}/{day}')
        plt.savefig(fname=os.path.join(save_directory,
                                       f'bias.{format}'),
                    format=format, dpi=dpi)

        # Absolute of Bias
        plt.figure()
        plt.plot(horizons, np.abs(opt_flow_bias),
                 marker=marker_dict['opt_flow'],
                 color=cpal_dict['opt_flow'])
        plt.plot(horizons, np.abs(wrf_bias),
                 marker=marker_dict['wrf_no_div'],
                 color=cpal_dict['wrf_no_div'])
        plt.plot(horizons, np.abs(mean_bias),
                 marker=marker_dict['owp_opt'],
                 color=cpal_dict['owp_opt'])
        plt.plot(horizons, np.abs(ensemble_bias), alpha=0.5, linestyle=':',
                 color=cpal_dict['ens_member'])
        plt.xlim([min(horizons), max(horizons)])
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt'],
                    legend_dict['ens_member']])
        plt.xlabel('Horizon')
        plt.ylabel('Bias (CI)')
        plt.title(f'Absolute of Bias: {month}/{day}')
        plt.savefig(fname=os.path.join(save_directory,
                                       f'abs_bias.{format}'),
                    format=format, dpi=dpi)


def main():
    format = 'png'
    dpi = 300
    cpal = sns.color_palette('deep')
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.0,
                    rc={'lines.linewidth': 1.0,
                        'lines.markersize': 7})
    cpal_dict = {'opt_flow': cpal[0],
                 'wrf_no_div': cpal[1],
                 'owp_opt': cpal[2],
                 'persistence': cpal[3],
                 'ens_member': cpal[4]}
    legend_dict = {'opt_flow': 'Opt. Flow',
                   'wrf_no_div': 'WRF',
                   'owp_opt': 'BACON',
                   'persistence': 'Persistence',
                   'ens_member': 'Ens. Member'}
    marker_dict = {'opt_flow': 'o',
                   'wrf_no_div': '^',
                   'owp_opt': 'd',
                   'persistence': 's',
                   'ens_member': '*'}

    # # plot smoothed
    # plot_smoothing_data(cpal_dict=cpal_dict,
    #                     legend_dict=legend_dict,
    #                     marker_dict=marker_dict,
    #                     format=format,
    #                     dpi=dpi)
    # plt.close('all')

    # # plot original error plots
    # plot_original_error(cpal_dict=cpal_dict,
    #                     legend_dict=legend_dict,
    #                     marker_dict=marker_dict,
    #                     format=format,
    #                     dpi=dpi)
    # plt.close('all')

    # # plot error by date plots
    # plot_daily_error(cpal_dict=cpal_dict,
    #                  legend_dict=legend_dict,
    #                  marker_dict=marker_dict,
    #                  format=format,
    #                  dpi=dpi)
    # plt.close('all')

    # get spaghetti data
    dates_dict = {
        'translation': (4, 15),
        'more_complex': (5, 29),
        'two_levels': (4, 26)
    }
    plot_spaghetti(cpal_dict=cpal_dict,
                   legend_dict=legend_dict,
                   marker_dict=marker_dict,
                   format=format,
                   dpi=dpi,
                   dates_dict=dates_dict)
    plt.close('all')


if __name__ == '__main__':
    main()
