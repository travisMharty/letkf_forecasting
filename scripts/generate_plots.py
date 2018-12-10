import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from letkf_forecasting import (
    analyse_results,
    letkf_io)


def return_smoothing_data(directory_name):
    runs = ['owp_opt']
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
        # 'owp_rmse': owp_rmse,
        # 'owp_sd': owp_sd,
        # 'of_rmse': of_rmse,
        # 'of_corr': of_corr,
        # 'of_sd': of_sd,
        # 'wrf_rmse': wrf_rmse,
        # 'wrf_corr': wrf_corr,
        # 'wrf_sd': wrf_sd,
        'one_of_rmse': one_of_rmse,
        # 'one_of_corr': one_of_corr,
        # 'one_of_sd': one_of_sd,
        'one_wrf_rmse': one_wrf_rmse,
        # 'one_wrf_corr': one_wrf_corr,
        # 'one_wrf_sd': one_wrf_sd,
    }
    return to_return


def plot_smoothing_data(*, cpal_dict, marker_dict, legend_dict, format, dpi,
                        smoothing_error, averaged_error, save_directory):
    save_directory = os.path.join(save_directory,
                                  'smoothing_plots')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # RMSE
    plt.figure()
    plt.plot(smoothing_error['one_of_rmse'],
             color=cpal_dict['opt_flow'],
             marker=marker_dict['opt_flow'])
    plt.plot(smoothing_error['one_wrf_rmse'],
             color=cpal_dict['wrf_no_div'],
             marker=marker_dict['wrf_no_div'])
    plt.plot(averaged_error['owp_opt']['rmse'].loc[slice(15, None)],
             color=cpal_dict['owp_opt'],
             marker=marker_dict['owp_opt'])
    plt.legend([legend_dict['opt_flow'],
                legend_dict['wrf_no_div'],
                legend_dict['owp_opt']])
    plt.title('RMSE for all days w/ equal SD')
    plt.xlabel('Forecast horizon (min.)')
    plt.ylabel('RMSE (CI)')
    plt.xlim([15, 60])
    plt.savefig(fname=os.path.join(save_directory,
                                   f'rmse.{format}'),
                format=format, dpi=dpi)


def return_original_stats(directory_name):
    runs = ['opt_flow', 'wrf_no_div', 'owp_opt', 'persistence']
    # directory_name = 'third_set_only_cloudy'
    #directory_name = 'third_set'
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


def plot_original_error(*, cpal_dict, marker_dict, legend_dict, format, dpi,
                        averaged_error, save_directory):
    save_directory = os.path.join(save_directory,
                                  'error_plots')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # RMSE
    plt.figure()
    this_to_plot = averaged_error[
        'persistence']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['persistence'],
             marker=marker_dict['persistence'])
    this_to_plot = averaged_error[
        'opt_flow']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['opt_flow'],
             marker=marker_dict['opt_flow'])
    this_to_plot = averaged_error[
        'wrf_no_div']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['wrf_no_div'],
             marker=marker_dict['wrf_no_div'])
    this_to_plot = averaged_error[
        'owp_opt']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['owp_opt'],
             marker=marker_dict['owp_opt'])
    plt.legend([legend_dict['persistence'],
                legend_dict['opt_flow'],
                legend_dict['wrf_no_div'],
                legend_dict['owp_opt']])
    plt.title('RMSE for all days')
    plt.xlabel('Forecast horizon (min.)')
    plt.ylabel('RMSE (CI)')
    plt.xlim([15, 60])
    plt.xticks([15, 30, 45, 60])
    plt.savefig(fname=os.path.join(save_directory,
                                   f'rmse.{format}'),
                format=format, dpi=dpi)


def return_daily_error():
    runs = ['persistence', 'opt_flow', 'wrf_no_div', 'owp_opt']
    base_folder = '/a2/uaren/travis/'
    year = 2014
    month_day = [[4, 2],  [4, 5], [4, 9],
                 [4, 10], [4, 11], [4, 12],
                 [4, 15], [4, 18], [4, 19],
                 [4, 20], [4, 21], [4, 22],
                 [4, 25], [4, 26],

                 [5, 5], [5, 6], [5, 7],
                 [5, 8], [5, 9], [5, 19], [5, 20],
                 [5, 21], [5, 22], [5, 23],
                 [5, 24], [5, 25], [5, 29],
                 [5, 30],

                 [6, 3],  [6, 10], [6, 11],
                 [6, 12], [6, 14], [6, 15],
                 [6, 16], [6, 17],
                 [6, 18], [6, 19], [6, 22]]
    # persistence =
    opt_flow = pd.DataFrame(columns=[15, 30, 45, 60])
    wrf_no_div = opt_flow.copy()
    owp_opt = opt_flow.copy()
    daily_error = {'opt_flow': opt_flow,
                   'wrf_no_div': wrf_no_div,
                   'owp_opt': owp_opt}
    # daily_error = {'persistence': persistence,
    #                'opt_flow': opt_flow,
    #                'wrf_no_div': wrf_no_div,
    #                'owp_opt': owp_opt}
    for this_month_day in month_day:
        month = this_month_day[0]
        day = this_month_day[1]
        this_date = pd.datetime(year, month, day).date()
        for run_name in daily_error.keys():
            results_folder_path = os.path.join(
                base_folder,
                'results',
                f'{year:04}',
                f'{month:02}',
                f'{day:02}',
                run_name)
            results_folder_path = letkf_io.find_latest_run(
                results_folder_path)
            results_folder_path = os.path.join(
                results_folder_path, 'single_day')
            stat_name = 'rmse'
            file_path = os.path.join(
                results_folder_path, f'{stat_name}.h5')
            rmse = pd.read_hdf(file_path, stat_name)
            daily_error[run_name].loc[this_date] = rmse['rmse']


    #     returned = analyse_results.find_error_stats(
    #         year, month, day, runs, base_folder)
    #     for this_stat in returned:
    #         name = this_stat['name']
    #         this_date = this_stat['rmse'].index[0].date()
    #         exec(
    #             name
    #             + '.loc[this_date] '
    #             + '= np.sqrt((this_stat[\'rmse\']**2)'
    #             + '.mean())')
    # persistence = persistence.sort_index()
    # opt_flow = opt_flow.sort_index()
    # wrf_no_div = wrf_no_div.sort_index()
    # owp_opt = owp_opt.sort_index()
    # adict = {'persistence': persistence,
    #          'opt_flow': opt_flow,
    #          'wrf_no_div': wrf_no_div,
    #          'owp_opt': owp_opt}
    return daily_error


def plot_daily_error(*, cpal_dict, marker_dict, legend_dict, format, dpi,
                     daily_error, averaged_error, save_directory):
    save_directory = os.path.join(save_directory,
                                  'daily_error')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    #persistence = daily_error['persistence']
    # persistence = pd.concat(
    #     [persistence,
    #      (averaged_error['persistence']['rmse'].T)[[15, 30, 45, 60]]])
    opt_flow = daily_error['opt_flow']
    xticks = [str(index.month) + ' ' + str(index.day)
              for index in opt_flow.index]
    opt_flow = pd.concat(
        [opt_flow,
         (averaged_error['opt_flow']['rmse'].T)[[15, 30, 45, 60]]])
    wrf_no_div = daily_error['wrf_no_div']
    wrf_no_div = pd.concat(
        [wrf_no_div,
         (averaged_error['wrf_no_div']['rmse'].T)[[15, 30, 45, 60]]])
    owp_opt = daily_error['owp_opt']
    owp_opt = pd.concat(
        [owp_opt,
         (averaged_error['owp_opt']['rmse'].T)[[15, 30, 45, 60]]])
    y_max = np.max([opt_flow.max(),
                    wrf_no_div.max(),
                    owp_opt.max()])

    # y_max = np.max([persistence.max(),
    #                 opt_flow.max(),
    #                 wrf_no_div.max(),
    #                 owp_opt.max()])

    xticks.append('All days')
    xarange = np.arange(len(xticks))
    figsize = plt.figaspect(0.3)
    width = 0.28
    # width = 0.20

    for hor in [15, 30, 45, 60]:
        plt.figure(figsize=figsize)
        # plt.bar(xarange,
        #         persistence[hor], width,
        #         color=cpal_dict['opt_flow'])
        # plt.bar(xarange + width,
        #         opt_flow[hor], width,
        #         color=cpal_dict['opt_flow'])
        # plt.bar(xarange + 2*width,
        #         wrf_no_div[hor], width,
        #         color=cpal_dict['wrf_no_div'])
        # plt.bar(xarange + 3*width,
        #         owp_opt[hor], width,
        #         color=cpal_dict['owp_opt'])

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
        plt.title(f'RMSE for a horizon of {hor} minutes')
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt']])
        plt.xlabel('Date')
        plt.ylabel('RMSE (CI)')
        plt.ylim([0, y_max])
        plt.xlim([xarange[0] - 0.5, xarange[-1] + 2*width + 0.5])
        plt.tight_layout()
        plt.savefig(fname=os.path.join(save_directory,
                                       f'rmse_{hor}.{format}'),
                    format=format, dpi=dpi)


def return_spaghetti_error(*, dates_dict, run_names,
                           base_folder='/a2/uaren/travis'):
    year = 2014
    spaghetti_error = {}
    for day_type, month_day in dates_dict.items():
        month = month_day[0]
        day = month_day[1]
        this_error = {}
        for run_name in run_names:
            if run_name[0] is 'ensemble':
                ensemble_flag = True
                run_name = run_name[1]
            else:
                ensemble_flag = False
            results_folder_path = os.path.join(
                base_folder,
                'results',
                f'{year:04}',
                f'{month:02}',
                f'{day:02}',
                run_name)
            results_folder_path = letkf_io.find_latest_run(
                results_folder_path)
            results_folder_path = os.path.join(
                results_folder_path, 'single_day')
            stat_name = 'rmse'
            if ensemble_flag:
                stat_name = stat_name + '_ens'
                run_name = 'ensemble'
            file_path = os.path.join(
                results_folder_path, f'{stat_name}.h5')
            print(file_path)
            rmse = pd.read_hdf(file_path, stat_name)
            this_error[run_name] = rmse
        spaghetti_error[day_type] = this_error
    return spaghetti_error


# def return_spaghetti_error(*, month, day):
#     year = 2014
#     horizons = [15, 30, 45, 60]
#     base_folder = '/a2/uaren/travis/'
#     truth = xr.open_dataset(
#         f'/a2/uaren/travis/data/{year:04}/{month:02}/{day:02}/data.nc')
#     truth = truth['ci']
#     truth = letkf_io.add_crop_attributes(truth)
#     truth = analyse_results.return_error_domain(truth)

#     truth = truth.load()

#     full_day = letkf_io.return_day(year,
#                                    month,
#                                    day,
#                                    'owp_opt',
#                                    base_folder)

#     full_day = letkf_io.add_crop_attributes(full_day)
#     full_day = analyse_results.return_error_domain(full_day)
#     full_day = full_day['ci'].load()

#     full_day_mean = analyse_results.return_ens_mean(full_day)

#     wrf = letkf_io.return_day(year,
#                               month,
#                               day,
#                               'wrf_no_div',
#                               base_folder)

#     wrf = letkf_io.add_crop_attributes(wrf)
#     wrf = analyse_results.return_error_domain(wrf)
#     wrf = wrf['ci'].load()

#     opt_flow = letkf_io.return_day(year,
#                                    month,
#                                    day,
#                                    'opt_flow',
#                                    base_folder)

#     opt_flow = letkf_io.add_crop_attributes(opt_flow)
#     opt_flow = analyse_results.return_error_domain(opt_flow)
#     opt_flow = opt_flow['ci'].load()

#     full_day.ensemble_number.size

#     ens_num = full_day.ensemble_number.size

#     ensemble_rmse = np.ones([len(horizons), ens_num]) * np.nan

#     mean_rmse = np.ones([len(horizons)]) * np.nan

#     wrf_rmse = np.ones([len(horizons)]) * np.nan

#     opt_flow_rmse = np.ones([len(horizons)]) * np.nan
#     for ii, hor in enumerate(horizons):
#         ensemble_rmse_temp = analyse_results.return_horizon(full_day, int(hor))
#         ensemble_rmse_temp = ensemble_rmse_temp - truth
#         ensemble_rmse[ii] = np.sqrt((ensemble_rmse_temp ** 2).mean(
#             dim=['south_north', 'west_east', 'time'])).values

#         mean_rmse_temp = analyse_results.return_horizon(full_day_mean, int(hor))
#         mean_rmse_temp = mean_rmse_temp - truth
#         mean_rmse[ii] = np.sqrt((mean_rmse_temp ** 2).mean(
#             dim=['south_north', 'west_east', 'time'])).values

#         wrf_rmse_temp = analyse_results.return_horizon(wrf, int(hor))
#         wrf_rmse_temp = wrf_rmse_temp - truth
#         wrf_rmse[ii] = np.sqrt((wrf_rmse_temp ** 2).mean(
#             dim=['south_north', 'west_east', 'time'])).values

#         opt_flow_rmse_temp = analyse_results.return_horizon(opt_flow, int(hor))
#         opt_flow_rmse_temp = opt_flow_rmse_temp - truth
#         opt_flow_rmse[ii] = np.sqrt((opt_flow_rmse_temp ** 2).mean(
#             dim=['south_north', 'west_east', 'time'])).values

#     return_dict = {'horizons': horizons,
#                    'ensemble_rmse': ensemble_rmse,
#                    'mean_rmse': mean_rmse,
#                    'wrf_rmse': wrf_rmse,
#                    'opt_flow_rmse': opt_flow_rmse}
#     return return_dict


def plot_spaghetti(*, cpal_dict, legend_dict, marker_dict,
                   format, dpi, dates_dict,
                   spaghetti_error,
                   save_directory):
    case_study_dict = {'translation': 'Case Study 1',
                       'more_complex': 'Case Study 2',
                       'two_levels': 'Case Study 3'}
    for day_type, month_day in dates_dict.items():
        month = month_day[0]
        day = month_day[1]
        this_title = case_study_dict[day_type]
        this_save_directory = os.path.join(save_directory,
                                           day_type)
        if not os.path.exists(this_save_directory):
            os.makedirs(this_save_directory)
        # save_directory = ('/home/travis/python_code/'
        #                   'letkf_forecasting/figures/'
        #                   + day_type)
        this_error = spaghetti_error[day_type]

        ensemble_rmse = this_error['ensemble']
        mean_rmse = this_error['owp_opt']
        wrf_rmse = this_error['wrf_no_div']
        opt_flow_rmse = this_error['opt_flow']
        persistence_rmse = this_error['persistence']
        horizons = mean_rmse.index.values

        # RMSE
        plt.figure()
        plt.plot(horizons, persistence_rmse,
                 marker=marker_dict['persistence'],
                 color=cpal_dict['persistence'])
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
        plt.xticks(horizons)
        legend = plt.legend([legend_dict['persistence'],
                    legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt'],
                    legend_dict['ens_member']])
        for handle in legend.legendHandles:
            handle.set_alpha(1)
        # plt.legend([legend_dict['persistence'],
        #             legend_dict['opt_flow'],
        #             legend_dict['wrf_no_div'],
        #             legend_dict['owp_opt'],
        #             legend_dict['ens_member']])
        plt.xlabel('Forecast horizon (min.)')
        plt.ylabel('RMSE (CI)')
        plt.title(f'RMSE for {this_title}: 2014/{month}/{day}')
        plt.savefig(fname=os.path.join(this_save_directory,
                                       f'rmse.{format}'),
                    format=format, dpi=dpi)


def main():
    format = 'png'
    dpi = 400
    cpal = sns.color_palette('deep')
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5,
                    rc={'lines.linewidth': 1.0,
                        'lines.markersize': 7})
    cpal_dict = {'opt_flow': cpal[0],
                 'wrf_no_div': cpal[1],
                 'owp_opt': cpal[2],
                 'persistence': cpal[3],
                 'ens_member': cpal[2]}
    legend_dict = {'opt_flow': 'Opt. Flow',
                   'wrf_no_div': 'NWP Winds',
                   'owp_opt': 'ANOC',
                   'persistence': 'Persistence',
                   'ens_member': 'ANOC ens. members'}
    marker_dict = {'opt_flow': 'o',
                   'wrf_no_div': '^',
                   'owp_opt': 'd',
                   'persistence': 's',
                   'ens_member': '*'}

    directory_name = 'third_set'
    save_directory = ('/home/travis/python_code/'
                      'letkf_forecasting/figures/')

    # smoothing_error = return_smoothing_data(
    #     directory_name=directory_name)
    averaged_error = return_original_stats(
        directory_name=directory_name)
    daily_error = return_daily_error()

    # # plot smoothed
    # plot_smoothing_data(cpal_dict=cpal_dict,
    #                     legend_dict=legend_dict,
    #                     marker_dict=marker_dict,
    #                     format=format,
    #                     dpi=dpi,
    #                     smoothing_error=smoothing_error,
    #                     averaged_error=averaged_error,
    #                     save_directory=save_directory)
    # plt.close('all')

    # plot original error plots
    plot_original_error(cpal_dict=cpal_dict,
                        legend_dict=legend_dict,
                        marker_dict=marker_dict,
                        format=format,
                        dpi=dpi,
                        averaged_error=averaged_error,
                        save_directory=save_directory)
    plt.close('all')

    # # plot error by date plots
    # plot_daily_error(cpal_dict=cpal_dict,
    #                  legend_dict=legend_dict,
    #                  marker_dict=marker_dict,
    #                  format=format,
    #                  dpi=dpi,
    #                  daily_error=daily_error,
    #                  averaged_error=averaged_error,
    #                  save_directory=save_directory)
    # plt.close('all')

    # plot spaghetti data
    dates_dict = {
        'translation': (4, 15),
        'more_complex': (5, 29),
        'two_levels': (4, 26)
    }
    run_names = ['persistence',
                 'opt_flow',
                 'wrf_no_div',
                 'owp_opt',
                 ['ensemble', 'owp_opt']]
    spaghetti_error = return_spaghetti_error(
        dates_dict=dates_dict,
        run_names=run_names)
    plot_spaghetti(cpal_dict=cpal_dict,
                   legend_dict=legend_dict,
                   marker_dict=marker_dict,
                   format=format,
                   dpi=dpi,
                   dates_dict=dates_dict,
                   spaghetti_error=spaghetti_error,
                   save_directory=save_directory)


    plt.close('all')


if __name__ == '__main__':
    main()
