import os
import re
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
        'one_of_rmse': one_of_rmse,
        'one_wrf_rmse': one_wrf_rmse,
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
                legend_dict['owp_opt']],
               ncol=2)
    plt.title('RMSE for all days w/ equal SD')
    plt.xlabel('Forecast horizon (min.)')
    plt.ylabel('RMSE (CI)')
    plt.xlim([15, 60])
    plt.savefig(fname=os.path.join(save_directory,
                                   f'rmse.{format}'),
                format=format, dpi=dpi)


def return_original_stats(directory_name):
    runs = ['opt_flow', 'wrf_no_div',
            'owp_opt', 'owp_opt_anly_fore', 'persistence']
    runs = ['owp_opt', 'owp_opt_anly_fore', 'persistence', 'opt_flow',
            'wrf_no_div', 'wrf_mean', 'radiosonde']
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


def plot_original_error(*, cpal_dict, marker_dict,
                        legend_dict, format, dpi,
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
             marker=marker_dict['persistence'],
             )
    this_to_plot = averaged_error[
        'opt_flow']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['opt_flow'],
             marker=marker_dict['opt_flow'],
             )
    this_to_plot = averaged_error[
        'wrf_no_div']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['wrf_no_div'],
             marker=marker_dict['wrf_no_div'],
             )
    this_to_plot = averaged_error[
        'owp_opt']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['owp_opt'],
             marker=marker_dict['owp_opt'],
             )
    this_to_plot = averaged_error[
        'owp_opt_anly_fore']['rmse'].loc[slice(15, None)]
    plt.plot(this_to_plot,
             color=cpal_dict['anly_fore'],
             marker=marker_dict['anly_fore'],
             )
    plt.legend([legend_dict['persistence'],
                legend_dict['opt_flow'],
                legend_dict['wrf_no_div'],
                legend_dict['owp_opt'],
                legend_dict['anly_fore']],
               ncol=2)
    plt.title('RMSE for all days')
    plt.xlabel('Forecast horizon (min.)')
    plt.ylabel('RMSE (CI)')
    plt.xlim([15, 60])
    plt.xticks([15, 30, 45, 60])
    plt.ylim([None, None])
    plt.savefig(fname=os.path.join(save_directory,
                                   f'rmse.{format}'),
                format=format, dpi=dpi)


def return_daily_error():
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
    opt_flow = pd.DataFrame(columns=[15, 30, 45, 60])
    wrf_no_div = opt_flow.copy()
    owp_opt = opt_flow.copy()
    anly_fore = opt_flow.copy()
    daily_error = {'opt_flow': opt_flow,
                   'wrf_no_div': wrf_no_div,
                   'owp_opt': owp_opt,
                   'anly_fore': anly_fore}
    for this_month_day in month_day:
        month = this_month_day[0]
        day = this_month_day[1]
        this_date = pd.datetime(year, month, day).date()
        for run_name in daily_error.keys():
            anly_fore_flag = False
            if run_name == 'anly_fore':
                run_name = 'owp_opt'
                anly_fore_flag = True
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
            if anly_fore_flag:
                stat_name = stat_name + '_anly_fore'
                run_name = 'anly_fore'
            file_path = os.path.join(
                results_folder_path, f'{stat_name}.h5')
            rmse = pd.read_hdf(file_path, stat_name)
            daily_error[run_name].loc[this_date] = rmse['rmse']
    return daily_error


def plot_daily_error(*, cpal_dict, marker_dict, legend_dict,
                     format, dpi,
                     daily_error, averaged_error, save_directory):
    save_directory = os.path.join(save_directory,
                                  'daily_error')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
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
    anly_fore = daily_error['anly_fore']
    anly_fore = pd.concat(
        [anly_fore,
         (averaged_error['owp_opt_anly_fore']['rmse'].T)[[15, 30, 45, 60]]])

    y_max = np.max([opt_flow.max(),
                    wrf_no_div.max(),
                    owp_opt.max()])
    y_min = 0

    xticks.append('All days')
    xarange = np.arange(len(xticks))
    figsize = plt.figaspect(0.3)
    width = 0.20

    for hor in [15, 30, 45, 60]:
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
        plt.bar(xarange + 3*width,
                anly_fore[hor], width,
                color=cpal_dict['anly_fore'])
        plt.xticks(xarange + width, xticks, rotation=90)
        plt.title(f'RMSE for a horizon of {hor} minutes')
        plt.legend([legend_dict['opt_flow'],
                    legend_dict['wrf_no_div'],
                    legend_dict['owp_opt'],
                    legend_dict['anly_fore']],
                   ncol=2)

        plt.xlabel('Date')
        plt.ylabel('RMSE (CI)')
        plt.ylim([y_min, y_max])
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
            ensemble_flag = False
            analysis_fore_flag = False
            if run_name[0] is 'ensemble':
                ensemble_flag = True
                run_name = run_name[1]
            elif run_name[0] is 'anly_fore':
                analysis_fore_flag = True
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
            elif analysis_fore_flag:
                stat_name = stat_name + '_anly_fore'
                run_name = 'owp_opt_anly_fore'
            file_path = os.path.join(
                results_folder_path, f'{stat_name}.h5')
            print(file_path)
            rmse = pd.read_hdf(file_path, stat_name)

            stat_name = 'bias'
            if ensemble_flag:
                stat_name = stat_name + '_ens'
                run_name = 'ensemble'
            elif analysis_fore_flag:
                stat_name = stat_name + '_anly_fore'
                run_name = 'owp_opt_anly_fore'
            file_path = os.path.join(
                results_folder_path, f'{stat_name}.h5')
            print(file_path)
            bias = pd.read_hdf(file_path, stat_name)

            stat_name = 'correlation'
            if ensemble_flag:
                stat_name = stat_name + '_ens'
                run_name = 'ensemble'
            elif analysis_fore_flag:
                stat_name = stat_name + '_anly_fore'
                run_name = 'owp_opt_anly_fore'
            file_path = os.path.join(
                results_folder_path, f'{stat_name}.h5')
            print(file_path)
            correlation = pd.read_hdf(file_path, stat_name)

            all_stats = {'rmse': rmse,
                         'bias': bias,
                         'correlation': correlation}
            this_error[run_name] = all_stats
        spaghetti_error[day_type] = this_error
    return spaghetti_error


def plot_spaghetti(*, cpal_dict, legend_dict, loc_dict, marker_dict,
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
        this_error = spaghetti_error[day_type]
        analy_fore_rmse = this_error['owp_opt_anly_fore']['rmse']
        ensemble_rmse = this_error['ensemble']['rmse']
        mean_rmse = this_error['owp_opt']['rmse']
        wrf_rmse = this_error['wrf_no_div']['rmse']
        opt_flow_rmse = this_error['opt_flow']['rmse']
        persistence_rmse = this_error['persistence']['rmse']
        horizons = mean_rmse.index.values

        # RMSE
        plt.figure()
        plt.plot(horizons, persistence_rmse,
                 marker=marker_dict['persistence'],
                 color=cpal_dict['persistence'],
                 )
        plt.plot(horizons, opt_flow_rmse,
                 marker=marker_dict['opt_flow'],
                 color=cpal_dict['opt_flow'],
                 )
        plt.plot(horizons, wrf_rmse,
                 marker=marker_dict['wrf_no_div'],
                 color=cpal_dict['wrf_no_div'],
                 )
        plt.plot(horizons, mean_rmse,
                 marker=marker_dict['owp_opt'],
                 color=cpal_dict['owp_opt'],
                 )
        plt.plot(horizons, analy_fore_rmse,
                 marker=marker_dict['anly_fore'],
                 color=cpal_dict['anly_fore'],
                 )
        plt.plot(horizons, ensemble_rmse, alpha=0.5, linestyle=':',
                 color=cpal_dict['ens_member'])
        plt.xlim([min(horizons), max(horizons)])
        plt.xticks(horizons)
        legend = plt.legend([legend_dict['persistence'],
                             legend_dict['opt_flow'],
                             legend_dict['wrf_no_div'],
                             legend_dict['owp_opt'],
                             legend_dict['anly_fore'],
                             legend_dict['ens_member']],
                            ncol=2,
                            loc=loc_dict[day_type])
        y_min = None
        y_max = None
        if day_type == 'translation':
            y_max = 0.1
        if day_type == 'more_complex':
            y_max = 0.29
        plt.ylim([y_min, y_max])
        for handle in legend.legendHandles:
            handle.set_alpha(1)
        plt.xlabel('Forecast horizon (min.)')
        plt.ylabel('RMSE (CI)')
        plt.title(f'RMSE for {this_title}: 2014/{month}/{day}')
        plt.savefig(fname=os.path.join(this_save_directory,
                                       f'rmse.{format}'),
                    format=format, dpi=dpi)


def table_original_error(*, save_directory,
                         legend_dict,
                         averaged_error):
    file_name = 'all_days'
    decimals = 2
    horizons = [15, 30, 45, 60]
    runs = ['owp_opt', 'owp_opt_anly_fore',
            'opt_flow', 'wrf_no_div', 'wrf_mean', 'radiosonde', 'persistence']
    rmse = pd.DataFrame(index=horizons, columns=runs)
    rmse.index.name = 'Horizon'
    correlation = rmse.copy()
    bias = rmse.copy()
    truth_sd = rmse.copy()
    for run_name in runs:
        stat_name = 'rmse'
        rmse[run_name] = averaged_error[run_name][stat_name].loc[horizons]

        stat_name = 'bias'
        bias[run_name] = averaged_error[run_name][stat_name].loc[horizons]

        stat_name = 'correlation'
        correlation[run_name] = (
            averaged_error[run_name][stat_name].loc[horizons])

        stat_name = 'truth_sd'
        truth_sd[run_name] = (
            averaged_error[run_name][stat_name].loc[horizons])

    peices = [rmse, correlation, bias]
    combined = pd.concat(peices, axis=0,
                         keys=['RMSE', 'Corr.', 'Bias'])
    combined = combined.rename(columns=legend_dict)

    def format_table(text, header_num=5, footer_num=2):
        text = text.split(' ')
        text = list(filter(is_empty, text))
        text = ' '.join(text)
        split_text = text.split('\n')
        split_titles2 = split_text[2]
        removed = split_titles2[-2:]
        split_titles2 = split_titles2[:-2]
        split_titles2 = split_titles2.split('&')
        for count, this in enumerate(split_titles2):
            if len(this) > 2:
                this = this[0] + '{' + this[1:-1] + '}' + this[-1]
                split_titles2[count] = this
        split_text[2] = '&'.join(split_titles2) + removed
        for line_num, line in enumerate(split_text[header_num:-footer_num - 1]):
            split_line = line.split(' ')
            if split_line[0] == 'Corr.':
                Corr = True
            elif split_line[0] != '':
                Corr = False
            num_slice = slice(4, None, 2)
            numbers_str = split_line[num_slice]
            numbers = np.array(
                split_line[num_slice],
                dtype='float')
            if Corr:
                best_num = numbers.max()
            else:
                best_num = numbers[np.abs(numbers).argmin()]
            argmins = np.where(numbers == best_num)[0]
            for argmin in argmins:
                numbers_str[argmin] = '\\B ' + numbers_str[argmin]
            split_line[num_slice] = numbers_str
            split_text[header_num + line_num] = ' '.join(split_line)
        return '\n'.join(split_text)
    column_format = 'll' + 'S[table-format=-1.3]' * len(runs)
    text = combined.round(decimals=decimals).to_latex(column_format=column_format)
    text2 = format_table(text)
    text2 = re.sub('\\\\textasciitilde', '~', text2, count=5)
    this_file = os.path.join(save_directory, f'{file_name}_results.tex')
    with open(this_file, 'w') as file:
        file.write(text2)

    # Skill Score table
    def format_table_SS(text, header_num=4, footer_num=2):
        text = text.split(' ')
        text = list(filter(is_empty, text))
        text = ' '.join(text)
        split_text = text.split('\n')
        hor = split_text[3]
        hor = hor.split('&')[0]
        split_text.pop(3)
        split_titles2 = split_text[2]
        removed = split_titles2[-2:]
        split_titles2 = split_titles2[:-2]
        split_titles2 = split_titles2.split('&')
        split_titles2[0] = hor
        for count, this in enumerate(split_titles2):
            if len(this) > 2:
                if this[0] == ' ':
                    this = this[1:]
                if this[-1] == ' ':
                    this = this[:-1]
                this = ' {' + this + '} '
                split_titles2[count] = this
        split_text[2] = '&'.join(split_titles2) + removed
        for line_num, line in enumerate(split_text[header_num:-footer_num - 1]):
            split_line = line.split(' ')
            num_slice = slice(2, None, 2)
            numbers_str = split_line[num_slice]
            numbers = np.array(
                split_line[num_slice],
                dtype='float')
            best_num = numbers.max()
            argmins = np.where(numbers == best_num)[0]
            for argmin in argmins:
                numbers_str[argmin] = '\\B ' + numbers_str[argmin]
            split_line[num_slice] = numbers_str
            split_text[header_num + line_num] = ' '.join(split_line)
        return '\n'.join(split_text)
    SS_per = (1 - rmse[
        ['owp_opt',
         'owp_opt_anly_fore',
         'opt_flow',
         'wrf_no_div']].div(
        rmse['persistence'], axis='index'))
    SS_per = SS_per.rename(columns=legend_dict)
    column_format = 'l' + 'S[table-format=1.3]' * 4
    text = SS_per.round(
        decimals=decimals).to_latex(
        column_format=column_format)
    text2 = format_table_SS(text)
    text2 = re.sub('\\\\textasciitilde', '~', text2, count=5)
    this_file = os.path.join(save_directory, f'{file_name}_SS.tex')
    with open(this_file, 'w') as file:
        file.write(text2)


def is_empty(str):
    return str != ''


def table_case_studies(*, save_directory, legend_dict,
                       spaghetti_error, dates_dict):
    decimals = 2
    horizons = [15, 30, 45, 60]
    runs = ['owp_opt', 'owp_opt_anly_fore',
            'opt_flow', 'wrf_no_div', 'wrf_mean', 'radiosonde', 'persistence']

    def format_table(text, header_num=5, footer_num=2):
        text = text.split(' ')
        text = list(filter(is_empty, text))
        text = ' '.join(text)
        split_text = text.split('\n')
        split_titles2 = split_text[2]
        removed = split_titles2[-2:]
        split_titles2 = split_titles2[:-2]
        split_titles2 = split_titles2.split('&')
        for count, this in enumerate(split_titles2):
            if len(this) > 2:
                this = this[0] + '{' + this[1:-1] + '}' + this[-1]
                split_titles2[count] = this
        split_text[2] = '&'.join(split_titles2) + removed
        for line_num, line in enumerate(split_text[header_num:-footer_num - 1]):
            split_line = line.split(' ')
            if split_line[0] == 'Corr.':
                Corr = True
            elif split_line[0] != '':
                Corr = False
            num_slice = slice(4, None, 2)
            numbers_str = split_line[num_slice]
            numbers = np.array(
                split_line[num_slice],
                dtype='float')
            if Corr:
                best_num = numbers.max()
            else:
                best_num = numbers[np.abs(numbers).argmin()]
            argmins = np.where(numbers == best_num)[0]
            for argmin in argmins:
                numbers_str[argmin] = '\\B ' + numbers_str[argmin]
            split_line[num_slice] = numbers_str
            split_text[header_num + line_num] = ' '.join(split_line)
        return '\n'.join(split_text)
    for day_type, month_day in dates_dict.items():
        this_error = spaghetti_error[day_type]
        rmse = pd.DataFrame(index=horizons, columns=runs)
        rmse.index.name = 'Horizon'
        correlation = rmse.copy()
        bias = rmse.copy()
        for run_name in runs:
            stat_name = 'rmse'
            rmse[run_name] = this_error[run_name][stat_name].loc[horizons]

            stat_name = 'bias'
            bias[run_name] = this_error[run_name][stat_name].loc[horizons]

            stat_name = 'correlation'
            correlation[run_name] = (
                this_error[run_name][stat_name].loc[horizons])

        peices = [rmse, correlation, bias]
        combined = pd.concat(peices, axis=0,
                             keys=['RMSE', 'Corr.', 'Bias'])
        combined = combined.rename(columns=legend_dict)
        column_format = 'll' + 'S[table-format=-1.3]' * len(runs)
        text = combined.round(decimals=decimals).to_latex(
            column_format=column_format)
        text2 = format_table(text)
        text2 = re.sub('\\\\textasciitilde', '~', text2, count=5)
        this_file = os.path.join(save_directory, f'{day_type}_results.tex')
        with open(this_file, 'w') as file:
            file.write(text2)


def main():
    format = 'png'
    dpi = 400
    cpal = sns.color_palette('colorblind')
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5,
                    rc={'lines.linewidth': 1.0,
                        'lines.markersize': 11})
    cpal_dict = {'opt_flow': cpal[0],
                 'wrf_no_div': cpal[1],
                 'owp_opt': cpal[2],
                 'anly_fore': cpal[5],
                 'persistence': 'gray',
                 'ens_member': cpal[2]}
    legend_dict = {'opt_flow': 'Opt. Flow',
                   'wrf_no_div': 'NWP Winds',
                   'owp_opt': 'ANOC Ens. Mean',
                   'anly_fore': 'ANOC Control',
                   'persistence': 'Persistence',
                   'ens_member': 'ANOC Ens. Members'}
    marker_dict = {'opt_flow': 'o',
                   'wrf_no_div': '^',
                   'owp_opt': 'd',
                   'anly_fore': 'X',
                   'persistence': 's',
                   'ens_member': '1'}

    directory_name = 'third_set'
    figure_directory = ('/home/travis/python_code/'
                        'letkf_forecasting/figures/')

    averaged_error = return_original_stats(
        directory_name=directory_name)
    daily_error = return_daily_error()

    # plot error by date plots
    plot_daily_error(cpal_dict=cpal_dict,
                     legend_dict=legend_dict,
                     marker_dict=marker_dict,

                     format=format,
                     dpi=dpi,
                     daily_error=daily_error,
                     averaged_error=averaged_error,
                     save_directory=figure_directory)
    plt.close('all')


if __name__ == '__main__':
    main()
