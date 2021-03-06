import argparse
import yaml
import os
import logging
import multiprocessing
import importlib
import time as time_py
import letkf_forecasting.letkf_forecasting as lf
import letkf_forecasting.analyse_results as analyse_results
from letkf_forecasting import __version__
import numpy as np
import pandas as pd


def set_up_configuration(args, l_w, i_w, l_of, i_of, sig_pw, l_pw):
    with open(args.file_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    if args.year is not None:
        cfg['date']['year'] = args.year
    if args.month is not None:
        cfg['date']['month'] = args.month
    if args.day is not None:
        cfg['date']['day'] = args.day
    i_w_str = str(i_w).replace('.', 'p')
    i_of_str = str(i_of).replace('.', 'p')
    sig_pw_str = str(sig_pw).replace('.', 'p')
    cfg['wrf']['loc'] = l_w
    cfg['wrf']['infl'] = i_w
    cfg['opt_flow']['loc'] = l_of
    cfg['opt_flow']['infl'] = i_of
    cfg['pert_params']['Lx_wind'] = l_pw
    cfg['pert_params']['Ly_wind'] = l_pw
    cfg['pert_params']['pert_sigma_wind'] = sig_pw
    opt_str = (f'_{l_w:03}_{i_w_str}_{l_of:03}_'
               + f'{i_of_str}_{l_pw:03}_{sig_pw_str}')
    cfg['io']['run_name'] = (
        cfg['io']['run_name'] + opt_str)
    cfg['version'] = __version__
    return cfg


def set_up_logging(results_folder_path):
    log_path = os.path.join(results_folder_path,
                            'forecast.log')
    logging.shutdown()
    importlib.reload(logging)
    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        filemode='w', level=logging.DEBUG)


def error_analysis(cfg, results_folder_path, home, optimize_folder):
    year = cfg['date']['year']
    month = cfg['date']['month']
    day = cfg['date']['day']
    run = cfg['io']['run_name']
    returned = analyse_results.error_stats(
        year, month, day, [run], home, optimize_folder)
    for k, v in returned[0].items():
        conditional = (isinstance(v, type(pd.Series()))
                       or isinstance(v, type(pd.DataFrame())))
        if conditional:
            file_path = os.path.join(
                results_folder_path,
                f'{k}.h5')
            v.to_hdf(file_path, k)


def main():
    parser = argparse.ArgumentParser(
        description='Run forecast_system using provided configuration file.')
    parser.add_argument('file_path', type=str,
                        help='The full path to the configuration file.')
    parser.add_argument('-y', '--year', type=int,
                        help='The year you wish to run.')
    parser.add_argument('-m', '--month', type=int,
                        help='The month you wish to run.')
    parser.add_argument('-d', '--day', type=int,
                        help='The day you wish to run.')
    parser.add_argument('--home', help='base directory for data and results',
                        default='/a2/uaren/travis/')
    parser.add_argument('--workers', help='Number of advection workers',
                        default=20, type=int)
    args = parser.parse_args()

    wrf_length_array = np.array([1])
    of_length_array = np.array([20])
    wrf_inflation_array = np.array([1.5, 4])
    of_inflation_array = np.array([1.5, 2])
    sig_pw = .25
    l_pw = 50
    for l_w in wrf_length_array:
        for i_w in wrf_inflation_array:
            for l_of in of_length_array:
                for i_of in of_inflation_array:
                    cfg = set_up_configuration(args, l_w, i_w,
                                               l_of, i_of, sig_pw, l_pw)
                    print(cfg['io']['run_name'])
                    year = cfg['date']['year']
                    month = cfg['date']['month']
                    day = cfg['date']['day']
                    results_folder_path = os.path.join(
                        f'{args.home}/results_opt/',
                        f'{year:04}',
                        f'{month:02}',
                        f'{day:02}',
                        cfg['io']['run_name'])
                    if not os.path.exists(results_folder_path):
                        os.makedirs(results_folder_path)
                    yaml_file_path = os.path.join(
                        results_folder_path,
                        'config_' + cfg['io']['run_name'] + '.yml')

                    with open(yaml_file_path, 'w') as ymlfile:
                        yaml.dump(cfg, ymlfile, default_flow_style=False)
                    data_file_path = cfg['io']['data_file_path'].format(
                        home=args.home, year=cfg['date']['year'],
                        month=cfg['date']['month'], day=cfg['date']['day'])
                    time0 = time_py.time()
                    set_up_logging(results_folder_path)
                    logging.info('Started')
                    # use less memory than fork
                    try:
                        multiprocessing.set_start_method('forkserver')
                    except RuntimeError:
                        pass
                    workers = (cfg['advect_params']['workers']
                               if 'workers' in cfg['advect_params']
                               else args.workers)
                    try:
                        lf.forecast_system(
                            data_file_path=data_file_path,
                            results_file_path=results_folder_path,
                            date=cfg['date'], io=cfg['io'], flags=cfg['flags'],
                            advect_params=cfg['advect_params'],
                            ens_params=cfg['ens_params'],
                            pert_params=cfg['pert_params'],
                            sat2sat=cfg['sat2sat'],
                            sat2wind=cfg['sat2wind'],
                            wrf=cfg['wrf'],
                            opt_flow=cfg['opt_flow'],
                            workers=workers)
                        logging.info('forecast_system ended')
                    except Exception:
                        logging.exception('forecast_system failed')
                    logging.info('begin error analysis')
                    try:
                        error_analysis(cfg, results_folder_path,
                                       args.home, 'results_opt')
                        logging.info('error analysis ended')
                    except Exception:
                        logging.exception('error_analysis failed')
                    time1 = time_py.time()
                    print('It took: ' + str((time1 - time0)/60))
                    logging.info('It took: ' + str((time1 - time0)/60))


if __name__ == '__main__':
    main()
