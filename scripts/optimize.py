import argparse
import yaml
import os
import logging
import importlib
import time as time_py
import letkf_forecasting.letkf_forecasting as lf
import letkf_forecasting.analyse_results as analyse_results
from letkf_forecasting import __version__
import numpy as np


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
    args = parser.parse_args()

    # wrf_length_array = np.array([4, 10])
    wrf_length_array = np.array([10])
    of_length_array = np.array([4, 20])
    wrf_inflation_array = np.array([1.5, 5])
    of_inflation_array = np.array([1.5, 5])
    sig_pw = 0
    l_pw = 0
    for l_w in wrf_length_array:
        for i_w in wrf_inflation_array:
            for l_of in of_length_array:
                for i_of in of_inflation_array:
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
                    opt_str = (f'_{l_w:03}_{i_w_str}_{l_of:03}_'
                               + f'{i_of_str}_{l_pw:03}_{sig_pw_str}')
                    cfg['io']['run_name'] = (
                        cfg['io']['run_name'] + opt_str)
                    print(cfg['io']['run_name'])
                    year = cfg['date']['year']
                    month = cfg['date']['month']
                    day = cfg['date']['day']
                    results_folder_path = os.path.join(
                        '/a2/uaren/travis/results_opt/',
                        f'{year:04}',
                        f'{month:02}',
                        f'{day:02}',
                        cfg['io']['run_name'])
                    if not os.path.exists(results_folder_path):
                        os.makedirs(results_folder_path)
                    # # Create path to save results
                    # results_file_path = letkf_io.create_folder(
                    #     cfg['date']['year'],
                    #     cfg['date']['month'],
                    #     cfg['date']['day'],
                    #     cfg['io']['run_name'])
                    yaml_file_path = os.path.join(
                        results_folder_path,
                        'config_' + cfg['io']['run_name'] + '.yml')
                    cfg['version'] = __version__
                    with open(yaml_file_path, 'w') as ymlfile:
                        yaml.dump(cfg, ymlfile, default_flow_style=False)
                    home = '/a2/uaren/travis/'
                    data_file_path = cfg['io']['data_file_path'].format(
                        home=home, year=cfg['date']['year'],
                        month=cfg['date']['month'], day=cfg['date']['day'])
                    time0 = time_py.time()

                    log_path = os.path.join(results_folder_path,
                                            'forecast.log')
                    logging.shutdown()
                    importlib.reload(logging)
                    logging.basicConfig(
                        filename=log_path,
                        filemode='w', level=logging.DEBUG)
                    logging.info('Started')
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
                            opt_flow=cfg['opt_flow'])
                    except Exception:
                        logging.exception('forecast_system failed')
                    logging.info('Ended')
                    time1 = time_py.time()
                    print('It took: ' + str((time1 - time0)/60))
                    logging.info('It took: ' + str((time1 - time0)/60))


if __name__ == '__main__':
    main()
