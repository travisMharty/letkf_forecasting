import argparse
import yaml
import os
import logging
# import multiprocessing
import importlib
import time as time_py
# import letkf_forecasting.letkf_forecasting as lf
import letkf_forecasting.analyse_results as analyse_results
import letkf_forecasting.letkf_io as letkf_io
import pandas as pd


def set_up_configuration(args):
    with open(args.file_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    if args.year is not None:
        cfg['date']['year'] = args.year
    if args.month is not None:
        cfg['date']['month'] = args.month
    if args.day is not None:
        cfg['date']['day'] = args.day
    return cfg


def set_up_logging(results_folder_path):
    log_path = os.path.join(results_folder_path,
                            'forecast.log')
    logging.shutdown()
    importlib.reload(logging)
    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        filemode='a', level=logging.DEBUG)


def error_analysis(cfg, results_folder_path, home):
    year = cfg['date']['year']
    month = cfg['date']['month']
    day = cfg['date']['day']
    run = cfg['io']['run_name']
    returned = analyse_results.error_stats(
        year, month, day, [run], home)
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
    cfg = set_up_configuration(args)
    year = cfg['date']['year']
    month = cfg['date']['month']
    day = cfg['date']['day']
    print(cfg['io']['run_name'], f': {year}, {month}, {day}')
    results_folder_path = os.path.join(
        args.home,
        'results',
        f'{year:04}',
        f'{month:02}',
        f'{day:02}',
        cfg['io']['run_name'])
    results_folder_path = letkf_io.find_latest_run(results_folder_path)
    time0 = time_py.time()
    set_up_logging(results_folder_path)
    logging.info('begin error analysis')
    try:
        error_analysis(cfg, results_folder_path,
                       args.home)
        logging.inlfo('error analysis ended')
    except Exception:
        logging.exception('error_analysis failed')
    time1 = time_py.time()
    print('It took: ' + str((time1 - time0)/60))
    logging.info('It took: ' + str((time1 - time0)/60))


if __name__ == '__main__':
    main()
