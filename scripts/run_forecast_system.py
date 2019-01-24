import argparse
import yaml
import os
import logging
import multiprocessing
import time as time_py
import letkf_forecasting.letkf_forecasting as lf
import letkf_forecasting.letkf_io as letkf_io
from letkf_forecasting import __version__


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

    with open(args.file_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    if args.year is not None:
        cfg['date']['year'] = args.year
    if args.month is not None:
        cfg['date']['month'] = args.month
    if args.day is not None:
        cfg['date']['day'] = args.day
    # Create path to save results
    if 'analysis_fore' in cfg['flags']:
        if cfg['flags']['analysis_fore']:
            year = cfg['date']['year']
            month = cfg['date']['month']
            day = cfg['date']['day']
            this_run = cfg['io']['run_name']
            results_file_path = (
                f'{args.home}/results/{year:04}/{month:02}/{day:02}/{this_run}')
            results_file_path = letkf_io.find_latest_run(results_file_path)
            yaml_file_path = os.path.join(
                results_file_path,
                'config_' + cfg['io']['run_name']
                + '_anlys_fore' + '.yml')
            log_path = os.path.join(results_file_path,
                                    'forecast_anlys_fore.log')
        else:
            results_file_path = letkf_io.create_folder(
                os.path.join(args.home, 'results'), cfg['date']['year'],
                cfg['date']['month'], cfg['date']['day'],
                cfg['io']['run_name'])
            yaml_file_path = os.path.join(
                results_file_path, 'config_' + cfg['io']['run_name'] + '.yml')
            log_path = os.path.join(results_file_path, 'forecast.log')
    else:
        results_file_path = letkf_io.create_folder(
            os.path.join(args.home, 'results'), cfg['date']['year'],
            cfg['date']['month'], cfg['date']['day'],
            cfg['io']['run_name'])
        yaml_file_path = os.path.join(
            results_file_path, 'config_' + cfg['io']['run_name'] + '.yml')
        log_path = os.path.join(results_file_path, 'forecast.log')
    cfg['version'] = __version__
    with open(yaml_file_path, 'w') as ymlfile:
        yaml.dump(cfg, ymlfile, default_flow_style=False)
    data_file_path = cfg['io']['data_file_path'].format(
        home=args.home, year=cfg['date']['year'],
        month=cfg['date']['month'], day=cfg['date']['day'])
    time0 = time_py.time()

    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        filemode='w', level=logging.DEBUG)
    logging.info('Started')

    # use less memory than fork
    try:
        multiprocessing.set_start_method('forkserver')
    except RuntimeError:
        pass
    workers = (cfg['advect_params']['workers']
               if 'workers' in cfg['advect_params']
               else args.workers)

    lf.forecast_system(
        data_file_path=data_file_path, results_file_path=results_file_path,
        date=cfg['date'], io=cfg['io'], flags=cfg['flags'],
        advect_params=cfg['advect_params'],
        ens_params=cfg['ens_params'], pert_params=cfg['pert_params'],
        sat2sat=cfg['sat2sat'], sat2wind=cfg['sat2wind'], wrf=cfg['wrf'],
        opt_flow=cfg['opt_flow'], workers=workers)
    logging.info('Ended')
    time1 = time_py.time()
    print('It took: ' + str((time1 - time0)/60))
    logging.info('It took: ' + str((time1 - time0)/60))


if __name__ == '__main__':
    main()
