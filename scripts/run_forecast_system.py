import argparse
import yaml
import os
import logging
import time as time_py
import letkf_forecasting.letkf_forecasting as lf
import letkf_forecasting.letkf_io as letkf_io


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

    with open(args.file_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    date = cfg['date']
    if args.year is not None:
        date['year'] = args.year
    if args.month is not None:
        date['month'] = args.month
    if args.day is not None:
        date['day'] = args.day
    # Create path to save results
    results_file_path = letkf_io.create_folder(date['year'],
                                               date['month'],
                                               date['day'],
                                               cfg['io']['run_name'])
    home = '/a2/uaren/travis/'
    data_file_path = cfg['io']['data_file_path'].format(
        home=home, year=date['year'],
        month=date['month'], day=date['day'])
    time0 = time_py.time()

    log_path = os.path.join(results_file_path, 'forecast.log')
    logging.basicConfig(
        filename=log_path,
        filemode='w', level=logging.DEBUG)
    logging.info('Started')

    lf.forecast_system(
        data_file_path=data_file_path, results_file_path=results_file_path,
        date=cfg['date'], io=cfg['io'], flags=cfg['flags'],
        advect_params=cfg['advect_params'],
        ens_params=cfg['ens_params'], pert_params=cfg['pert_params'],
        sat2sat=cfg['sat2sat'], sat2wind=cfg['sat2wind'], wrf=cfg['wrf'],
        opt_flow=cfg['opt_flow'])
    logging.info('Ended')
    time1 = time_py.time()
    print('It took: ' + str((time1 - time0)/60))
    logging.info('It took: ' + str((time1 - time0)/60))


if __name__ == '__main__':
    main()
