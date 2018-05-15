import argparse
import yaml
import os
import logging
import time as time_py
import pandas as pd
import letkf_forecasting.letkf_forecasting as lf
import letkf_forecasting.letkf_io as letkf_io
import letkf_forecasting.interpolate_data as interpolate_data
import letkf_forecasting.get_wrf_data as get_wrf_data


def main():
    time0 = time_py.time()
    dx = 0.25
    solar_noon = pd.Timestamp('12:30:00')
    step = pd.Timedelta('3h')
    parser = argparse.ArgumentParser(
        description='Create needed data given a date.')
    parser.add_argument('-y', '--year', type=int,
                        help='The year you wish to run.')
    parser.add_argument('-m', '--month', type=int,
                        help='The month you wish to run.')
    parser.add_argument('-d', '--day', type=int,
                        help='The day you wish to run.')
    args = parser.parse_args()

    year = args.year
    month = args.month
    day = args.day
    # Create path to save results
    home = os.path.expanduser('~')
    results_file_path = os.path.join('/a2/uaren/travis/letkf_data/',
                                     f'{year:04}',
                                     f'{month:02}',
                                     f'{day:02}')
    if not os.path.exists(results_file_path):
        os.makedirs(results_file_path)
    results_file_path = os.path.join(results_file_path,
                                     'data.nc')
    wrf_path = os.path.join('/a2/uaren/',
                            f'{year:04}',
                            f'{month:02}',
                            f'{day:02}',
                            'solar_3/wrfsolar_d02_hourly.nc')

    data_file_path = os.path.join(home, 'data/satellite_data/')

    solar_noon = solar_noon.replace(year=year, month=month, day=day)
    start = solar_noon - step
    end = solar_noon + step
    start_wrf = (start - pd.Timedelta('30min')).round(freq='1h')
    end_wrf = (end + pd.Timedelta('30min')).round(freq='1h')
    time_range_wrf = pd.date_range(start_wrf, end_wrf, freq='1h')
    time_range_wrf = time_range_wrf.tz_localize('MST')
    start_ci = (start - pd.Timedelta('7.5min')).round(freq='15min')
    end_ci = (end + pd.Timedelta('7.5min')).round(freq='15min')
    time_range_ci = pd.date_range(start_ci, end_ci, freq='15min')
    time_range_ci = time_range_ci.tz_localize('MST')

    log_path = os.path.join(
        os.path.split(results_file_path)[0], 'data_creation.log')
    logging.basicConfig(
        filename=log_path,
        filemode='w', level=logging.DEBUG)
    logging.info('Started')
    logging.info('Start to interpolate satellite data.')

    interpolated_ci = interpolate_data.interp_sat(
        time_range_ci, dx, data_file_path)

    logging.info('Retrieve WRF data.')
    raw_winds = get_wrf_data.main(time_range_wrf, wrf_path,
                                  interpolated_ci)

    logging.info('Interpolate WRF data.')
    interpolated_wrf = interpolate_data.interp_wind(
        interpolated_ci, raw_winds)

    logging.info('Saving Data.')
    letkf_io.save_newly_created_data(results_file_path, interpolated_ci,
                                     interpolated_wrf)

    time1 = time_py.time()
    print('It took: ' + str((time1 - time0)/60))
    logging.info('It took: ' + str((time1 - time0)/60))


if __name__ == '__main__':
    main()
