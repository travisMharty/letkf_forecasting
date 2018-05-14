import argparse
import yaml
import os
import logging
import time as time_py
import pandas as pd
import letkf_forecasting.letkf_forecasting as lf
import letkf_forecasting.letkf_io as letkf_io
import letkf_forecasting.interpolate_data as interpolate_data


def main():
    time0 = time_py.time()
    dx = 0.25
    solar_noon = pd.Timestamp('12:30:00')
    step = pd.Timestamp('3h')
    parser = argparse.ArgumentParser(
        description='Run forecast_system using provided configuration file.')
    parser.add_argument('data_folder', type=str,
                        help='The path to the satellite data folder.')
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
    results_file_path = os.path.join(home,
                                     'data',
                                     f'{year:04}',
                                     f'{month:02}',
                                     f'{day:02}',
                                     'data.nc')
    data_file_path = args.data_folder

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

    # logging.basicConfig(
    #     filename=os.path.join(data_file_path)letkf.log',  # noqa
    #     filemode='w', level=logging.DEBUG)
    logging.info('Started')
    returned = interpolate_data.interp_sat(
        time_range_ci, dx, data_file_path)
    ci = returned['ci']
    x = returned['x']
    y = returned['y']
    ci_shape = returned['ci_shape']

    retuned = gwd.main(time_ragne_wrf, wrf_path)
    U = returned['U']
    V = returned['V']
    bottom_top = returned['bottom_top']
    wind_lats = returned['wind_lats']
    wind_lons = returned['wind_lons']
    U_shape = returned['U_shape']
    V_shape = returned['V_shape']

    time1 = time_py.time()
    print('It took: ' + str((time1 - time0)/60))
    logging.info('It took: ' + str((time1 - time0)/60))


if __name__ == '__main__':
    main()
