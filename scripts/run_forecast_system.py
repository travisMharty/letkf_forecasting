import os
import logging
import time as time_py
import letkf_forecasting.letkf_forecasting as lf


def main():
    year = 2014
    month = 5
    day = 29

    home = os.path.expanduser("~")
    file_path = f'{home}/data/{year:04}/{month:02}/{day:02}/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path_store = os.path.join(file_path, 'data.nc')
    print(file_path_store)
    param_dic = {
        # data_paths
        'data_file_path': file_path_store,
        'run_name': 'test',

        # Switches
        'assim_test': 0,
        'perturbation_test': 0,
        'div_test': 0,
        'assim_of_test': 0,
        'assim_sat2sat_test': 0,
        'assim_sat2wind_test': 0,
        'assim_wrf_test': 0,

        # advection_params
        'start_time': 0,
        'end_time': 0,
        'C_max': 0.7,
        'max_horizon': '1hour',
        'client_address': '127.0.0.1:8786',

        # assimilation_params
        # assim_sat2sat
        'sig_sat2sat': 0.05,
        'loc_sat2sat': 1*4,
        'infl_sat2sat': 1.5,
        'assim_gs_sat2sat': 20,
        # assim_sat2wind
        'sig_sat2wind': 1,
        'loc_sat2wind': 30*4,
        'infl_sat2wind': 4,
        'assim_gs_sat2wind': 20,
        # assim_wrf
        'sig_wrf': 0.5,
        'infl_wrf': 1,
        'loc_wrf': 1*4,
        'assim_gs_wrf': 5,
        # assim_OF
        'sig_of': 1,
        'loc_of': 20,  # in km not grid spaces,
        'infl_of': 4,

        # ensemble_params
        'ens_num': 20,
        'winds_sigma': (1, 1),
        'ci_sigma': 0.4,

        # perturbation_params
        'Lx': 5,
        'Ly': 5,
        'tol': 0.005,
        'pert_sigma': 0.15/3,
        'pert_mean': 0,
        'edge_weight': 1,
    }

    time0 = time_py.time()

    logging.basicConfig(
        filename='/home2/travis/python_code/letkf_forecasting_other_things/logs/letkf.log',  # noqa
        filemode='w', level=logging.DEBUG)
    logging.info('Started')

    lf.forecast_system(param_dic, **param_dic)
    logging.info('Ended')
    time1 = time_py.time()
    print('It took: ' + str((time1 - time0)/60))
    logging.infor('It took: ' + str((time1 - time0)/60))


if __name__ == '__main__':
    main()
