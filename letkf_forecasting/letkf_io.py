import os
import numpy as np
import pandas as pd
import glob
from netCDF4 import Dataset, date2num, num2date


def create_path(year, month, day, run_name):
    home = os.path.expanduser('~')
    run_num = 0
    file_path_r = (f'{home}/results/{year:04}'
                   f'/{month:02}/{day:02}/' +
                   run_name + '_000')
    if not os.path.exists(file_path_r):
        os.makedirs(file_path_r)
    else:
        file_path_r = os.path.split(file_path_r)[0]
        run_num = glob.glob(os.path.join(file_path_r, run_name + '_???'))
        run_num.sort()
        run_num = run_num[-1]
        run_num = int(run_num[-3:]) + 1
        file_path_r = os.path.join(file_path_r, run_name + f'_{run_num:03}')
        os.makedirs(file_path_r)
    return file_path_r


def time2name(Timestamp):
    year = Timestamp.year
    month = Timestamp.month
    day = Timestamp.day
    hour = Timestamp.hour
    minute = Timestamp.minute
    return f'{year:04}{month:02}{day:02}_{hour:02}{minute:02}Z.nc'


def save_netcdf(file_path_r, U, V, ci, param_dic, we_crop, sn_crop,
                we_stag_crop, sn_stag_crop,
                sat_times, ens_num):
    file_path = os.path.join(file_path_r, time2name(sat_times[0]))
    with Dataset(file_path, mode='w') as store:
        for k, v in param_dic.items():
            setattr(store, k, v)
        store.createDimension('west_east', size=we_crop.size)
        store.createDimension('south_north', size=sn_crop.size)
        store.createDimension('west_east_stag', size=we_stag_crop.size)
        store.createDimension('south_north_stag', size=sn_stag_crop.size)
        store.createDimension('time', size=sat_times.size)
        we_nc = store.createVariable('west_east', 'f4', ('west_east',),
                                     zlib=True)
        sn_nc = store.createVariable('south_north', 'f4', ('south_north',),
                                     zlib=True)
        we_stag_nc = store.createVariable('west_east_stag', 'f4',
                                          ('west_east_stag',),
                                          zlib=True)
        sn_stag_nc = store.createVariable('south_north_stag', 'f4',
                                          ('south_north_stag',),
                                          zlib=True)
        time_nc = store.createVariable('time', 'u4', ('time',))
        we_nc[:] = we_crop
        sn_nc[:] = sn_crop
        we_stag_nc[:] = we_stag_crop
        sn_stag_nc[:] = sn_stag_crop
        sat_times_nc = sat_times.to_pydatetime()
        time_units = 'seconds since 1970-1-1'
        sat_times_nc = date2num(sat_times_nc, time_units)
        time_nc[:] = sat_times_nc
        time_nc.units = time_units
        time_nc.init = sat_times_nc[0]
        store.createDimension('ensemble_number', size=ens_num)
        ensemble_number_nc = store.createVariable('ensemble_number',
                                                  'u4', ('ensemble_number'))
        ensemble_number_nc[:] = np.arange(ens_num)
        ci_nc = store.createVariable('ci', 'f4',
                                     ('time', 'ensemble_number',
                                      'south_north', 'west_east',),
                                     zlib=True)
        U_nc = store.createVariable('U', 'f4',
                                    ('time', 'ensemble_number',
                                     'south_north', 'west_east_stag',),
                                    zlib=True)
        V_nc = store.createVariable('V', 'f4',
                                    ('time', 'ensemble_number',
                                     'south_north_stag', 'west_east',),
                                    zlib=True)
        U_nc[:, :, :, :] = U
        V_nc[:, :, :, :] = V
        ci_nc[:, :, :, :] = ci


def extract_components(ensemble_array, ens_num, time_num,
                       U_shape, V_shape, ci_shape):
    U_size = U_shape[0]*U_shape[1]
    V_size = V_shape[0]*V_shape[1]
    wind_size = U_size + V_size
    ensemble_array = np.transpose(ensemble_array, (0, 2, 1))
    U = ensemble_array[:, :, :U_size].reshape(
        time_num, ens_num, U_shape[0], U_shape[1])
    V = ensemble_array[:, :, U_size:wind_size].reshape(
        time_num, ens_num, V_shape[0], V_shape[1])
    ci = ensemble_array[:, :, wind_size:].reshape(
        time_num, ens_num, ci_shape[0], ci_shape[1])
    return U, V, ci
