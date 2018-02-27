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
        we_nc = store.createVariable('west_east', 'f8', ('west_east',),
                                     zlib=True)
        sn_nc = store.createVariable('south_north', 'f8', ('south_north',),
                                     zlib=True)
        we_stag_nc = store.createVariable('west_east_stag', 'f8',
                                          ('west_east_stag',),
                                          zlib=True)
        sn_stag_nc = store.createVariable('south_north_stag', 'f8',
                                          ('south_north_stag',),
                                          zlib=True)
        time_nc = store.createVariable('time', 'f8', ('time',))
        we_nc[:] = we_crop
        sn_nc[:] = sn_crop
        we_stag_nc[:] = we_stag_crop
        sn_stag_nc[:] = sn_stag_crop
        sat_times_nc = sat_times.to_pydatetime()
        time_units = 'seconds since 1970-1-1'
        sat_times_nc = date2num(sat_times_nc, time_units)
        time_nc[:] = sat_times_nc
        time_nc.units = time_units
        store.createDimension('ensemble_number', size=ens_num)
        ensemble_number_nc = store.createVariable('ensemble_number',
                                                  'i8', ('ensemble_number'))
        ensemble_number_nc[:] = np.arange(ens_num)
        ci_nc = store.createVariable('ci', 'f8',
                                     ('time', 'ensemble_number',
                                      'south_north', 'west_east',),
                                     zlib=True)
        U_nc = store.createVariable('U', 'f8',
                                    ('time', 'ensemble_number',
                                     'south_north', 'west_east_stag',),
                                    zlib=True)
        V_nc = store.createVariable('V', 'f8',
                                    ('time', 'ensemble_number',
                                     'south_north_stag', 'west_east',),
                                    zlib=True)
        U_nc[:, :, :, :] = U
        V_nc[:, :, :, :] = V
        ci_nc[:, :, :, :] = ci
