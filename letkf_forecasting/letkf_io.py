import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num, num2date


def time2string(Timestamp):
    year = Timestamp.year
    month = Timestamp.month
    day = Timestamp.day
    hour = Timestamp.hour
    minute = Timestamp.minute
    return f'{year:04}{month:02}{day:02}_{hour:02}{minute:02}Z'


def save_netcdf(file_path_r, data, param_dic, we_crop, sn_crop,
                we_stag_crop, sn_stag_crop,
                sat_times, ens_num):
    with Dataset(file_path_r, mode='w') as store:
        for k, v in param_dic.items():
            setattr(store, k, v)
        store.createDimension('west_east', size=we_crop.size)
        store.createDimension('south_north', size=sn_crop.size)
        store.createDimension('we_stag', size=we_stag_crop.size)
        store.createDimension('sn_stag', size=sn_stag_crop.size)
        store.createDimension('time', size=sat_times.size)
        we_nc = store.createVariable('west_east', 'f8', ('west_east',),
                                     zlib=True)
        sn_nc = store.createVariable('south_north', 'f8', ('south_north',),
                                     zlib=True)
        we_stag_nc = store.createVariable('we_stag', 'f8', ('we_stag',),
                                          zlib=True)
        sn_stag_nc = store.createVariable('sn_stag', 'f8', ('sn_stag',),
                                          zlib=True)
        time_nc = store.createVariable('time', 'f8', ('time',))
        forecast_horizon_nc = store.createVariable('forecast_horizon', 'i8',
                                                   ('forecast_horizon',))
        we_nc[:] = we_crop
        sn_nc[:] = sn_crop
        we_stag_nc[:] = we_stag_crop
        sn_stag_nc[:] = sn_stag_crop
        sat_times_nc = (sat_times
                        .tz_convert('UTC').tz_convert(None)
                        .to_pydatetime())
        time_units = 'seconds since 1970-1-1'
        sat_times_nc = date2num(sat_times_nc, time_units)
        time_nc[:] = sat_times_nc
        forecast_horizon_nc[:] = np.arange(num_of_horizons + 1)*15
        time_nc.units = time_units
        forecast_horizon_nc.units = 'minutes since {time}'
        store.createDimension('ensemble_number', size=ens_num)
        ensemble_number_nc = store.createVariable('ensemble_number',
                                                  'i8', ('ensemble_number'))
        ensemble_number_nc[:] = np.arange(ens_num)
        store.createVariable('ci', 'f8',
                             ('time', 'forecast_horizon', 'ensemble_number',
                              'south_north', 'west_east',),
                             zlib=True)
        store.createVariable('U', 'f8',
                             ('time', 'ensemble_number',
                              'south_north', 'we_stag',),
                             zlib=True)
        store.createVariable('V', 'f8',
                             ('time', 'ensemble_number',
                              'sn_stag', 'west_east',),
                             zlib=True)
