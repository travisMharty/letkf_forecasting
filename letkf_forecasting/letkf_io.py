import os
from collections import namedtuple
import numpy as np
import pandas as pd
import glob
from netCDF4 import Dataset, date2num, num2date
import xarray as xr


def create_folder(results_base_path, year, month, day, run_name):
    file_path_r = os.path.join(results_base_path,
                               f'{year:04}',
                               f'{month:02}',
                               f'{day:02}',
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


def return_analysis_ensemble(*, sat_time, results_file_path):
    analysis_file = time2name(sat_time)
    analysis_file = os.path.join(results_file_path, analysis_file)
    this_ensemble = xr.open_dataset(analysis_file)
    analysis_mean = this_ensemble.isel(time=0).mean(dim='ensemble_number')
    U = analysis_mean['U'].values.astype('float64')
    V = analysis_mean['V'].values.astype('float64')
    CI = analysis_mean['ci'].values.astype('float64')
    ensemble = np.concatenate([U.ravel(), V.ravel(), CI.ravel()])[:, None]
    return ensemble


def time2name(Timestamp):
    year = Timestamp.year
    month = Timestamp.month
    day = Timestamp.day
    hour = Timestamp.hour
    minute = Timestamp.minute
    return f'{year:04}{month:02}{day:02}_{hour:02}{minute:02}Z.nc'


def save_netcdf(file_path_r, U, V, ci, param_dict, we_crop, sn_crop,
                we_stag_crop, sn_stag_crop,
                sat_times, ens_num, flags):
    if 'analysis_fore' in flags:
        if flags['analysis_fore']:
            file_name = time2name(sat_times[0])
            file_name = file_name[:-3] + '_anlys_fore' + '.nc'
        else:
            file_name = time2name(sat_times[0])
    else:
        file_name = time2name(sat_times[0])
    file_path = os.path.join(file_path_r, file_name)
    with Dataset(file_path, mode='w') as store:
        for k, v in param_dict.items():
            try:
                setattr(store, k, v)
            except Exception:
                setattr(store, k, str(v))

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


def calc_time_range(*, sat_times, advect_params):
    # Use all possible satellite images in system unless told to limit
    sat_times_all = sat_times.copy()
    start_time = advect_params['start_time']
    end_time = advect_params['end_time']
    if (start_time != 0) & (end_time != 0):
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        start_time = start_time.tz_localize('UTC')
        end_time = end_time.tz_localize('UTC')
        sat_times_temp = pd.date_range(start_time, end_time, freq='15 min')
        sat_times = sat_times.intersection(sat_times_temp)
    elif start_time != 0:
        start_time = pd.Timestamp(start_time)
        start_time = start_time.tz_localize('UTC')
        sat_times_temp = pd.date_range(start_time, sat_times[-1],
                                       freq='15 min')
        sat_times = sat_times.intersection(sat_times_temp)
    elif end_time != 0:
        end_time = pd.Timestamp(end_time)
        end_time = end_time.tz_localize('UTC')
        sat_times_temp = pd.date_range(sat_times[0], end_time,
                                       freq='15 min')
        sat_times = sat_times.intersection(sat_times_temp)
    return sat_times, sat_times_all


def calc_crop(*, U, V, advect_params):
    horizon = pd.Timedelta(advect_params['max_horizon'])
    horizon = horizon.seconds
    U_min = U.min()
    U_max = U.max()
    V_min = V.min()
    V_max = V.max()
    if U_max > 0:
        left = int(U_max*horizon/250) + 12
    else:
        left = 40
    if U_min < 0:
        right = int(abs(U_min)*horizon/250) + 12
    else:
        right = 40
    if V_max > 0:
        down = int(V_max*horizon/250) + 12
    else:
        down = 40
    if V_min < 0:
        up = int(abs(V_min)*horizon/250) + 12
    else:
        up = 40
    # based on sensor locations
    we_min = 640
    we_max = 800
    sn_min = 600
    sn_max = 824

    we_min_crop = np.max([we_min - left, 0])
    we_max_crop = we_max + right
    sn_min_crop = np.max([sn_min - down, 0])
    sn_max_crop = sn_max + up
    we_stag_min_crop = we_min_crop
    we_stag_max_crop = we_max_crop + 1
    sn_stag_min_crop = sn_min_crop
    sn_stag_max_crop = sn_max_crop + 1
    to_return = {'we_min_crop': we_min_crop,
                 'we_max_crop': we_max_crop,
                 'sn_min_crop': sn_min_crop,
                 'sn_max_crop': sn_max_crop,
                 'we_stag_min_crop': we_stag_min_crop,
                 'we_stag_max_crop': we_stag_max_crop,
                 'sn_stag_min_crop': sn_stag_min_crop,
                 'sn_stag_max_crop': sn_stag_max_crop}
    return to_return


def read_coords(*, data_file_path, advect_params, flags):
    with Dataset(data_file_path, mode='r') as store:
        sat_times = store.variables['time']
        sat_times = num2date(sat_times[:], sat_times.units)
        sat_times = pd.DatetimeIndex(
            sat_times).tz_localize('UTC')
        we = store.variables['west_east'][:]
        sn = store.variables['south_north'][:]
        U = store.variables['U'][:]
        V = store.variables['V'][:]
        wind_times = store.variables['time_wind']
        wind_times = num2date(wind_times[:], wind_times.units)
        wind_times = pd.DatetimeIndex(
            wind_times).tz_localize('UTC')
    crops = calc_crop(U=U, V=V,
                      advect_params=advect_params)
    we_slice = slice(crops['we_min_crop'],
                     crops['we_max_crop'] + 1)
    sn_slice = slice(crops['sn_min_crop'],
                     crops['sn_max_crop'] + 1)
    we_stag_slice = slice(crops['we_stag_min_crop'],
                          crops['we_stag_max_crop'] + 1)
    sn_stag_slice = slice(crops['sn_stag_min_crop'],
                          crops['sn_stag_max_crop'] + 1)
    we_crop = we[we_slice]
    sn_crop = sn[sn_slice]
    we_stag_crop = we[we_stag_slice]
    sn_stag_crop = sn[sn_stag_slice]
    sat_times, sat_times_all = calc_time_range(sat_times=sat_times,
                                               advect_params=advect_params)
    # don't have optical flow for first timestep
    if flags['opt_flow']:
        sat_times = sat_times[1:]
    Coords = namedtuple('coords', ['we', 'sn', 'we_crop', 'sn_crop',
                                   'we_stag_crop', 'sn_stag_crop',
                                   'sat_times', 'sat_times_all', 'wind_times',
                                   'we_slice', 'sn_slice',
                                   'we_stag_slice', 'sn_stag_slice'])
    coords = Coords(we=we, sn=sn, we_crop=we_crop, sn_crop=sn_crop,
                    we_stag_crop=we_stag_crop, sn_stag_crop=sn_stag_crop,
                    sat_times=sat_times, sat_times_all=sat_times_all,
                    wind_times=wind_times,
                    we_slice=we_slice, sn_slice=sn_slice,
                    we_stag_slice=we_stag_slice, sn_stag_slice=sn_stag_slice)

    return coords


def return_single_time(data_file_path, times, time,
                       sn_slice_list, we_slice_list, variables_list):
    index_num = times.get_loc(time)
    to_return = []
    with Dataset(data_file_path, mode='r') as store:
        for variable, sn_slice, we_slice in zip(variables_list,
                                                sn_slice_list,
                                                we_slice_list):
            temp = store.variables[variable][
                index_num, sn_slice, we_slice]
            to_return.append(temp)
    return to_return


def find_run_folder(folder_path):
    if os.path.exists(folder_path):
        return folder_path
    else:
        folder_path = find_latest_run(folder_path)
        return folder_path


def find_latest_run(folder_path):
    folder_path = folder_path + '_???'
    folder_paths = glob.glob(folder_path)
    folder_paths.sort()
    folder_path = folder_paths[-1]
    return folder_path


def add_horizon(ds):
    ds.coords['horizon'] = (ds.time - ds.time[0])/60
    return ds


def return_day(year, month, day, run_name, base_folder, optimize_folder=None,
               analysis_fore_flag=False):
    path = base_folder
    if optimize_folder is None:
        path = os.path.join(
            path,
            f'results/{year:04}/{month:02}/{day:02}/' + run_name)
    else:
        path = os.path.join(
            path, optimize_folder,
            f'{year:04}/{month:02}/{day:02}/' + run_name)
    path = find_run_folder(path)
    if analysis_fore_flag:
        path = os.path.join(path, '????????_????Z_anlys_fore.nc')
    else:
        path = os.path.join(path, '????????_????Z.nc')
    full_day = xr.open_mfdataset(path,
                                 preprocess=add_horizon,
                                 decode_cf=False)
    full_day.horizon.attrs['units'] = 'minutes'
    full_day = xr.decode_cf(full_day)
    return full_day


def add_crop_attributes(ds):
    ds.attrs['we_er_min'] = 240
    ds.attrs['we_er_max'] = 280
    ds.attrs['sn_er_min'] = 32
    ds.attrs['sn_er_max'] = 88
    return ds


def preprocess_for_many_days(ds, buff=0):
    ds = add_crop_attributes(ds)
    ds = ds.sel(
        west_east=slice(ds.we_er_min - buff,
                        ds.we_er_max + buff),
        south_north=slice(ds.sn_er_min - buff,
                          ds.sn_er_max + buff),
        west_east_stag=slice(ds.we_er_min - buff,
                             ds.we_er_max + buff),
        south_north_stag=slice(ds.sn_er_min - buff,
                               ds.sn_er_max + buff))
    ds.coords['horizon'] = (ds.time - ds.time[0])/60
    return ds


def return_many_days_files(
        dates, run, base_folder, only_of_times=False,
        analysis_fore_flag=False):
    files = []
    for date in dates:
        year = date.year
        month = date.month
        day = date.day
        folder = os.path.join(
            base_folder,
            f'results/{year:04}/{month:02}/{day:02}/{run}')
        folder = find_run_folder(folder)
        if analysis_fore_flag:
            these_files = os.path.join(folder, '????????_????Z_anlys_fore.nc')
        else:
            these_files = os.path.join(folder, '????????_????Z.nc')
        these_files = glob.glob(these_files)
        if only_of_times & (run != 'opt_flow'):
            these_files.sort()
            these_files = these_files[1:]
        files += these_files
    return files


def return_many_days(
        dates, run, base_folder, only_of_times=False,
        mean_win_size=None, analysis_fore_flag=False):
    files = return_many_days_files(
        dates, run, base_folder, only_of_times=only_of_times,
        analysis_fore_flag=analysis_fore_flag)
    if mean_win_size is None:
        all_days = xr.open_mfdataset(
            files,
            preprocess=preprocess_for_many_days,
            decode_cf=False)
        all_days.horizon.attrs['units'] = 'minutes'
        all_days = xr.decode_cf(all_days)
    else:
        buff = (mean_win_size - 1)/2

        def this_preprocess(ds):
            ds = preprocess_for_many_days(ds, buff)
            return ds
        all_days = xr.open_mfdataset(
            files,
            preprocess=this_preprocess,
            decode_cf=False)
        all_days.horizon.attrs['units'] = 'minutes'
        all_days = xr.decode_cf(all_days)
        all_days = all_days['ci'].load()
        all_days = all_days.rolling(west_east=mean_win_size,
                                    center=True).mean()
        all_days = all_days.rolling(south_north=mean_win_size,
                                    center=True).mean()
        all_days = add_crop_attributes(all_days)
        all_days = all_days.sel(
            west_east=slice(all_days.we_er_min,
                            all_days.we_er_max),
            south_north=slice(all_days.sn_er_min,
                              all_days.sn_er_max))
    return all_days


def return_truth_files(dates, base_folder):
    files = []
    for date in dates:
        year = date.year
        month = date.month
        day = date.day
        file = os.path.join(base_folder,
                            f'data/{year:04}/{month:02}/{day:02}/data.nc')
        files += [file]
    return files


def preprocess_many_truths(ds):
    ds = ds['ci'].to_dataset()
    return ds


def return_many_truths(dates, base_folder):
    truth_files = return_truth_files(dates, base_folder)
    truth = xr.open_mfdataset(truth_files, preprocess=preprocess_many_truths)
    return truth


def save_newly_created_data(results_file_path, interpolated_ci,
                            interpolated_wrf):
    sat_shape = interpolated_ci['fine_shape']
    U_shape = interpolated_wrf['U_fine_shape']
    V_shape = interpolated_wrf['V_fine_shape']
    x = interpolated_ci['x_fine']
    y = interpolated_ci['y_fine']
    ci = interpolated_ci['ci_fine']
    time = ci.index.tz_convert(None).to_pydatetime()
    time = date2num(time, 'seconds since 1970-1-1')
    ci = ci.values.reshape(
        [time.size, sat_shape[0], sat_shape[1]])
    U = interpolated_wrf['U_fine']
    V = interpolated_wrf['V_fine']
    time_wind = U.index.tz_convert(None).to_pydatetime()
    time_wind = date2num(time_wind, 'seconds since 1970-1-1')
    U = U.values.reshape(time_wind.size, U_shape[0], U_shape[1])
    V = V.values.reshape(time_wind.size, V_shape[0], V_shape[1])
    dx = (x[1] - x[0])
    x_stag = np.concatenate([x - dx/2, [x[-1] + dx/2]])
    dy = (y[1] - y[0])
    y_stag = np.concatenate([y - dy/2, [y[-1] + dy/2]])
    with Dataset(results_file_path, 'w') as store:
        store.createDimension('west_east', size=x.size)
        store.createDimension('south_north', size=y.size)
        store.createDimension('we_stag', size=x_stag.size)
        store.createDimension('sn_stag', size=y_stag.size)
        store.createDimension('time', size=time.size)
        store.createDimension('time_wind', size=time_wind.size)
        wenc = store.createVariable(
            'west_east', 'f8', ('west_east',), zlib=True)
        snnc = store.createVariable(
            'south_north', 'f8', ('south_north',), zlib=True)
        we_stagnc = store.createVariable(
            'we_stag', 'f8', ('we_stag',), zlib=True)
        sn_stagnc = store.createVariable(
            'sn_stag', 'f8', ('sn_stag',), zlib=True)
        timenc = store.createVariable('time', 'f8', ('time',), zlib=True)
        time_windnc = store.createVariable(
            'time_wind', 'f8', ('time_wind',), zlib=True)
        cinc = store.createVariable(
            'ci', 'f8', ('time', 'south_north', 'west_east',), zlib=True)
        Unc = store.createVariable(
            'U', 'f8', ('time_wind', 'south_north', 'we_stag',), zlib=True)
        Vnc = store.createVariable(
            'V', 'f8', ('time_wind', 'sn_stag', 'west_east',), zlib=True)
        wenc[:] = x
        snnc[:] = y
        we_stagnc[:] = x_stag
        sn_stagnc[:] = y_stag
        timenc[:] = time
        time_windnc[:] = time_wind
        cinc[:] = ci
        Unc[:] = U
        Vnc[:] = V
        timenc.units = 'seconds since 1970-1-1'
        time_windnc.units = 'seconds since 1970-1-1'
