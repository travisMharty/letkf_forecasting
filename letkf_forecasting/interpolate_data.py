import os
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import letkf_forecasting.prepare_sat_data as prep


def interp_sat(times, dx, sat_path):
    sat_shape = np.load(sat_path + 'domain_shape.npy')
    sat_x = np.load(sat_path + 'x.npy')
    sat_y = np.load(sat_path + 'y.npy')
    cloudiness_index = pd.read_hdf(sat_path + 'cloudiness_index.h5')
    cloudiness_index.index = cloudiness_index.index.tz_convert('MST')
    ci = cloudiness_index.ix[times].dropna()

    data = ci.iloc[0].values.reshape(sat_shape)
    xi = sat_x.reshape(sat_shape)[0, :]
    yi = sat_y.reshape(sat_shape)[:, 0]
    f = interpolate.interp2d(xi, yi, data, kind='linear')

    x_fine = np.arange(sat_x[0], sat_x[-1] + dx, dx)
    y_fine = np.arange(sat_y[0], sat_y[-1] + dx, dx)
    fine_shape = (y_fine.size, x_fine.size)
    sat_times = ci.index
    data = f(x_fine, y_fine).ravel()
    ci_fine = pd.DataFrame(data=data[None, :], index=[sat_times[0]])

    for t in np.arange(sat_times.size - 1) + 1:
        this_time = sat_times[t]
        data = ci.loc[this_time].values.reshape(sat_shape)
        f = interpolate.interp2d(xi, yi, data, kind='linear')
        data = f(x_fine, y_fine).ravel()
        temp = pd.DataFrame(data=data[None, :], index=[this_time])
        ci_fine = ci_fine.append(temp)
    to_return = {'ci_fine': ci_fine, 'x_fine': x_fine,
                 'y_fine': y_fine, 'fine_shape': fine_shape,
                 'x_coarse': sat_x, 'y_coarse': sat_y,
                 'coarse_shape': sat_shape}
    return to_return


def interp_wind(U, V, save_path, date):
    suffix = '_' + str(date.month) + '_' + str(date.day)
    x_coarse = np.load(save_path + 'x.npy')
    y_coarse = np.load(save_path + 'y.npy')
    coarse_shape = np.load(save_path + 'domain_shape.npy')
    load_path = (save_path + '{var_name}' + '{extension}')
    x_fine = np.load(load_path.format(var_name='x_fine', extension='.npy'))
    y_fine = np.load(load_path.format(var_name='y_fine', extension='.npy'))
    dx = x_fine[1] - x_fine[0]
    fine_shape = np.load(
        load_path.format(var_name='fine_shape', extension='.npy'))

    load_path = (save_path + 'for' + suffix + '/raw_winds/{var}')
    U = pd.read_hdf(load_path.format(var='U.h5'))
    V = pd.read_hdf(load_path.format(var='V.h5'))

    ## delete after addressing bug
    U.index = U.index.tz_convert('MST')
    V.index = V.index.tz_convert('MST')
    ## delete after addressing bug

    U_shape = np.load(load_path.format(var='U_shape.npy'))
    V_shape = np.load(load_path.format(var='V_shape.npy'))
    wind_lats = np.load(
        load_path.format(var='wind_lats.npy'))
    wind_lons = np.load(
        load_path.format(var='wind_lons.npy'))
    wind_times = U.index

    wind_x, wind_y = prep.sphere_to_lcc(wind_lats, wind_lons)
    wind_x = wind_x.reshape([U_shape[0], V_shape[1]])
    wind_y = wind_y.reshape([U_shape[0], V_shape[1]])
    y_step_sn = np.diff(wind_y, axis=0).mean(axis=0)
    y_step_we = np.diff(wind_y, axis=1).mean(axis=1)
    x_step_sn = np.diff(wind_x, axis=0).mean(axis=0)
    x_step_we = np.diff(wind_x, axis=1).mean(axis=1)

    U_y = np.concatenate([wind_y - y_step_we[:, None]/2,
                          (wind_y[:, -1] + y_step_we/2)[:, None]], axis=1)
    U_x = np.concatenate([wind_x - x_step_we[:, None]/2,
                          (wind_x[:, -1] + x_step_we/2)[:, None]], axis=1)

    V_y = np.concatenate([wind_y - y_step_sn[None, :]/2,
                          (wind_y[-1, :] + y_step_sn/2)[None, :]], axis=0)
    V_x = np.concatenate([wind_x - x_step_sn[None, :]/2,
                          (wind_x[-1, :] + x_step_sn/2)[None, :]], axis=0)

    U_data_positions = np.stack([U_x.ravel(), U_y.ravel()], axis=1)
    V_data_positions = np.stack([V_x.ravel(), V_y.ravel()], axis=1)

    y_coarse = y_coarse.reshape(coarse_shape)
    x_coarse = x_coarse.reshape(coarse_shape)
    U_y_coarse = np.concatenate([y_coarse,
                                 (y_coarse[:, -1])[:, None]], axis=1)
    U_x_coarse = np.concatenate([x_coarse - dx/2,
                                 (x_coarse[:, -1] + dx/2)[:, None]], axis=1)
    V_y_coarse = np.concatenate([y_coarse - dx/2,
                                 (y_coarse[-1, :] + dx/2)[None, :]], axis=0)
    V_x_coarse = np.concatenate([x_coarse,
                                 (x_coarse[-1, :])[None, :]], axis=0)
    U_coarse_shape = U_x_coarse.shape
    V_coarse_shape = V_x_coarse.shape
    U_interp_positions = np.stack(
        [U_x_coarse.ravel(), U_y_coarse.ravel()], axis=1)
    V_interp_positions = np.stack(
        [V_x_coarse.ravel(), V_y_coarse.ravel()], axis=1)

    y_fine = y_fine.reshape(fine_shape)[:, 0]
    x_fine = x_fine.reshape(fine_shape)[0, :]
    U_x_fine = np.concatenate([x_fine - dx/2,
                               [x_fine[-1] + dx/2]], axis=0)
    V_y_fine = np.concatenate([y_fine - dx/2,
                               [y_fine[-1] + dx/2]], axis=0)

    U_fine_shape = (y_fine.size, U_x_fine.size)
    V_fine_shape = (V_y_fine.size, x_fine.size)

    data = U.values.T
    f = interpolate.NearestNDInterpolator(U_data_positions, data)
    data = f(U_interp_positions).T
    fine_data = data[0].reshape(U_coarse_shape)
    f = interpolate.interp2d(
        U_x_coarse[0, :], U_y_coarse[:, 0], fine_data)
    fine_data = f(U_x_fine, y_fine).ravel()
    U_fine = pd.DataFrame(data=fine_data[None, :], index=[wind_times[0]])

    for t in np.arange(wind_times.size - 1) + 1:
        this_time = wind_times[t]
        fine_data = data[t].reshape(U_coarse_shape)
        f = interpolate.interp2d(
            U_x_coarse[0, :], U_y_coarse[:, 0], fine_data)
        fine_data = f(U_x_fine, y_fine).ravel()
        temp = pd.DataFrame(data=fine_data[None, :], index=[this_time])
        U_fine = U_fine.append(temp)

    data = V.values.T
    f = interpolate.NearestNDInterpolator(V_data_positions, data)
    data = f(V_interp_positions).T
    fine_data = data[0].reshape(V_coarse_shape)
    f = interpolate.interp2d(
        V_x_coarse[0, :], V_y_coarse[:, 0], fine_data)
    fine_data = f(x_fine, V_y_fine).ravel()
    V_fine = pd.DataFrame(data=fine_data[None, :], index=[wind_times[0]])

    for t in np.arange(wind_times.size - 1) + 1:
        this_time = wind_times[t]
        fine_data = data[t].reshape(V_coarse_shape)
        f = interpolate.interp2d(
            V_x_coarse[0, :], V_y_coarse[:, 0], fine_data)
        fine_data = f(x_fine, V_y_fine).ravel()
        temp = pd.DataFrame(data=fine_data[None, :], index=[this_time])
        V_fine = V_fine.append(temp)

    save_path = save_path + 'for' + suffix + '/' + '{var}'
    U_fine.to_hdf(save_path.format(var='U.h5'), 'U')
    V_fine.to_hdf(save_path.format(var='V.h5'), 'V')
    np.save(save_path.format(var='U_shape'), U_fine_shape)
    np.save(save_path.format(var='V_shape'), V_fine_shape)
