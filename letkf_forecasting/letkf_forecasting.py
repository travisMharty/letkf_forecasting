import logging
import numpy as np
from distributed import Client
import pandas as pd
import fenics as fe
from netCDF4 import Dataset, num2date

import letkf_forecasting.random_functions as rf
import letkf_forecasting.letkf_io as letkf_io
from letkf_forecasting.optical_flow import optical_flow
from letkf_forecasting.letkf_io import (
    extract_components
)
from letkf_forecasting.random_functions import (
    perturb_irradiance
)
from letkf_forecasting.advection import (
    advect_5min_ensemble,
    remove_divergence_ensemble,
    noise_fun,
    advect_5min,
    remove_divergence_single
)
from letkf_forecasting.assimilation_accessories import (
    ensemble_creator,
    assimilation_position_generator
)
from letkf_forecasting.assimilation import (
    assimilate_sat_to_wind,
    assimilate_wrf,
    reduced_enkf,
)


def forecast_system(data_file_path, results_file_path,
                    date, io, flags, advect_params, ens_params, pert_params,
                    sat2sat, sat2wind, wrf, opt_flow):
    param_dic = date.copy()
    param_dic.update(io)
    param_dic.update(advect_params)
    param_dic.update(ens_params)
    param_dic.update(pert_params)
    for dic in [flags, sat2sat, sat2wind, wrf, opt_flow]:
        temp = dic.copy()
        name = temp['name'] + '_'
        del temp['name']
        keys = list(temp.keys())
        for k in keys:
            temp[name + k] = temp.pop(k)
        param_dic.update(temp)
    start_time = advect_params['start_time']
    end_time = advect_params['end_time']
    # read initial data from satellite store
    with Dataset(data_file_path, mode='r') as store:
        sat_times = store.variables['time']
        sat_times = num2date(sat_times[:], sat_times.units)
        sat_times = pd.DatetimeIndex(
            sat_times).tz_localize('UTC')
        we = store.variables['west_east'][:]
        sn = store.variables['south_north'][:]
        we_min_crop = store.variables['ci'].we_min_crop
        we_max_crop = store.variables['ci'].we_max_crop
        sn_min_crop = store.variables['ci'].sn_min_crop
        sn_max_crop = store.variables['ci'].sn_max_crop
        wind_times_all = store.variables['time_wind']
        wind_times_all = num2date(wind_times_all[:], wind_times_all.units)
        wind_times_all = pd.DatetimeIndex(
            wind_times_all).tz_localize('UTC')
        we_stag_min_crop = store.variables['U'].we_min_crop
        we_stag_max_crop = store.variables['U'].we_max_crop
        sn_stag_min_crop = store.variables['V'].sn_min_crop
        sn_stag_max_crop = store.variables['V'].sn_max_crop
    we_crop = we[we_min_crop:we_max_crop + 1]
    sn_crop = sn[sn_min_crop:sn_max_crop + 1]
    we_stag_crop = we[we_stag_min_crop:we_stag_max_crop + 1]
    sn_stag_crop = sn[sn_stag_min_crop:sn_stag_max_crop + 1]
    dx = (we[1] - we[0])*1000
    dy = (sn[1] - sn[0])*1000  # dx, dy in m not km
    max_horizon = pd.Timedelta(advect_params['max_horizon'])
    ci_crop_shape = np.array([sn_max_crop - sn_min_crop + 1,
                              we_max_crop - we_min_crop + 1],
                             dtype='int')
    U_crop_shape = np.array([sn_max_crop - sn_min_crop + 1,
                             we_stag_max_crop - we_stag_min_crop + 1],
                            dtype='int')
    V_crop_shape = np.array([sn_stag_max_crop - sn_stag_min_crop + 1,
                             we_max_crop - we_min_crop + 1],
                            dtype='int')
    U_crop_size = U_crop_shape[0]*U_crop_shape[1]
    V_crop_size = V_crop_shape[0]*V_crop_shape[1]
    wind_size = U_crop_size + V_crop_size

    # Use all possible satellite images in system unless told to limit
    sat_times_all = sat_times.copy()
    if (start_time != 0) & (end_time != 0):
        sat_times_temp = pd.date_range(start_time, end_time, freq='15 min')
        sat_times = sat_times.intersection(sat_times_temp)
    elif start_time != 0:
        sat_times_temp = pd.date_range(start_time, sat_times[-1],
                                       freq='15 min')
        sat_times = sat_times.intersection(sat_times_temp)
    elif end_time != 0:
        sat_times_temp = pd.date_range(sat_times[0], end_time,
                                       freq='15 min')
        sat_times = sat_times.intersection(sat_times_temp)

    # Advection calculations
    num_of_horizons = int((max_horizon/15).seconds/60)

    # Creat stuff used to remove divergence
    remove_div_flag = flags['div']
    if flags['div']:
        mesh = fe.RectangleMesh(fe.Point(0, 0),
                                fe.Point(int(V_crop_shape[1] - 1),
                                         int(U_crop_shape[0] - 1)),
                                int(V_crop_shape[1] - 1),
                                int(U_crop_shape[0] - 1))
        FunctionSpace_wind = fe.FunctionSpace(mesh, 'P', 1)

    # Create things needed for assimilations
    if flags['assim']:
        # start cluster
        client = Client(advect_params['client_address'])
        if flags['assim_sat2sat']:
            assim_pos, assim_pos_2d, full_pos_2d = (
                assimilation_position_generator(ci_crop_shape,
                                                sat2sat['grid_size']))
            noise_init = noise_fun(ci_crop_shape)
            noise = noise_init.copy()
        if flags['assim_sat2wind']:
            assim_pos_sat2wind, assim_pos_2d_sat2wind, full_pos_2d_sat2wind = (
                assimilation_position_generator(ci_crop_shape,
                                                sat2wind['grid_size']))
        # Check if these are needed
        if flags['assim_sat2wind']:
            assim_pos_U, assim_pos_2d_U, full_pos_2d_U = (
                assimilation_position_generator(U_crop_shape,
                                                sat2wind['grid_size']))
            assim_pos_V, assim_pos_2d_V, full_pos_2d_V = (
                assimilation_position_generator(V_crop_shape,
                                                sat2wind['grid_size']))
        if flags['assim_wrf']:
            assim_pos_U_wrf, assim_pos_2d_U_wrf, full_pos_2d_U_wrf = (
                assimilation_position_generator(U_crop_shape,
                                                wrf['grid_size']))
            assim_pos_V_wrf, assim_pos_2d_V_wrf, full_pos_2d_V_wrf = (
                assimilation_position_generator(V_crop_shape,
                                                wrf['grid_size']))
        if flags['perturbation']:
            rf_eig, rf_vectors = rf.eig_2d_covariance(
                x=we_crop, y=sn_crop,
                Lx=pert_params['Lx'],
                Ly=pert_params['Ly'], tol=pert_params['tol'])
            rf_approx_var = (
                rf_vectors * rf_eig[None, :] * rf_vectors).sum(-1).mean()
        if flags['assim_of']:
            wind_x_range = (np.max([we_min_crop, we_stag_min_crop]),
                            np.min([we_max_crop, we_stag_max_crop]))
            wind_y_range = (np.max([sn_min_crop, sn_stag_min_crop]),
                            np.min([sn_max_crop, sn_stag_max_crop]))
        sat_time = sat_times[0]
        int_index_wind = wind_times_all.get_loc(sat_times[0],
                                                method='pad')
        wind_time = wind_times_all[int_index_wind]
        with Dataset(data_file_path, mode='r') as store:
            q = store.variables['ci'][sat_times_all == sat_time,
                                      sn_min_crop:sn_max_crop + 1,
                                      we_min_crop:we_max_crop + 1]
            U = store.variables['U'][wind_times_all == wind_time,
                                     sn_min_crop:sn_max_crop + 1,
                                     we_stag_min_crop:we_stag_max_crop + 1]
            V = store.variables['V'][wind_times_all == wind_time,
                                     sn_stag_min_crop:sn_stag_max_crop + 1,
                                     we_min_crop:we_max_crop + 1]
            #  boolean indexing does not drop dimension
            q = q[0]
            U = U[0]
            V = V[0]
        ensemble = ensemble_creator(
            q, U, V, CI_sigma=ens_params['ci_sigma'],
            wind_sigma=ens_params['winds_sigma'],
            ens_size=ens_params['ens_num'])
        del q, U, V
        ens_shape = ensemble.shape
    else:
        ens_params['ens_num'] = 1
    for time_index in range(sat_times.size - 1):
        sat_time = sat_times[time_index]
        save_times = pd.date_range(sat_time, periods=(num_of_horizons + 1),
                                   freq='15min')
        save_times = save_times.tz_convert(None)
        logging.info(str(sat_time))
        int_index_wind = wind_times_all.get_loc(sat_times[0],
                                                method='pad')
        wind_time = wind_times_all[int_index_wind]
        num_of_advec = int((
            sat_times[time_index + 1] -
            sat_times[time_index]).seconds/(60*15))
        if not flags['assim']:  # assums no perturbation
            with Dataset(data_file_path, mode='r') as store:
                q = store.variables['ci'][sat_times_all == sat_time,
                                          sn_min_crop:sn_max_crop + 1,
                                          we_min_crop:we_max_crop + 1]
                U = store.variables['U'][wind_times_all == wind_time,
                                         sn_min_crop:sn_max_crop + 1,
                                         we_stag_min_crop:we_stag_max_crop + 1]
                V = store.variables['V'][wind_times_all == wind_time,
                                         sn_stag_min_crop:sn_stag_max_crop + 1,
                                         we_min_crop:we_max_crop + 1]
                #  boolean indexing does not drop dimension
                q = q[0]
                U = U[0]
                V = V[0]
            if flags['wrf_mean']:
                U = np.ones_like(U)*U.mean()
                V = np.ones_like(V)*V.mean()
            elif flags['div']:
                logging.debug('remove divergence')
                U, V = remove_divergence_single(
                    FunctionSpace_wind, U, V, 4)  # hardwired smoothing
            q_array = q.copy()[None, :, :]
            cx = abs(U).max()
            cy = abs(V).max()
            T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)/advect_params['C_max']))
            dt = (5*60)/T_steps
            for m in range(num_of_horizons):
                logging.info(str(pd.Timedelta('15min')*(m + 1)))
                for n in range(3):
                    q = 1 - q
                    q = advect_5min(q, dt, U, dx, V, dy, T_steps)
                    q = 1 - q
                q_array = np.concatenate([q_array, q[None, :, :]], axis=0)
            letkf_io.save_netcdf(
                results_file_path,
                np.repeat(U[None, None, :, :], num_of_horizons + 1, axis=0),
                np.repeat(V[None, None, :, :], num_of_horizons + 1, axis=0),
                q_array[:, None, :, :],
                param_dic, we_crop, sn_crop,
                we_stag_crop, sn_stag_crop,
                save_times, ens_params['ens_num'])
        else:
            if time_index != 0:
                if flags['assim_sat2wind']:
                    logging.debug('Assim sat2wind')
                    with Dataset(data_file_path, mode='r') as store:
                        q = store.variables['ci'][sat_times_all == sat_time,
                                                  sn_min_crop:sn_max_crop + 1,
                                                  we_min_crop:we_max_crop + 1]
                        #  boolean indexing does not drop dimension
                        q = q[0]
                    ensemble = assimilate_sat_to_wind(
                        ensemble=ensemble,
                        observations=q.ravel(),
                        R_inverse_wind=1/sat2wind['sig']**2,
                        wind_inflation=sat2wind['infl'],
                        domain_shape=ci_crop_shape,
                        U_shape=U_crop_shape, V_shape=V_crop_shape,
                        localization_length_wind=sat2wind['loc'],
                        assimilation_positions=assim_pos_sat2wind,
                        assimilation_positions_2d=assim_pos_2d_sat2wind,
                        full_positions_2d=full_pos_2d_sat2wind)
                    remove_div_flag = True
                    del q

                if sat_time == wind_time:
                    logging.debug('Assim WRF')
                    with Dataset(data_file_path, mode='r') as store:
                        U = store.variables['U'][wind_times_all == wind_time,
                                                 sn_min_crop:sn_max_crop + 1,
                                                 we_stag_min_crop:
                                                 we_stag_max_crop + 1]
                        V = store.variables['V'][wind_times_all == wind_time,
                                                 sn_stag_min_crop:
                                                 sn_stag_max_crop + 1,
                                                 we_min_crop:we_max_crop + 1]
                        #  boolean indexing does not drop dimension
                        q = q[0]
                        U = U[0]
                        V = V[0]
                    remove_div_flag = True
                    if flags['assim_wrf']:
                        ensemble[:U_crop_size] = assimilate_wrf(
                            ensemble=ensemble[:U_crop_size],
                            observations=U.ravel(),
                            R_inverse=1/wrf['sig']**2,
                            wind_inflation=wrf['infl'],
                            wind_shape=U_crop_shape,
                            localization_length_wind=wrf['loc'],
                            assimilation_positions=assim_pos_U_wrf,
                            assimilation_positions_2d=assim_pos_2d_U_wrf,
                            full_positions_2d=full_pos_2d_U_wrf)

                        ensemble[U_crop_size:
                                 U_crop_size + V_crop_size] = assimilate_wrf(
                                     ensemble=ensemble[U_crop_size:
                                                       U_crop_size + V_crop_size],
                                     observations=V.ravel(),
                                     R_inverse=1/wrf['sig']**2,
                                     wind_inflation=wrf['infl'],
                                     wind_shape=V_crop_shape,
                                     localization_length_wind=wrf['loc'],
                                     assimilation_positions=assim_pos_V_wrf,
                                     assimilation_positions_2d=assim_pos_2d_V_wrf,
                                     full_positions_2d=full_pos_2d_V_wrf)
                        del U, V
                    else:
                        random_nums = np.random.normal(
                            loc=0,
                            scale=ens_params['wind_sigma'][0],
                            size=ens_shape[1])
                        ensemble[:U_crop_size] = (U.ravel()[:, None]
                                                  + random_nums[None, :])
                        random_nums = np.random.normal(
                            loc=0,
                            scale=ens_params['wind_sigma'][1],
                            size=ens_shape[1])
                        ensemble[U_crop_size:
                                 U_crop_size + V_crop_size] = (
                                     V.ravel()[:, None]
                                     + random_nums[None, :])
                if flags['assim_of']:
                    logging.debug('calc of')
                    # retreive OF vectors
                    time0 = sat_times[time_index - 1]
                    with Dataset(data_file_path, mode='r') as store:
                        this_U = store.variables['U'][
                            wind_times_all == wind_time, :, :]
                        this_V = store.variables['V'][
                            wind_times_all == wind_time, :, :]
                        image0 = store.variables['ci'][sat_times_all == time0,
                                                       :, :]
                        image1 = store.variables['ci'][
                            sat_times_all == sat_time, :, :]
                        # boolean indexing does not drop dimension
                        this_U = this_U[0]
                        this_V = this_V[0]
                        image0 = image0[0]
                        image1 = image1[0]
                    u_of, v_of, pos = optical_flow(image0, image1,
                                                   time0, sat_time,
                                                   this_U, this_V)
                    del this_U, this_V, image0, image1
                    pos = pos*4  # optical flow done on coarse grid

                    # need to select only pos in crop domain; convert to crop
                    keep = np.logical_and(
                        np.logical_and(pos[:, 0] > wind_x_range[0],
                                       pos[:, 0] < wind_x_range[1]),
                        np.logical_and(pos[:, 1] > wind_y_range[0],
                                       pos[:, 1] < wind_y_range[1]))
                    pos = pos[keep]
                    u_of = u_of[keep]
                    v_of = v_of[keep]
                    pos[:, 0] -= wind_x_range[0]
                    pos[:, 1] -= wind_y_range[0]
                    pos = pos.T
                    pos = pos[::-1]
                    u_of_flat_pos = np.ravel_multi_index(pos, U_crop_shape)
                    v_of_flat_pos = np.ravel_multi_index(pos, V_crop_shape)
                    logging.debug('assim of')
                    remove_div_flag = True
                    x_temp = np.arange(U_crop_shape[1])*dx/1000  # in km not m
                    y_temp = np.arange(U_crop_shape[0])*dx/1000
                    x_temp, y_temp = np.meshgrid(x_temp, y_temp)
                    ensemble[:U_crop_size] = reduced_enkf(
                        ensemble=ensemble[:U_crop_size],
                        observations=u_of, R_sig=opt_flow['sig'],
                        flat_locations=u_of_flat_pos,
                        inflation=opt_flow['infl'],
                        localization=opt_flow['loc'],
                        x=x_temp.ravel(), y=y_temp.ravel())
                    x_temp = np.arange(V_crop_shape[1])*dx/1000
                    y_temp = np.arange(V_crop_shape[0])*dx/1000
                    x_temp, y_temp = np.meshgrid(x_temp, y_temp)
                    ensemble[U_crop_size:
                             U_crop_size + V_crop_size] = reduced_enkf(
                        ensemble=ensemble[U_crop_size:U_crop_size +
                                          V_crop_size],
                        observations=v_of, R_sig=opt_flow['sig'],
                        flat_locations=v_of_flat_pos,
                        inflation=opt_flow['infl'],
                        localization=opt_flow['loc'],
                        x=x_temp.ravel(), y=y_temp.ravel())
                if not flags['assim_sat2sat']:
                    with Dataset(data_file_path, mode='r') as store:
                        q = store.variables['ci'][sat_times_all == sat_time,
                                                  sn_min_crop:sn_max_crop + 1,
                                                  we_min_crop:we_max_crop + 1]
                        # boolean indexing does not drop dimension
                        q = q[0]
                    ensemble[wind_size:] = q.ravel()[:, None]

            if remove_div_flag and flags['div']:
                logging.debug('remove divergence')
                remove_div_flag = False
                ensemble[:wind_size] = remove_divergence_ensemble(
                    FunctionSpace_wind, ensemble[:wind_size],
                    U_crop_shape, V_crop_shape, 4)  # hardwired smoothing
            temp_ensemble = ensemble.copy()
            ensemble_array = temp_ensemble.copy()[None, :, :]
            cx = abs(temp_ensemble[:U_crop_size]).max()
            cy = abs(temp_ensemble[U_crop_size:
                                   U_crop_size + V_crop_size]).max()
            T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)
                                  / advect_params['C_max']))
            dt = (5*60)/T_steps
            for m in range(num_of_horizons):
                logging.info(str(pd.Timedelta('15min')*(m + 1)))
                for n in range(3):
                    temp_ensemble = advect_5min_ensemble(
                        ensemble, dt, dx, dy, T_steps,
                        U_crop_shape, V_crop_shape,
                        ci_crop_shape, client)
                    if flags['perturbation']:
                        temp_ensemble[wind_size:] = perturb_irradiance(
                            temp_ensemble[wind_size:], ci_crop_shape,
                            pert_params['edge_weight'],
                            pert_params['pert_mean'],
                            pert_params['pert_sigma'],
                            rf_approx_var, rf_eig, rf_vectors)
                ensemble_array = np.concatenate(
                    [ensemble_array, temp_ensemble[None, :, :]],
                    axis=0)
                if num_of_advec == m:
                    ensemble = temp_ensemble.copy()
            U, V, ci = extract_components(
                ensemble_array, ens_params['ens_num'], num_of_horizons + 1,
                U_crop_shape, V_crop_shape, ci_crop_shape)
            letkf_io.save_netcdf(
                results_file_path, U, V, ci, param_dic,
                we_crop, sn_crop, we_stag_crop, sn_stag_crop,
                save_times, ens_params['ens_num'])
    return
