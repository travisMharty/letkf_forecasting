import logging
import os
from collections import namedtuple
import numpy as np
import pandas as pd
import fenics as fe
import scipy as sp
from netCDF4 import Dataset, num2date
from metpy.units import units
import metpy.calc.thermo as thermo
from siphon.simplewebservice.wyoming import WyomingUpperAir
from letkf_forecasting import __version__
from letkf_forecasting.optical_flow import optical_flow
from letkf_forecasting.letkf_io import (
    return_analysis_ensemble,
    return_single_time,
    extract_components,
    save_netcdf,
    read_coords,
)
from letkf_forecasting.random_functions import (
    perturb_irradiance,
    eig_2d_covariance,
    perturb_winds,
)
from letkf_forecasting.advection import (
    advect_5min_ensemble,
    remove_divergence_ensemble,
    noise_fun,
    advect_5min_single,
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


def set_up_param_dict(*, date, io, flags, advect_params, ens_params,
                      pert_params, sat2sat, sat2wind, wrf, opt_flow):
    param_dict = date.copy()
    param_dict.update(io)
    param_dict.update(advect_params)
    param_dict.update(ens_params)
    param_dict.update(pert_params)
    for adict in [flags, sat2sat, sat2wind, wrf, opt_flow]:
        temp = adict.copy()
        name = temp['name'] + '_'
        del temp['name']
        keys = list(temp.keys())
        for k in keys:
            temp[name + k] = temp.pop(k)
        param_dict.update(temp)
    param_dict['git_version'] = __version__
    return param_dict


def dict2nt(adict, aname):
    nt = namedtuple(aname, adict.keys())(**adict)
    return nt


def calc_system_variables(*, coords, advect_params, flags, pert_params):
    dx = (coords.we[1] - coords.we[0])*1000
    dy = (coords.sn[1] - coords.sn[0])*1000  # dx, dy in m not km
    max_horizon = pd.Timedelta(advect_params['max_horizon'])
    ci_crop_shape = np.array([coords.sn_crop.size,
                              coords.we_crop.size],
                             dtype='int')
    U_crop_shape = np.array([coords.sn_crop.size,
                             coords.we_stag_crop.size],
                            dtype='int')
    V_crop_shape = np.array([coords.sn_stag_crop.size,
                             coords.we_crop.size],
                            dtype='int')
    U_crop_size = U_crop_shape[0]*U_crop_shape[1]
    V_crop_size = V_crop_shape[0]*V_crop_shape[1]
    wind_size = U_crop_size + V_crop_size
    num_of_horizons = int((max_horizon/15).seconds/60)
    sys_vars = {'dx': dx, 'dy': dy,
                'num_of_horizons': num_of_horizons,
                'max_horizon': max_horizon,
                'ci_crop_shape': ci_crop_shape,
                'U_crop_shape': U_crop_shape,
                'V_crop_shape': V_crop_shape,
                'U_crop_size': U_crop_size,
                'V_crop_size': V_crop_size,
                'wind_size': wind_size}
    if flags['div']:
        mesh = fe.RectangleMesh(fe.Point(0, 0),
                                fe.Point(int(V_crop_shape[1] - 1),
                                         int(U_crop_shape[0] - 1)),
                                int(V_crop_shape[1] - 1),
                                int(U_crop_shape[0] - 1))
        FunctionSpace_wind = fe.FunctionSpace(mesh, 'P', 1)
        sys_vars['FunctionSpace_wind'] = FunctionSpace_wind
    if flags['perturbation']:
        rf_eig, rf_vectors = eig_2d_covariance(
            x=coords.we_crop, y=coords.sn_crop,
            Lx=pert_params['Lx'],
            Ly=pert_params['Ly'], tol=pert_params['tol'])
        rf_approx_var = (
            rf_vectors * rf_eig[None, :] * rf_vectors).sum(-1).mean()
        sys_vars['rf_eig'] = rf_eig
        sys_vars['rf_vectors'] = rf_vectors
        sys_vars['rf_approx_var'] = rf_approx_var
    if flags['perturb_winds']:
        rf_eig, rf_vectors = eig_2d_covariance(
            coords.we_crop, coords.sn_crop,
            Lx=pert_params['Lx_wind'],
            Ly=pert_params['Ly_wind'], tol=pert_params['tol_wind'])
        rf_approx_var = (
            rf_vectors * rf_eig[None, :] * rf_vectors).sum(-1).mean()
        rf_eig = rf_eig*pert_params['Lx_wind']**2
        sys_vars['rf_eig_wind'] = rf_eig
        sys_vars['rf_vectors_wind'] = rf_vectors
        sys_vars['rf_approx_var_wind'] = rf_approx_var
    sys_vars = dict2nt(sys_vars, 'sys_vars')
    return sys_vars


def calc_assim_variables(*, sys_vars, advect_params, flags, sat2sat, sat2wind,
                         wrf):
    assim_vars = {}
    if flags['assim_sat2sat']:
        assim_pos, assim_pos_2d, full_pos_2d = (
            assimilation_position_generator(sys_vars.ci_crop_shape,
                                            sat2sat['grid_size']))
        noise_init = noise_fun(sys_vars.ci_crop_shape)
        assim_vars['assim_pos'] = assim_pos
        assim_vars['assim_pos_2d'] = assim_pos_2d
        assim_vars['full_pos_2d'] = full_pos_2d
        assim_vars['noise_init'] = noise_init
    if flags['assim_sat2wind']:
        assim_pos_sat2wind, assim_pos_2d_sat2wind, full_pos_2d_sat2wind = (
            assimilation_position_generator(sys_vars.ci_crop_shape,
                                            sat2wind['grid_size']))
        assim_vars['assim_pos_sat2wind'] = assim_pos_sat2wind
        assim_vars['assim_pos_2d_sat2wind'] = assim_pos_2d_sat2wind
        assim_vars['full_pos_2d_sat2wind'] = full_pos_2d_sat2wind
        assim_pos_U, assim_pos_2d_U, full_pos_2d_U = (
            assimilation_position_generator(sys_vars.U_crop_shape,
                                            sat2wind['grid_size']))
        assim_pos_V, assim_pos_2d_V, full_pos_2d_V = (
            assimilation_position_generator(sys_vars.V_crop_shape,
                                            sat2wind['grid_size']))
        assim_vars['assim_pos_U'] = assim_pos_U
        assim_vars['assim_pos_2d_U'] = assim_pos_2d_U
        assim_vars['full_pos_2d_U'] = full_pos_2d_U
        assim_vars['assim_pos_V'] = assim_pos_V
        assim_vars['assim_pos_2d_V'] = assim_pos_2d_V
        assim_vars['full_pos_2d_V'] = full_pos_2d_V
    if flags['assim_wrf']:
        assim_pos_U_wrf, assim_pos_2d_U_wrf, full_pos_2d_U_wrf = (
            assimilation_position_generator(sys_vars.U_crop_shape,
                                            wrf['grid_size']))
        assim_pos_V_wrf, assim_pos_2d_V_wrf, full_pos_2d_V_wrf = (
            assimilation_position_generator(sys_vars.V_crop_shape,
                                            wrf['grid_size']))
        assim_vars['assim_pos_U_wrf'] = assim_pos_U_wrf
        assim_vars['assim_pos_2d_U_wrf'] = assim_pos_2d_U_wrf
        assim_vars['full_pos_2d_U_wrf'] = full_pos_2d_U_wrf
        assim_vars['assim_pos_V_wrf'] = assim_pos_V_wrf
        assim_vars['assim_pos_2d_V_wrf'] = assim_pos_2d_V_wrf
        assim_vars['full_pos_2d_V_wrf'] = full_pos_2d_V_wrf
    assim_vars = dict2nt(assim_vars, 'assim_vars')
    return assim_vars


def return_wind_time(*, sat_time, coords):
    int_index_wind = coords.wind_times.get_loc(sat_time,
                                               method='pad')
    wind_time = coords.wind_times[int_index_wind]
    return wind_time


def return_ensemble(*, data_file_path, ens_params, coords, flags):
    sat_time = coords.sat_times[0]
    wind_time = return_wind_time(sat_time=sat_time, coords=coords)
    q = return_single_time(data_file_path, coords.sat_times_all,
                           sat_time, [coords.sn_slice],
                           [coords.we_slice], ['ci'])[0]
    if flags['radiosonde']:
        station = 'TUS'
        df = WyomingUpperAir.request_data(sat_time.date(), station)
        T = df['temperature'].values * units(df.units['temperature'])
        Td = df['dewpoint'].values * units(df.units['dewpoint'])
        u_wind = df['u_wind'].values * units(df.units['u_wind'])
        u_wind = u_wind.to(units.meter/units.second)
        v_wind = df['v_wind'].values * units(df.units['v_wind'])
        v_wind = v_wind.to(units.meter/units.second)
        rh = thermo.relative_humidity_from_dewpoint(T, Td)
        max_arg = np.argmax(rh)
        u_size = coords.we_stag_crop.size * coords.sn_crop.size
        v_size = coords.we_crop.size * coords.sn_stag_crop.size
        U = np.ones(u_size)*u_wind[max_arg]
        V = np.ones(v_size)*v_wind[max_arg]
    elif flags['opt_flow']:
        opt_flow_folder = os.path.split(data_file_path)[0]
        opt_flow_file = os.path.join(opt_flow_folder, 'data_opt_flow.nc')
        of_sat_time = coords.sat_times[1]
        U, V = return_single_time(opt_flow_file, coords.sat_times_all,
                                  of_sat_time,
                                  [coords.sn_slice, coords.sn_stag_slice],
                                  [coords.we_stag_slice, coords.we_slice],
                                  ['U_opt_flow', 'V_opt_flow'])

        time_step = (of_sat_time - sat_time).seconds
        U = U * (250 / time_step)
        V = V * (250 / time_step)
        U = U.clip(min=-50, max=50)
        V = V.clip(min=-50, max=50)
    else:
        U, V = return_single_time(data_file_path, coords.wind_times, wind_time,
                                  [coords.sn_slice, coords.sn_stag_slice],
                                  [coords.we_stag_slice, coords.we_slice],
                                  ['U', 'V'])
        U, V = smooth_winds(U, V)
        if flags['wrf_mean']:
            U = np.ones_like(U)*U.mean()
            V = np.ones_like(V)*V.mean()
    if flags['assim']:
        ensemble = ensemble_creator(
            q, U, V, CI_sigma=ens_params['ci_sigma'],
            wind_sigma=ens_params['winds_sigma'],
            ens_size=ens_params['ens_num'])
    else:
        ensemble = np.concatenate([U.ravel(), V.ravel(), q.ravel()])[:, None]
    shape = ensemble.shape
    ensemble = np.ma.compressed(ensemble).reshape(shape)
    return ensemble


def forecast_setup(*, data_file_path, date, io, advect_params, ens_params,
                   pert_params, flags, sat2sat, sat2wind, wrf, opt_flow,
                   results_file_path):
    param_dict = set_up_param_dict(
        date=date, io=io, advect_params=advect_params, ens_params=ens_params,
        pert_params=pert_params, flags=flags, sat2sat=sat2sat,
        sat2wind=sat2wind, wrf=wrf, opt_flow=opt_flow)
    coords = read_coords(data_file_path=data_file_path,
                         advect_params=advect_params, flags=flags)
    sys_vars = calc_system_variables(
        coords=coords, advect_params=advect_params, flags=flags,
        pert_params=pert_params)
    if 'analysis_fore' in flags:
        if flags['analysis_fore']:
            sat_time = coords.sat_times[0]
            ensemble = return_analysis_ensemble(
                sat_time=sat_time, results_file_path=results_file_path)
        else:
            ensemble = return_ensemble(data_file_path=data_file_path,
                                       ens_params=ens_params,
                                       coords=coords, flags=flags)
    else:
        ensemble = return_ensemble(data_file_path=data_file_path,
                                   ens_params=ens_params,
                                   coords=coords, flags=flags)
    if flags['assim']:
        assim_vars = calc_assim_variables(sys_vars=sys_vars,
                                          advect_params=advect_params,
                                          flags=flags, sat2sat=sat2sat,
                                          sat2wind=sat2wind, wrf=wrf)
    else:
        assim_vars = None
    return param_dict, coords, sys_vars, assim_vars, ensemble


def preprocess(*, ensemble, flags, remove_div_flag, coords, sys_vars):
    if remove_div_flag and flags['div']:
        logging.debug('remove divergence')
        ensemble[:sys_vars.wind_size] = remove_divergence_ensemble(
            FunctionSpace=sys_vars.FunctionSpace_wind,
            wind_ensemble=ensemble[:sys_vars.wind_size],
            U_crop_shape=sys_vars.U_crop_shape,
            V_crop_shape=sys_vars.V_crop_shape, sigma=4)
    return ensemble


def forecast(*, ensemble, flags, coords, time_index, sat_time,
             sys_vars, advect_params, pert_params, assim_vars, workers):
    save_times = pd.date_range(sat_time,
                               periods=(sys_vars.num_of_horizons + 1),
                               freq='15min')
    save_times = save_times.tz_convert(None)
    if time_index + 1 < coords.sat_times.size:
        num_of_advect = int((
            coords.sat_times[time_index + 1] -
            coords.sat_times[time_index]).seconds/(60*15))
    else:
        num_of_advect = 0
    background = None
    logging.debug(f'15min steps to background: {num_of_advect}')
    ensemble_array = ensemble.copy()[None, :, :]
    cx = abs(ensemble[:sys_vars.U_crop_size]).max()
    cy = abs(ensemble[sys_vars.U_crop_size:
                      sys_vars.wind_size]).max()
    T_steps = int(np.ceil((5*60)*(cx/sys_vars.dx
                                  + cy/sys_vars.dy)
                          / advect_params['C_max']))
    dt = (5*60)/T_steps
    for m in range(sys_vars.num_of_horizons):
        logging.info(str(pd.Timedelta('15min')*(m + 1)))
        for n in range(3):
            if flags['perturb_winds']:
                ensemble[:sys_vars.wind_size] = perturb_winds(
                    ensemble[:sys_vars.wind_size], sys_vars, pert_params)

            if flags['assim']:
                ensemble = advect_5min_ensemble(
                    ensemble, dt, sys_vars.dx, sys_vars.dy,
                    T_steps,
                    sys_vars.U_crop_shape, sys_vars.V_crop_shape,
                    sys_vars.ci_crop_shape, workers)
            else:
                ensemble[:, 0] = advect_5min_single(
                    ensemble[:, 0], dt, sys_vars.dx, sys_vars.dy,
                    T_steps,
                    sys_vars.U_crop_shape, sys_vars.V_crop_shape,
                    sys_vars.ci_crop_shape)
            ensemble[sys_vars.wind_size:] = (ensemble[sys_vars.wind_size:]
                                             .clip(min=0, max=1))
            if flags['perturbation']:
                ensemble[sys_vars.wind_size:] = perturb_irradiance(
                    ensemble[sys_vars.wind_size:], sys_vars.ci_crop_shape,
                    pert_params['edge_weight'],
                    pert_params['pert_mean'],
                    pert_params['pert_sigma'],
                    sys_vars.rf_approx_var,
                    sys_vars.rf_eig, sys_vars.rf_vectors)
        ensemble_array = np.concatenate(
            [ensemble_array, ensemble[None, :, :]],
            axis=0)
        if num_of_advect == (m + 1):
            background = ensemble.copy()
    return ensemble_array, save_times, background


def save(*, ensemble_array, coords, ens_params, param_dict, sys_vars,
         save_times, results_file_path, flags):
    U, V, ci = extract_components(
        ensemble_array, ens_params['ens_num'], sys_vars.num_of_horizons + 1,
        sys_vars.U_crop_shape, sys_vars.V_crop_shape, sys_vars.ci_crop_shape)
    save_netcdf(
        results_file_path, U, V, ci, param_dict,
        coords.we_crop, coords.sn_crop,
        coords.we_stag_crop, coords.sn_stag_crop,
        save_times, ens_params['ens_num'], flags)


def maybe_assim_sat2sat(*, ensemble, data_file_path, sat_time,
                        coords, sys_vars, flags):
    if 'analysis_fore' in flags:
        if flags['analysis_fore']:
            return ensemble
    if flags['assim_sat2sat']:
        raise NotImplementedError
    else:
        q = return_single_time(data_file_path, coords.sat_times_all,
                               sat_time, [coords.sn_slice], [coords.we_slice],
                               ['ci'])[0]
        ensemble[sys_vars.wind_size:] = q.ravel()[:, None]
    return ensemble


def maybe_assim_sat2wind(*, ensemble, data_file_path, sat_time,
                         coords, sys_vars, assim_vars, sat2wind,
                         flags):
    if 'analysis_fore' in flags:
        if flags['analysis_fore']:
            return ensemble, False
    if flags['assim_sat2wind']:
        logging.debug('Assim sat2wind')
        q = return_single_time(data_file_path, coords.sat_times_all,
                               sat_time, [coords.sn_slice], [coords.we_slice],
                               ['ci'])[0]
        ensemble = assimilate_sat_to_wind(
            ensemble=ensemble,
            observations=q.ravel(),
            R_inverse_wind=1/sat2wind['sig']**2,
            wind_inflation=sat2wind['infl'],
            domain_shape=sys_vars.ci_crop_shape,
            U_shape=sys_vars.U_crop_shape,
            V_shape=sys_vars.V_crop_shape,
            localization_length_wind=sat2wind['loc'],
            assimilation_positions=assim_vars.assim_pos_sat2wind,
            assimilation_positions_2d=assim_vars.assim_pos_2d_sat2wind,
            full_positions_2d=assim_vars.full_pos_2d_sat2wind)
        div_sat2wind_flag = True
    else:
        div_sat2wind_flag = False
    return ensemble, div_sat2wind_flag


def maybe_assim_wrf(*, ensemble, data_file_path, sat_time,
                    coords, sys_vars, assim_vars, wrf,
                    ens_params, flags):
    if 'analysis_fore' in flags:
        if flags['analysis_fore']:
            return ensemble, False
    wind_time = return_wind_time(sat_time=sat_time, coords=coords)
    if sat_time == wind_time and not flags['radiosonde']:
        U, V = return_single_time(data_file_path, coords.wind_times,
                                  wind_time,
                                  [coords.sn_slice, coords.sn_stag_slice],
                                  [coords.we_stag_slice, coords.we_slice],
                                  ['U', 'V'])
        U, V = smooth_winds(U, V)
        if flags['wrf_mean']:
            U = np.ones_like(U)*U.mean()
            V = np.ones_like(V)*V.mean()
        div_wrf_flag = True
        if flags['assim_wrf']:
            logging.debug('Assim WRF')
            R_inverse = 1/wrf['sig']**2
            localization_length_wind = int(round(
                wrf['loc'] / (coords.we[1] - coords.we[0])))
            ensemble[:sys_vars.U_crop_size] = assimilate_wrf(
                ensemble=ensemble[:sys_vars.U_crop_size],
                observations=U.ravel(),
                R_inverse=R_inverse,
                wind_inflation=wrf['infl'],
                wind_shape=sys_vars.U_crop_shape,
                localization_length_wind=localization_length_wind,
                assimilation_positions=assim_vars.assim_pos_U_wrf,
                assimilation_positions_2d=assim_vars.assim_pos_2d_U_wrf,
                full_positions_2d=assim_vars.full_pos_2d_U_wrf)

            ensemble[sys_vars.U_crop_size:sys_vars.wind_size] = assimilate_wrf(
                ensemble=ensemble[sys_vars.U_crop_size:
                                  sys_vars.wind_size],
                observations=V.ravel(),
                R_inverse=R_inverse,
                wind_inflation=wrf['infl'],
                wind_shape=sys_vars.V_crop_shape,
                localization_length_wind=localization_length_wind,
                assimilation_positions=assim_vars.assim_pos_V_wrf,
                assimilation_positions_2d=assim_vars.assim_pos_2d_V_wrf,
                full_positions_2d=assim_vars.full_pos_2d_V_wrf)
        elif not flags['opt_flow']:
            logging.debug('replace WRF')
            if ensemble.shape[1] > 1:
                random_nums = np.random.normal(
                    loc=0,
                    scale=ens_params['winds_sigma'][0],
                    size=ens_params['ens_num'])
                ensemble[:sys_vars.U_crop_size] = (U.ravel()[:, None]
                                                   + random_nums[None, :])
                random_nums = np.random.normal(
                    loc=0,
                    scale=ens_params['winds_sigma'][1],
                    size=ens_params['ens_num'])
                ensemble[sys_vars.U_crop_size:
                         sys_vars.wind_size] = (
                             V.ravel()[:, None]
                             + random_nums[None, :])
            else:
                ensemble[:sys_vars.U_crop_size] = U.ravel()[:, None]
                ensemble[sys_vars.U_crop_size:
                         sys_vars.wind_size] = V.ravel()[:, None]
    else:
        div_wrf_flag = False
    return ensemble, div_wrf_flag


def smooth_winds(U, V):
    U = sp.ndimage.filters.gaussian_filter(U, sigma=60)
    V = sp.ndimage.filters.gaussian_filter(V, sigma=60)
    return U, V


def return_opt_flow(*, coords, time_index, sat_time, data_file_path, sys_vars):
    # retreive OPT_FLOW vectors
    wind_time = return_wind_time(sat_time=sat_time, coords=coords)
    time0 = coords.sat_times[time_index - 1]
    this_U, this_V = return_single_time(data_file_path, coords.wind_times,
                                        wind_time,
                                        [slice(None), slice(None)],
                                        [slice(None), slice(None)],
                                        ['U', 'V'])
    image0 = return_single_time(data_file_path, coords.sat_times_all,
                                time0, [slice(None)], [slice(None)],
                                ['ci'])[0]
    image1 = return_single_time(data_file_path, coords.sat_times_all,
                                sat_time, [slice(None)], [slice(None)],
                                ['ci'])[0]
    u_opt_flow, v_opt_flow, pos = optical_flow(image0, image1,
                                               time0, sat_time,
                                               this_U, this_V)
    if u_opt_flow.size == 0:
        nothing = np.array([])
        return nothing, nothing, nothing, nothing
    del this_U, this_V, image0, image1
    pos = pos*4  # optical flow done on coarse grid

    # need to select only pos in crop domain; convert to crop
    keep = np.logical_and(
        np.logical_and(pos[:, 0] > coords.we_slice.start,
                       pos[:, 0] < coords.we_slice.stop),
        np.logical_and(pos[:, 1] > coords.sn_slice.start,
                       pos[:, 1] < coords.sn_slice.stop))
    pos = pos[keep]
    u_opt_flow = u_opt_flow[keep]
    v_opt_flow = v_opt_flow[keep]
    pos[:, 0] -= coords.we_slice.start
    pos[:, 1] -= coords.sn_slice.start
    pos = pos.T
    pos = pos[::-1]
    u_opt_flow_flat_pos = np.ravel_multi_index(pos, sys_vars.U_crop_shape)
    v_opt_flow_flat_pos = np.ravel_multi_index(pos, sys_vars.V_crop_shape)
    return u_opt_flow, v_opt_flow, u_opt_flow_flat_pos, v_opt_flow_flat_pos


def maybe_assim_opt_flow(*, ensemble, data_file_path, sat_time, time_index,
                         coords, sys_vars, flags, opt_flow):
    if flags['assim_opt_flow']:
        div_opt_flow_flag = True
        logging.debug('calc opt_flow')
        returned = return_opt_flow(
            coords=coords, time_index=time_index, sat_time=sat_time,
            data_file_path=data_file_path, sys_vars=sys_vars)
        u_opt_flow, v_opt_flow = returned[:2]
        u_opt_flow_flat_pos, v_opt_flow_flat_pos = returned[2:]
        if u_opt_flow.size == 0:
            div_opt_flow_flag = False
            to_return = (ensemble, div_opt_flow_flag)
            return to_return
        logging.debug('assim opt_flow')
        x_temp = np.arange(sys_vars.U_crop_shape[1])*sys_vars.dx/1000  # in km
        y_temp = np.arange(sys_vars.U_crop_shape[0])*sys_vars.dy/1000
        x_temp, y_temp = np.meshgrid(x_temp, y_temp)
        ensemble[:sys_vars.U_crop_size] = reduced_enkf(
            ensemble=ensemble[:sys_vars.U_crop_size],
            observations=u_opt_flow, R_sig=opt_flow['sig'],
            flat_locations=u_opt_flow_flat_pos,
            inflation=opt_flow['infl'],
            localization=opt_flow['loc'],
            x=x_temp.ravel(), y=y_temp.ravel())
        x_temp = np.arange(sys_vars.V_crop_shape[1])*sys_vars.dx/1000
        y_temp = np.arange(sys_vars.V_crop_shape[0])*sys_vars.dy/1000
        x_temp, y_temp = np.meshgrid(x_temp, y_temp)
        ensemble[sys_vars.U_crop_size:
                 sys_vars.wind_size] = reduced_enkf(
                     ensemble=ensemble[sys_vars.U_crop_size:
                                       sys_vars.wind_size],
                     observations=v_opt_flow, R_sig=opt_flow['sig'],
                     flat_locations=v_opt_flow_flat_pos,
                     inflation=opt_flow['infl'],
                     localization=opt_flow['loc'],
                     x=x_temp.ravel(), y=y_temp.ravel())
        to_return = (ensemble, div_opt_flow_flag, u_opt_flow, v_opt_flow,
                     u_opt_flow_flat_pos, v_opt_flow_flat_pos)
    elif flags['opt_flow']:
        div_opt_flow_flag = True
        opt_flow_folder = os.path.split(data_file_path)[0]
        opt_flow_file = os.path.join(opt_flow_folder, 'data_opt_flow.nc')
        U, V = return_single_time(opt_flow_file, coords.sat_times_all,
                                  sat_time,
                                  [coords.sn_slice, coords.sn_stag_slice],
                                  [coords.we_stag_slice, coords.we_slice],
                                  ['U_opt_flow', 'V_opt_flow'])
        time_step = (sat_time - coords.sat_times[time_index - 1]).seconds
        U = U * (250 / time_step)
        V = V * (250 / time_step)
        U = U.clip(min=-50, max=50)
        V = V.clip(min=-50, max=50)
        ensemble[:sys_vars.U_crop_size] = U.ravel()[:, None]
        ensemble[sys_vars.U_crop_size:
                 sys_vars.wind_size] = V.ravel()[:, None]
    else:
        div_opt_flow_flag = False
    to_return = (ensemble, div_opt_flow_flag)
    return to_return


def maybe_load_analysis(*, sat_time, results_file_path, flags,
                        ensemble):
    if 'analysis_fore' in flags:
        if flags['analysis_fore']:
            ensemble = return_analysis_ensemble(
                sat_time=sat_time, results_file_path=results_file_path)
            return ensemble
        else:
            return ensemble
    else:
        return ensemble


def forecast_system(*, data_file_path, results_file_path,
                    date, io, flags, advect_params, ens_params, pert_params,
                    sat2sat, sat2wind, wrf, opt_flow, workers):
    param_dict, coords, sys_vars, assim_vars, ensemble = forecast_setup(
        data_file_path=data_file_path, date=date, io=io,
        flags=flags, advect_params=advect_params,
        ens_params=ens_params, pert_params=pert_params,
        sat2sat=sat2sat, sat2wind=sat2wind, wrf=wrf,
        opt_flow=opt_flow, results_file_path=results_file_path)
    remove_div_flag = True
    ensemble = preprocess(
        ensemble=ensemble, flags=flags,
        remove_div_flag=remove_div_flag,
        coords=coords, sys_vars=sys_vars)
    time_index = 0
    sat_time = coords.sat_times[time_index]
    ensemble_array, save_times, ensemble = forecast(
        ensemble=ensemble, sat_time=sat_time,
        flags=flags, coords=coords, time_index=time_index,
        sys_vars=sys_vars,
        advect_params=advect_params, pert_params=pert_params,
        assim_vars=assim_vars, workers=workers)
    save(ensemble_array=ensemble_array, coords=coords,
         ens_params=ens_params, param_dict=param_dict,
         sys_vars=sys_vars, save_times=save_times,
         results_file_path=results_file_path,
         flags=flags)
    for time_index in range(1, coords.sat_times.size):
        sat_time = coords.sat_times[time_index]
        logging.info(str(sat_time))
        ensemble = maybe_load_analysis(
            sat_time=sat_time, flags=flags,
            ensemble=ensemble, results_file_path=results_file_path)
        ensemble = maybe_assim_sat2sat(
            ensemble=ensemble, data_file_path=data_file_path,
            sat_time=sat_time, coords=coords, sys_vars=sys_vars,
            flags=flags)
        ensemble, div_sat2wind_flag = maybe_assim_sat2wind(
            ensemble=ensemble, data_file_path=data_file_path,
            sat_time=sat_time, coords=coords, sys_vars=sys_vars,
            assim_vars=assim_vars, sat2wind=sat2wind,
            flags=flags)
        ensmeble, div_wrf_flag = maybe_assim_wrf(
            ensemble=ensemble, data_file_path=data_file_path,
            sat_time=sat_time, coords=coords, sys_vars=sys_vars,
            assim_vars=assim_vars, wrf=wrf,
            ens_params=ens_params,
            flags=flags)
        ensemble, div_opt_flow_flag = maybe_assim_opt_flow(
            ensemble=ensemble, data_file_path=data_file_path,
            sat_time=sat_time, time_index=time_index,
            coords=coords, sys_vars=sys_vars,
            flags=flags, opt_flow=opt_flow)
        remove_div_flag = (div_sat2wind_flag
                           or div_wrf_flag
                           or div_opt_flow_flag)
        ensemble = preprocess(
            ensemble=ensemble, flags=flags,
            remove_div_flag=remove_div_flag,
            coords=coords, sys_vars=sys_vars)
        ensemble_array, save_times, ensemble = forecast(
            ensemble=ensemble, sat_time=sat_time,
            flags=flags, coords=coords, time_index=time_index,
            sys_vars=sys_vars,
            advect_params=advect_params, pert_params=pert_params,
            assim_vars=assim_vars, workers=workers)
        save(ensemble_array=ensemble_array, coords=coords,
             ens_params=ens_params, param_dict=param_dict,
             sys_vars=sys_vars, save_times=save_times,
             results_file_path=results_file_path,
             flags=flags)
