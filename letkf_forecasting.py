import numpy as np
import pandas as pd
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


# To DO:
# Make module for preparing satellite data and cleaning up sensor data.

a = 6371000  # average radius of earth when modeled as a sphere From wikipedia


def time_deriv_3(q, dt, u, dx, v, dy):
    k = space_deriv_4(q, u, dx, v, dy)
    k = space_deriv_4(q + dt/3*k, u, dx, v, dy)
    k = space_deriv_4(q + dt/2*k, u, dx, v, dy)
    qout = q + dt*k
    return qout


def space_deriv_4(q, u, dx, v, dy):
    qout = np.zeros_like(q)
    F_x = np.zeros_like(u)
    F_y = np.zeros_like(v)

    # middle calculation
    F_x[:, 2:-2] = u[:, 2:-2]/12*(
        7*(q[:, 2:-1] + q[:, 1:-2]) - (q[:, 3:] + q[:, :-3]))
    F_y[2:-2, :] = v[2:-2, :]/12*(
        7*(q[2:-1, :] + q[1:-2, :]) - (q[3:, :] + q[:-3, :]))
    qout[:, 2:-2] = qout[:, 2:-2] - (F_x[:, 3:-2] - F_x[:, 2:-3])/dx
    qout[2:-2, :] = qout[2:-2, :] - (F_y[3:-2, :] - F_y[2:-3, :])/dy

    # boundary calculation
    u_w = u[:, 0:2].clip(max=0)
    u_e = u[:, -2:].clip(min=0)
    qout[:, 0:2] = qout[:, 0:2] - ((u_w/dx)*(
        q[:, 1:3] - q[:, 0:2]) + (q[:, 0:2]/dx)*(u[:, 1:3] - u[:, 0:2]))
    qout[:, -2:] = qout[:, -2:] - ((u_e/dx)*(
        q[:, -2:] - q[:, -3:-1]) + (q[:, -2:]/dx)*(u[:, -2:] - u[:, -3:-1]))

    v_n = v[-2:, :].clip(min=0)
    v_s = v[0:2, :].clip(max=0)
    qout[0:2, :] = qout[0:2, :] - ((v_s/dx)*(
        q[1:3, :] - q[0:2, :]) + (q[0:2, :]/dx)*(v[1:3, :] - v[0:2, :]))
    qout[-2:, :] = qout[-2:, :] - ((v_n/dx)*(
        q[-2:, :] - q[-3:-1, :]) + (q[-2:, :]/dx)*(v[-2:, :] - v[-3:-1, :]))

    return qout


def cot(theta):
    """Why doesn't numpy have cot?"""
    return np.cos(theta)/np.sin(theta)


def parallax_shift(cloud_height,
                   satellite_azimuth,
                   satellite_elevation,
                   solar_azimuth,
                   solar_elevation):
    """Returns x and y shift required to match satellite pixel to earth location
    based on satellite and solar position. Units of x and y correction will be
    in the units of cloud_height.

    Parameters
    ----------
    cloud_hieght : float
         Height of cloud.

    satellite_azimuth : float
         Azimuth angle of satellite in radians.

    satellite_altitude : float
         Altitude angle of satellite in radians.

    solar_azimuth : float
         Azimuth angle of the sun in radians.

    solar_altitude : float
         Altitude angle of the sun in radians.

    Returns
    -------
    x_correction, y_correction : float
         x_correction and y_correction are the values which must be added to
         the satellite position to find actual position of cloud shadow.
    """
    satellite_displacement = cloud_height*cot(satellite_elevation)
    solar_displacement = cloud_height*cot(solar_elevation)
    x_correction = (
        solar_displacement*np.cos(np.pi/2 - solar_azimuth) -
        satellite_displacement*np.cos(np.pi/2 - satellite_azimuth))
    y_correction = (
        solar_displacement*np.sin(np.pi/2 - solar_azimuth) -
        satellite_displacement*np.sin(np.pi/2 - satellite_azimuth))

    return x_correction, y_correction


def forward_obs_mat(sensor_loc, sat_loc):
    """Returns the forward observation matrix H which maps sat locations to
    sensor locations.

    Parameters
    ----------
    sensor_loc : array
         A kx2 array where k is the number of sensors and each row is the
         position of the sensor.

    sat_loc : array
         A nx2 array where n is the number of elements in the domain and each
         row is the position of an element.

    Returns
    -------
    H : array
         A kxn forward observation matrix which maps sensor locations to
         satellite locations.
    sensor_loc : array
         The same as the inputed sensor_loc with an additional third column
         which is the index number of the domain corresponding to the row
         location.
    """
    sensor_num = sensor_loc.shape[0]
    domain_size = sat_loc.shape[0]
    sensor_loc = np.concatenate((sensor_loc, np.zeros(sensor_num)[:, None]),
                                axis=1)
    H = np.zeros([sensor_num, domain_size])
    for id in range(0, sensor_num):
        index = np.sqrt(
            (sat_loc[:, 0] - sensor_loc[id, 0])**2
            + (sat_loc[:, 1] - sensor_loc[id, 1])**2).argmin()
        sensor_loc[id, 2] = index
        H[id, index] = 1

    return H, sensor_loc


def to_lat_lon(x, y, loc_lat, loc_lon):
    """Converts a displacement in meters to a displacement in degrees.

    Parameters
    ----------
    x : float
         Displacement in meters in east west direction.
    y : float
         Displacement in meters in north south direction.
    loc_lat : float
         Latitude for location.
    loc_lon : float
         Longitude for location.

    Returns
    -------
    lat, lon : float
         Displacement converted to degrees.
    """
    lon = x*360/(2*np.pi*a*np.cos(loc_lat))
    lat = y*360/(2*np.pi*a)
    return lat, lon


def nearest_positions(loc, shape, dist):
    """Returns the indices of a vector which are dist distance from loc in
    either the x or y direction when that vector is unraveled given shape.

    Parameters
    ----------
    loc : int
         The index of the raveled vector.
    shape : (int, int)
         The shape of the unraveled array. Currently assumed to be square.
    dist : int
         The distance which can be traveled in x or y in the unraveled array.

    Returns
    -------
    near_positions : array
         Array of indices for the raveled vector near loc.
    """


    # the shape has to be square
    position = np.unravel_index(loc, shape)
    row_min = (position[0] - dist).clip(min=0)
    row_max = (position[0] + dist).clip(max=(shape[0] - 1))
    col_min = (position[1] - dist).clip(min=0)
    col_max = (position[1] + dist).clip(max=(shape[1] - 1))
    row_positions, col_positions = np.meshgrid(np.arange(row_min, row_max + 1),
                                               np.arange(col_min, col_max + 1))
    row_positions = np.ravel(row_positions)
    col_positions = np.ravel(col_positions)
    near_positions = np.ravel_multi_index((row_positions, col_positions),
                                          shape)
    near_positions.sort()
    return near_positions


def assimilate(ensemble, observations, H, R_inverse, inflation, shape=False,
               localization_length=False, assimilation_positions=False,
               assimilation_positions_2d=False, full_positions_2d=False):
    """
    *** NEED TO REWRITE
    Assimilates observations into ensemble using the LETKF.

    Parameters
    ----------
    ensemble : array
         The ensemble of size kxn where k is the number of ensemble members
         and n is the state vector size.
    observations : array
         An observation vector of length m.
    H : array
         Forward observation matrix of size mxn. **may need changing**
    R_inverse : array
         Inverse of observation error matrix. **will need changing**
    inflation : float
         Inflation parameter.
    localization_length : float
         Localization distance in each direction so that assimilation will take
         on (2*localization + 1)**2 elements. If equal to False then no
         localization will take place.
    assimilation_positions : array
         Row and column index of state domain over which assimilation will
         take place. First column contains row positions, second column
         contains column positions, total number of rows is number of
         assimilations. If False the assimilation will take place over
         full_positions. If localization_length is False then this variable
         will not be used.
    full_positions : array
         Array similar to assimilation_positions including the positions of
         all elements of the state.

    Return
    ------
    ensemble : array
         Analysis ensemble of the same size as input ensemble
    """
    ## Change to allow for R to not be pre-inverted?
    # x_bar = ensemble.mean(axis=1)
    # ensemble -= x_bar[:, None]
    # ens_size = ensemble.shape[1]
    # print(ens_size)

    # if localization_length is False:
    #     # LETKF without localization
    #     Y_b = np.einsum('ij,jk...->ik...', H, ensemble)
    #     y_b_bar = Y_b.mean(axis=1)
    #     Y_b -= y_b_bar[:, np.newaxis]
    #     C = (Y_b.T).dot(R_inverse)
    #     eig_value, eig_vector = np.linalg.eigh(
    #         (ens_size-1)*np.eye(ens_size)/inflation + C.dot(Y_b))
    #     P_tilde = eig_vector.copy()
    #     W_a = eig_vector.copy()
    #     for i, num in enumerate(eig_value):
    #         P_tilde[:, i] *= 1/num
    #         W_a[:, i] *= 1/np.sqrt(num)
    #     P_tilde = P_tilde.dot(eig_vector.T)
    #     W_a = W_a.dot(eig_vector.T)
    #     w_a_bar = P_tilde.dot(C.dot(observations - y_b_bar))
    #     W_a += w_a_bar[:, None]
    #     ensemble = x_bar[:, None] + ensemble.dot(W_a)
    #     return ensemble

    # else:
    #     # LETKF with localization assumes H is I
    #     ## NEED: to include wind in ensemble will require reworking due to
    #     ## new H and different localization.
    #     ## NEED: Change to include some form of H for paralax correction??
    #     ## Maybe: ^ not if paralax is only corrected when moving to ground sensors.
    #     ## SHOULD: Will currently write as though R_inverse is a scalar.
    #     ## May need to change at some point but will likely need to do
    #     ## something clever since R_inverse.size is 400 billion
    #     ## best option: form R_inverse inside of localization routine
    #     ## good option: assimilate sat images at low resolution (probabily should do this either way)
    #     kal_count = 0
    #     W_interp = np.zeros([assimilation_positions.size, ens_size**2])
    #     for interp_position in assimilation_positions:
    #         local_positions = nearest_positions(interp_position, shape,
    #                                             localization_length)
    #         local_ensemble = ensemble[local_positions]
    #         local_x_bar = x_bar[local_positions]
    #         local_obs = observations[local_positions] # assume H is I
    #         C = (local_ensemble.T)*R_inverse  # assume R_inverse is diag+const

    #         # This should be better, but I can't get it to work
    #         # eig_value, eig_vector = np.linalg.eigh(
    #         #     (ens_size-1)*np.eye(ens_size)/inflation + C.dot(local_ensemble))
    #         # P_tilde = eig_vector.copy()
    #         # W_a = eig_vector.copy()
    #         # for i, num in enumerate(eig_value):#             x_bar = ensemble.mean(axis=1)
    #             ensemble -= x_bar[:, np.newaxis] # Y_b is ensemble[wind_size::, :]
    #             ## localization with interpolation
    #             kal_count = 0
    #             for interp_position in kalman_positions:
    #                 local_positions = nearest_positions(interp_position, shape, dist)
    #                 local_ensemble = ensemble[local_positions + wind_size]
    #                 local_x_bar = x_bar[local_positions + wind_size]
    #                 local_obs = y_obs[local_positions]
    #                 local_noise = flat_noise[local_positions]
    #                 C = (local_ensemble.T)/(sig_obs**2)
    # #                 C = C*local_noise[None, :]
    # #                 P_a = np.linalg.inv((ens_size - 1)*np.eye(ens_size)/flat_noise[interp_position] +
    # #                                     C.dot(local_ensemble)) # with noise inflation
    #                 P_a = np.linalg.inv((ens_size - 1)*np.eye(ens_size)/inflation +
    #                                     C.dot(local_ensemble)) # withOUT noise inflation
    #                 W_a = np.real(scipy.linalg.sqrtm((ens_size - 1)*P_a))
    #                 w_a_bar = P_a.dot(C.dot(local_obs - local_x_bar))
    #                 W_a += w_a_bar[:, None]
    #                 W_interp[kal_count] = np.ravel(W_a) ## separate w_bar??
    #                 kal_count += 1

    #             W_fun = scipy.interpolate.LinearNDInterpolator(kalman_positions_2d, W_interp)
    #             W_fine_mesh = W_fun(full_positions_2d)
    #             W_fine_mesh = W_fine_mesh.reshape(size, ens_size, ens_size)
    #             ensemble[wind_size::] = x_bar[wind_size::, None] + np.einsum('ij, ijk->ik', ensemble[wind_size::], W_fine_mesh)
    #             ensemble[0:wind_size] += x_bar[0:wind_size, None]
    #         #     P_tilde[:, i] *= 1/num
    #         #     W_a[:, i] *= 1/np.sqrt(num)
    #         # P_tilde = P_tilde.dot(eig_vector.T)
    #         # W_a = W_a.dot(eig_vector.T)*(ens_size - 1)

    #         P_tilde = np.linalg.inv(
    #             (ens_size - 1)*np.eye(ens_size)/inflation +
    #             C.dot(local_ensemble))
    #         W_a = np.real(sp.linalg.sqrtm((ens_size - 1)*P_tilde))
    #         w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
    #         W_a += w_a_bar[:, None]
    #         ensemble = x_bar[:, None] + ensemble.dot(W_a)
    #         W_interp[kal_count] = np.ravel(W_a) ## separate w_bar??
    #         kal_count += 1
    #     W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
    #                                              W_interp)
    #     W_fine_mesh = W_fun(full_positions_2d)
    #     W_fine_mesh = W_fine_mesh.reshape(shape[0]*shape[1],
    #                                       ens_size, ens_size)
    #     ensemble = x_bar[:, None] + np.einsum(
    #         'ij, ijk->ik', ensemble, W_fine_mesh)

    domain_size = shape[0]*shape[1]
    ens_size = ensemble.shape[1]
    x_bar = ensemble.mean(axis=1)
    ensemble -= x_bar[:, np.newaxis] # Y_b is ensemble[wind_size::, :]
    ## localization with interpolation
    kal_count = 0
    W_interp = np.zeros([assimilation_positions.size, ens_size**2])
    for interp_position in assimilation_positions:
        local_positions = nearest_positions(interp_position, shape, domain_size)
        local_ensemble = ensemble[local_positions]
        local_x_bar = x_bar[local_positions]
        local_obs = observations[local_positions]
        C = (local_ensemble.T)*R_inverse
        P_a = np.linalg.inv((ens_size - 1)*np.eye(ens_size)/inflation +
                            C.dot(local_ensemble)) # withOUT noise inflation
        W_a = np.real(sp.linalg.sqrtm((ens_size - 1)*P_a))
        w_a_bar = P_a.dot(C.dot(local_obs - local_x_bar))
        W_a += w_a_bar[:, None]
        W_interp[kal_count] = np.ravel(W_a) ## separate w_bar??
        kal_count += 1

    W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d, W_interp)
    W_fine_mesh = W_fun(full_positions_2d)
    W_fine_mesh = W_fine_mesh.reshape(domain_size, ens_size, ens_size)
    ensemble = x_bar[:, None] + np.einsum('ij, ijk->ik', ensemble, W_fine_mesh)
    return ensemble


def calc_sensor_error(sensor_values, sensor_loc, H, q, time):
    """check back later
    """
    data = H.dot(q.ravel())[None, :]
    sat_values = pd.DataFrame(data=data,
                              index=[time],
                              columns=sensor_loc['id'])
    return sat_values - sensor_values


def ensemble_creator(sat_image, CI_sigma, wind_size, wind_sigma, ens_size):
    """check back later"""
    half_wind = int(round(wind_size/2))
    ens_wind = int(round(ens_size*half_wind))
    ensemble = np.random.normal(
        loc=0,
        scale=wind_sigma[0],
        size=ens_wind).reshape(half_wind, ens_size)
    ensemble = np.concatenate(
        [ensemble, np.random.normal(
            loc=0,
            scale=wind_sigma[1],
            size=ens_wind).reshape(half_wind, ens_size)], axis=0)
    ensemble = np.concatenate(
        [ensemble,
         np.repeat(sat_image.ravel()[:, None], ens_size, axis=1)], axis=0)
    return ensemble



def assimilation_position_generator(domain_shape, assimilation_grid_size):
    domain_size = domain_shape[0]*domain_shape[1]
    row_positions, col_positions = np.meshgrid(
        np.arange(0, domain_shape[0], assimilation_grid_size),
        np.arange(0, domain_shape[1], assimilation_grid_size))
    row_positions = np.ravel(row_positions)
    col_positions = np.ravel(col_positions)
    assimilation_positions = np.ravel_multi_index(
        (row_positions, col_positions), domain_shape)
    assimilation_positions.sort()
    assimilation_positions_2d = np.unravel_index(assimilation_positions,
                                                 domain_shape)
    assimilation_positions_2d = np.stack(assimilation_positions_2d, axis=1)
    full_positions_2d = np.unravel_index(np.arange(0, domain_size),
                                         domain_shape)
    full_positions_2d = np.stack(full_positions_2d, axis=1)
    return assimilation_positions, assimilation_positions_2d, full_positions_2d


def simulation(sat, wind, sensor_data, sensor_loc, start_time, end_time,
               dx, dy, C_max, assimilation_grid_size, localization_length,
               sat_sig, sensor_sig, ens_size, wind_sigma, wind_size):
    """Check back later."""
    ## NEED: Incorporate IO? Would need to reformulate so that P is smaller.
    time_range = (pd.date_range(start_time, end_time, freq='15 min')
                  .tz_localize('MST').astype(int))
    all_time = sat.time.values
    time_range = np.intersect1d(time_range, all_time)
    wind_smoothing_size = 30*10
    max_CI = 1.4
    sensor_error = pd.DataFrame(data=None, index=None, columns=sensor_loc.id)
    sensor_error_ens = sensor_error.copy()
    sat_loc = np.concatenate(
        (sat['lat'].values.ravel()[:, None],
         sat['long'].values.ravel()[:, None]), axis=1)
    domain_shape = sat['clear_sky_good'].isel(time=0).shape
    domain_size = domain_shape[0]*domain_shape[1]
    assimilation_positions, assimilation_positions_2d, full_positions_2d = (
        assimilation_position_generator(domain_shape, assimilation_grid_size))
    ## This is only for now. Eventually H will be a function of time and cloud height.
    H, delete = forward_obs_mat(sensor_loc[['lat', 'lon']].values, sat_loc)
    H = np.concatenate((np.zeros((H.shape[0], 2)), H), axis=1)

    ensemble = ensemble_creator(
        sat['clear_sky_good'].sel(time=time_range[0]).values,
        CI_sigma=None, wind_size=wind_size, wind_sigma=wind_sigma, ens_size=ens_size)
    for date_index in range(time_range.size - 1):
        start_time = time_range[date_index]
        end_time = time_range[date_index + 1]
        U = wind.sel(time=start_time, method='pad').U.values
        V = wind.sel(time=start_time, method='pad').V.values
        U = ndimage.filters.uniform_filter(U, size=wind_smoothing_size)
        V = ndimage.filters.uniform_filter(V, size=wind_smoothing_size)
        cx = abs(U).max()
        cy = abs(V).max()
        N = np.ceil((5*60)*(cx/dx+cy/dy)/C_max)
        dt = (5*60)/N
        T = (end_time - start_time)*10**(-9)
        T_steps = int(T/dt)
        q = sat['clear_sky_good'].sel(time=start_time).values
        this_time = pd.Timestamp(
            start_time).tz_localize('UTC').tz_convert('MST')

        ## Maybe this isn't working? Maybe not though.
        if date_index != 0:
            print('Starting Full image')
            ensemble[wind_size::] = assimilate(
                ensemble[wind_size::],
                sat['clear_sky_good'].sel(
                    time=time_range[date_index]).values.ravel(),
                None, 1/sat_sig**2, 1, shape=domain_shape,
                localization_length=localization_length,
                assimilation_positions=assimilation_positions,
                assimilation_positions_2d=assimilation_positions_2d,
                full_positions_2d=full_positions_2d)

            this_index = 2
            plt.figure()
            plt.pcolormesh(
                ensemble[wind_size::, this_index].reshape(domain_shape),
                cmap='Blues_r')
            plt.show()
            print(ensemble[wind_size::, this_index].mean())
        #     return None

        ## only calculate error for forecasts
        # sensor_error = sensor_error.append(
        #     calc_sensor_error(sensor_data.ix[this_time],
        #                       sensor_loc, H, q, this_time))
        # plt.figure()
        # im = plt.pcolormesh(sat.long, sat.lat, q, cmap='Blues', vmin=0, vmax=1)
        # plt.colorbar(im)
        # plt.title(this_time)
        # plt.scatter(sensor_loc.lon, sensor_loc.lat, c='r')
        # plt.show()

        for t in range(T_steps):
            q = time_deriv_3(q, dt, U, dx, V, dy)
            q = q.clip(max=max_CI, min=0)
            for ens_index in range(ens_size):
                ensemble[wind_size::, ens_index] = time_deriv_3(
                    ensemble[wind_size::, ens_index].reshape(domain_shape), dt,
                    U + ensemble[0, ens_index], dx,
                    V + ensemble[1, ens_index], dy).reshape(domain_size)
            nearest_up = 5*np.round((t + 1)*dt/(60*5))
            test = (abs(nearest_up - (t + 1)*dt/60) <
                    abs(nearest_up - (t + 2)*dt/60))
            test = test and (abs(nearest_up-(t+1)*dt/60) <
                             abs(nearest_up-(t)*dt/60))
            test = test and (nearest_up != 0)
            if test:
                this_time = pd.Timestamp(
                    start_time + (t+1)*dt*10**9).tz_localize('UTC'
                    ).tz_convert('MST')
                ensemble = assimilate(ensemble, sensor_data.ix[this_time],
                                      H, 1/sensor_sig**2, 1)
                ensemble_mean = ensemble[wind_size::].mean(axis=1).reshape(
                    domain_shape)
                sensor_error = sensor_error.append(
                    calc_sensor_error(sensor_data.ix[this_time],
                                      sensor_loc, H[:, wind_size::],
                                      q, this_time))
                sensor_error_ens = sensor_error_ens.append(
                    calc_sensor_error(
                        sensor_data.ix[this_time], sensor_loc,
                        H[:, wind_size::], ensemble_mean, this_time))

                print(sensor_error.ix[this_time].abs().mean())
                print(sensor_error_ens.ix[this_time].abs().mean())

                plt.figure()
                im = plt.pcolormesh(sat.long, sat.lat, q,
                                    cmap='Blues', vmin=0, vmax=1)
                plt.colorbar(im)
                plt.title('Just advect: ' + str(this_time))
                plt.scatter(sensor_loc.lon, sensor_loc.lat, c='r')
                plt.show()

                plt.figure()
                im = plt.pcolormesh(sat.long, sat.lat,
                                    ensemble_mean,
                                    cmap='Blues', vmin=0, vmax=1)
                plt.colorbar(im)
                plt.title('Ensemble: ' + str(this_time))
                plt.scatter(sensor_loc.lon, sensor_loc.lat, c='r')
                plt.show()
