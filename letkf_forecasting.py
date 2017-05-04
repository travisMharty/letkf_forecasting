import numpy as np
import pandas as pd
import scipy as sp


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

    return sensor_loc, H


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
               full_positions=False):
    """Assimilates observations into ensemble using the LETKF.

    Parameters
    ----------
    ensemble : array
         The ensemble of size kxn where k is the number of ensemble members
         and n is the state vector size.
    observations : array
         An observation vector of length m.
    H : array
         Forward observation matrix of size mxn.
    R_inverse : array
         Inverse of observation error matrix.
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
    x_bar = ensemble.mean(axis=1)
    ensemble -= x_bar
    ens_size = ensemble.shape[1]

    if localization_length is False:
        # LETKF without localization
        Y_b = np.einsum('ij,jk...->ik...', H, ensemble)
        y_b_bar = Y_b.mean(axis=1)
        Y_b -= y_b_bar[:, np.newaxis]
        C = (Y_b.T).dot(R_inverse)
        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/inflation + C.dot(Y_b))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)
        w_a_bar = P_tilde.dot(C.dot(observations - y_b_bar))
        W_a += w_a_bar[:, None]
        ensemble = x_bar[:, None] + ensemble.dot(W_a)
        return ensemble

    else:
        # LETKF with localization assumes H is I
        ## Change to include wind in ensemble will require reworking due to
        ## new H and different localization.
        kal_count = 0
        W_interp = np.zeros([assimilation_positions.size, ens_size**2])
        for interp_position in assimilation_positions:
            local_positions = nearest_positions(interp_position, shape,
                                                localization_length)
            local_j, local_i = np.meshgrid(local_positions, local_positions)
            local_R_inverse = R_inverse[local_i, local_j]
            local_ensemble = ensemble[local_positions]
            local_x_bar = x_bar[local_positions]
            local_obs = observations[local_positions]
            C = (local_ensemble.T).dot(local_R_inverse)
            eig_value, eig_vector = np.linalg.eigh(
                (ens_size-1)*np.eye(ens_size)/inflation + C.dot(local_ensemble))
            P_tilde = eig_vector.copy()
            W_a = eig_vector.copy()
            for i, num in enumerate(eig_value):
                P_tilde[:, i] *= 1/num
                W_a[:, i] *= 1/np.sqrt(num)
            P_tilde = P_tilde.dot(eig_vector.T)
            W_a = W_a.dot(eig_vector.T)
            w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
            W_a += w_a_bar[:, None]
            ensemble = x_bar[:, None] + ensemble.dot(W_a)
            W_interp[kal_count] = np.ravel(W_a) ## separate w_bar??
            kal_count += 1
        assimilation_positions_2d = np.unravel_index(assimilation_positions,
                                                     shape)
        W_fun = sp.interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                    W_interp)
        W_fine_mesh = W_fun(full_positions)
        W_fine_mesh = W_fine_mesh.reshape(shape[0]*shape[1],
                                          ens_size, ens_size)
        ensemble = x_bar + np.einsum('ij, ijk->ik', ensemble, W_fine_mesh)
        return ensemble


def simulation(sat, wind, sensor_data, sensor_loc, start_time, end_time,
               dx, dy, C_max, assimilation_grid_size, location):
    """"""
