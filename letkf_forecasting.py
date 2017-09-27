import sys
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pvlib as pv
import numexpr as ne
from distributed import LocalCluster, Client
from skimage import filters as ski_filters
import fenics as fe
import cv2

sys.path.append('/home/travis/python_code/letkf_forecasting/')
import prepare_sat_data as prep

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
    # F_x[:, 2:-2] = u[:, 2:-2]/12*(
    #     7*(q[:, 2:-1] + q[:, 1:-2]) - (q[:, 3:] + q[:, :-3]))
    # F_y[2:-2, :] = v[2:-2, :]/12*(
    #     7*(q[2:-1, :] + q[1:-2, :]) - (q[3:, :] + q[:-3, :]))
    # qout[:, 2:-2] = qout[:, 2:-2] - (F_x[:, 3:-2] - F_x[:, 2:-3])/dx
    # qout[2:-2, :] = qout[2:-2, :] - (F_y[3:-2, :] - F_y[2:-3, :])/dy

    # with numexpr
    u22 = u[:, 2:-2]
    q21 = q[:, 2:-1]
    q12 = q[:, 1:-2]
    q3 = q[:, 3:]
    qn3 = q[:, :-3]
    F_x[:, 2:-2] = ne.evaluate('u22 / 12 * (7 * (q21 + q12) - (q3 + qn3))')

    v22 = v[2:-2, :]
    q21 = q[2:-1, :]
    q12 = q[1:-2, :]
    q3 = q[3:, :]
    qn3 = q[:-3, :]
    F_y[2:-2, :] = ne.evaluate('v22 / 12 * (7 * (q21 + q12) - (q3 + qn3))')

    qo22 = qout[:, 2:-2]
    fx32 = F_x[:, 3:-2]
    fx23 = F_x[:, 2:-3]
    qout[:, 2:-2] = ne.evaluate('qo22 - (fx32 - fx23) / dx')

    qo22 = qout[2:-2, :]
    fy32 = F_y[3:-2, :]
    fy23 = F_y[2:-3, :]
    qout[2:-2, :] = ne.evaluate('qo22 - (fy32 - fy23) / dy')


    # boundary calculation
    u_w = u[:, 0:2].clip(max=0)
    u_e = u[:, -2:].clip(min=0)
    # qout[:, 0:2] = qout[:, 0:2] - ((u_w/dx)*(
    #     q[:, 1:3] - q[:, 0:2]) + (q[:, 0:2]/dx)*(u[:, 1:3] - u[:, 0:2]))
    # qout[:, -2:] = qout[:, -2:] - ((u_e/dx)*(
    #    q[:, -2:] - q[:, -3:-1]) + (q[:, -2:]/dx)*(u[:, -2:] - u[:, -3:-1]))

    qo02 = qout[:, 0:2]
    q13 = q[:, 1:3]
    q02 = q[:, 0:2]
    u13 = u[:, 1:3]
    u02 = u[:, 0:2]
    qout[:, 0:2] = ne.evaluate('qo02 - ((u_w/dx)*(q13 - q02) + (q02/dx)*(u13 - u02))')

    qo2 = qout[:, -2:]
    q2 = q[:, -2:]
    q31 = q[:, -3:-1]
    u2 = u[:, -2:]
    u31 = u[:, -3:-1]
    qout[:, -2:] = ne.evaluate('qo2 - ((u_e/dx)*(q2 - q31) + (q2/dx)*(u2 - u31))')

    v_n = v[-2:, :].clip(min=0)
    v_s = v[0:2, :].clip(max=0)
    # qout[0:2, :] = qout[0:2, :] - ((v_s/dx)*(
    #     q[1:3, :] - q[0:2, :]) + (q[0:2, :]/dx)*(v[1:3, :] - v[0:2, :]))
    # qout[-2:, :] = qout[-2:, :] - ((v_n/dx)*(
    #     q[-2:, :] - q[-3:-1, :]) + (q[-2:, :]/dx)*(v[-2:, :] - v[-3:-1, :]))

    qo02 = qout[0:2, :]
    q13 = q[1:3, :]
    q02 = q[0:2, :]
    v13 = v[1:3, :]
    v02 = v[0:2, :]
    qout[0:2, :] = ne.evaluate('qo02 - ((v_s/dx)*(q13 - q02) + (q02/dx)*(v13 - v02))')

    qo2 = qout[-2:, :]
    q2 = q[-2:, :]
    q31 = q[-3:-1, :]
    v2 = v[-2:, :]
    v31 = v[-3:-1, :]
    qout[-2:, :] = ne.evaluate('qo2 - ((v_n/dx)*(q2 - q31) + (q2/dx)*(v2 - v31))')
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
         Azimuth angle of satellite in degrees.

    satellite_altitude : float
         Altitude angle of satellite in degrees.

    solar_azimuth : float
         Azimuth angle of the sun in degrees.

    solar_altitude : float
         Altitude angle of the sun in degrees.

    Returns
    -------
    x_correction, y_correction : float
         x_correction and y_correction are the values which must be added to
         the satellite position to find actual position of cloud shadow.
    """
    satellite_displacement = cloud_height*cot(satellite_elevation*2*np.pi/360)
    solar_displacement = cloud_height*cot(solar_elevation*2*np.pi/360)
    x_correction = (
        solar_displacement*np.cos(-np.pi/2 - solar_azimuth*2*np.pi/360) -
        satellite_displacement*np.cos(-np.pi/2 - satellite_azimuth*2*np.pi/360))
    y_correction = (
        solar_displacement*np.sin(-np.pi/2 - solar_azimuth*2*np.pi/360) -
        satellite_displacement*np.sin(-np.pi/2 - satellite_azimuth*2*np.pi/360))

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




def to_nearest_indices(array, values):
    """Modified from
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = np.searchsorted(array, values, side="left")
    for i in range(idx.size):
        if idx[i] > 0 and (idx[i] == len(array) or
                           np.abs(values[i] - array[idx[i] - 1]) <
                           np.abs(values[i] - array[idx[i]])):
            idx[i] = idx[i] - 1
    return idx

def to_lat_lon(x, y, loc_lat):
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
    lon = x*360/(2*np.pi*a*np.cos(loc_lat*2*np.pi/360))
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


def optimal_interpolation(background, b_sig,
                          observations, o_sig,
                          P_field,
                          flat_locations, localization, sat_inflation,
                          spacial_localization=False, x=None, y=None):

    if spacial_localization:
        PHT = np.sqrt((x[:, None] - x[None, flat_locations])**2 +
                      (y[:, None] - y[None, flat_locations])**2)
        PHT = np.exp(-PHT/(localization))
        PHT = sat_inflation*PHT*b_sig**2

    if P_field is not None:
        PHT = np.abs(P_field[:, None] - P_field[None, flat_locations])
        PHT = np.exp(-PHT/(localization))
        PHT = sat_inflation*PHT*b_sig**2
    # K = sp.linalg.inv(
    #     PHT[flat_locations, :] + o_sig**2*np.eye(flat_locations.size))
    # K = PHT.dot(K)
    K = sp.linalg.solve(
        (PHT[flat_locations, :] + o_sig**2*np.eye(flat_locations.size)),
        PHT.T).T

    # ###
    # plt.figure()
    # plt.plot(observations)
    # plt.plot(background[flat_locations])
    # plt.plot(observations - background[flat_locations])
    # ###


    return background + K.dot(observations - background[flat_locations])

def assimilate(ensemble, observations, flat_sensor_indices, R_inverse,
                        inflation, domain_shape=False,
                        localization_length=False, assimilation_positions=False,
                        assimilation_positions_2d=False,
                        full_positions_2d=False):
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
    if localization_length is False:

        # LETKF without localization
        Y_b = ensemble[flat_sensor_indices, :]
        y_b_bar = Y_b.mean(axis=1)
        Y_b -= y_b_bar[:, None]
        x_bar = ensemble.mean(axis=1) ## Need to bring this back
        ensemble -= x_bar[:, None]
        ens_size = ensemble.shape[1]
        # C = (Y_b.T).dot(R_inverse)
        C = Y_b.T*R_inverse
        ## Not working??
        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/inflation + C.dot(Y_b))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*(np.sqrt(ens_size - 1))
        # P_tilde = np.linalg.inv(
        #     (ens_size - 1)*np.eye(ens_size)/inflation +
        #     C.dot(Y_b))
        # W_a = np.real(sp.linalg.sqrtm((ens_size - 1)*P_tilde))
        w_a_bar = P_tilde.dot(C.dot(observations - y_b_bar))
        W_a += w_a_bar[:, None]
        ensemble = x_bar[:, None] + ensemble.dot(W_a)
        return ensemble

    else:
        # LETKF with localization assumes H is I
        ## NEED: to include wind in ensemble will require reworking due to
        ## new H and different localization.
        ## NEED: Change to include some form of H for paralax correction??
        ## Maybe: ^ not if paralax is only corrected when moving to ground sensors.
        ## SHOULD: Will currently write as though R_inverse is a scalar.
        ## May need to change at some point but will likely need to do
        ## something clever since R_inverse.size is 400 billion
        ## best option: form R_inverse inside of localization routine
        ## good option: assimilate sat images at low resolution (probabily should do this either way)
        x_bar = ensemble.mean(axis=1) ## Need to bring this back
        ensemble -= x_bar[:, None]
        ens_size = ensemble.shape[1]
        kal_count = 0
        W_interp = np.zeros([assimilation_positions.size, ens_size**2])
        for interp_position in assimilation_positions:
            local_positions = nearest_positions(interp_position, domain_shape,
                                                localization_length)
            local_ensemble = ensemble[local_positions]
            local_x_bar = x_bar[local_positions]
            local_obs = observations[local_positions] # assume H is I
            C = (local_ensemble.T)*R_inverse  # assume R_inverse is diag+const

            # This should be better, but I can't get it to work
            eig_value, eig_vector = np.linalg.eigh(
                (ens_size-1)*np.eye(ens_size)/inflation + C.dot(local_ensemble))
            P_tilde = eig_vector.copy()
            W_a = eig_vector.copy()
            for i, num in enumerate(eig_value):
                P_tilde[:, i] *= 1/num
                W_a[:, i] *= 1/np.sqrt(num)
            P_tilde = P_tilde.dot(eig_vector.T)
            W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)

            # P_tilde = np.linalg.inv(
            #     (ens_size - 1)*np.eye(ens_size)/inflation +
            #     C.dot(local_ensemble))
            # W_a = np.real(sp.linalg.sqrtm((ens_size - 1)*P_tilde))
            w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
            W_a += w_a_bar[:, None]
            W_interp[kal_count] = np.ravel(W_a) ## separate w_bar??
            kal_count += 1
        if assimilation_positions_2d.size != full_positions_2d.size:
            W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                     W_interp)
            W_fine_mesh = W_fun(full_positions_2d)
            W_fine_mesh = W_fine_mesh.reshape(domain_shape[0]*domain_shape[1],
                                              ens_size, ens_size)
        else:
            W_fine_mesh = W_interp.reshape(domain_shape[0]*domain_shape[1],
                                              ens_size, ens_size)
        ensemble = x_bar[:, None] + np.einsum(
            'ij, ijk->ik', ensemble, W_fine_mesh)

        return ensemble


def assimilate_wind(ensemble, observations, flat_sensor_indices, R_inverse,
                    inflation, wind_size):
    """
    *** NEED TO REWRITE
    Assimilates observations into ensemble using the LETKF.

    Parameters
    ----------
    ensemble : array
         The ensemble of size kxn where k is the number of ensemble members
         and n is the state vector size.
    observations : array
         An observation vector of length m
    R_inverse : array
         Inverse of observation error matrix. **will need changing**
    inflation : float
         Inflation parameter

    Return
    ------
    winds
    """

    # LETKF without localization
    Y_b = ensemble[wind_size:].copy()
    y_b_bar = Y_b.mean(axis=1)
    Y_b -= y_b_bar[:, None]
    x_bar = ensemble.mean(axis=1) ## Need to bring this back
    ensemble -= x_bar[:, None]
    ens_size = ensemble.shape[1]
    C = Y_b.T*R_inverse
    eig_value, eig_vector = np.linalg.eigh(
        (ens_size-1)*np.eye(ens_size)/inflation + C.dot(Y_b))
    P_tilde = eig_vector.copy()
    W_a = eig_vector.copy()
    for i, num in enumerate(eig_value):
        P_tilde[:, i] *= 1/num
        W_a[:, i] *= 1/np.sqrt(num)
    P_tilde = P_tilde.dot(eig_vector.T)
    W_a = W_a.dot(eig_vector.T)*(np.sqrt(ens_size - 1))
    # P_tilde = np.linalg.inv(
    #     (ens_size - 1)*np.eye(ens_size)/inflation +
    #     C.dot(Y_b))
    # W_a = np.real(sp.linalg.sqrtm((ens_size - 1)*P_tilde))
    w_a_bar = P_tilde.dot(C.dot(observations - y_b_bar))
    W_a += w_a_bar[:, None]
    ensemble = x_bar[:, None] + ensemble.dot(W_a)
    return ensemble[:wind_size]


def assimilate_enkf(ensemble, observations, flat_sensor_indices, R_inverse,
                        inflation, domain_shape=False,
                        localization_length=False, assimilation_positions=False,
                        assimilation_positions_2d=False,
                        full_positions_2d=False):
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
    if localization_length is False:

        ## this is not done
        Y_b = ensemble[flat_sensor_indices, :]
        y_b_bar = Y_b.mean(axis=1)
        Y_b -= y_b_bar[:, None]
        x_bar = ensemble.mean(axis=1) ## Need to bring this back
        ensemble -= x_bar[:, None]
        PHT = ensemble.dot(Y_b.T)
        loc_mat = np.abs(x_bar[:, None] - x_bar[None, flat_locations])
        loc_mat = np.exp(-loc_mat/(enkf_localization))
        PHT = PHT*loc_mat
        PHT = inflation*PHT
        # K = sp.linalg.inv(
        #     PHT[flat_locations, :] + o_sig**2*np.eye(flat_locations.size))
        # K = PHT.dot(K)
        K = sp.linalg.solve(
            (PHT[flat_locations, :] + (1/R_inverse)*np.eye(flat_locations.size)),
            PHT.T).T
        background + K.dot(observations - background[flat_locations])

        # # LETKF without localization
        # Y_b = ensemble[flat_sensor_indices, :]
        # y_b_bar = Y_b.mean(axis=1)
        # Y_b -= y_b_bar[:, None]
        # x_bar = ensemble.mean(axis=1) ## Need to bring this back
        # ensemble -= x_bar[:, None]
        # ens_size = ensemble.shape[1]
        # # C = (Y_b.T).dot(R_inverse)
        # C = Y_b.T*R_inverse
        # ## Not working??
        # eig_value, eig_vector = np.linalg.eigh(
        #     (ens_size-1)*np.eye(ens_size)/inflation + C.dot(Y_b))
        # P_tilde = eig_vector.copy()
        # W_a = eig_vector.copy()
        # for i, num in enumerate(eig_value):
        #     P_tilde[:, i] *= 1/num
        #     W_a[:, i] *= 1/np.sqrt(num)
        # P_tilde = P_tilde.dot(eig_vector.T)
        # W_a = W_a.dot(eig_vector.T)*(np.sqrt(ens_size - 1))
        # # P_tilde = np.linalg.inv(
        # #     (ens_size - 1)*np.eye(ens_size)/inflation +
        # #     C.dot(Y_b))
        # # W_a = np.real(sp.linalg.sqrtm((ens_size - 1)*P_tilde))
        # w_a_bar = P_tilde.dot(C.dot(observations - y_b_bar))
        # W_a += w_a_bar[:, None]
        # ensemble = x_bar[:, None] + ensemble.dot(W_a)
        # return ensemble

    else:
        # LETKF with localization assumes H is I
        ## NEED: to include wind in ensemble will require reworking due to
        ## new H and different localization.
        ## NEED: Change to include some form of H for paralax correction??
        ## Maybe: ^ not if paralax is only corrected when moving to ground sensors.
        ## SHOULD: Will currently write as though R_inverse is a scalar.
        ## May need to change at some point but will likely need to do
        ## something clever since R_inverse.size is 400 billion
        ## best option: form R_inverse inside of localization routine
        ## good option: assimilate sat images at low resolution (probabily should do this either way)
        x_bar = ensemble.mean(axis=1) ## Need to bring this back
        ensemble -= x_bar[:, None]
        ens_size = ensemble.shape[1]
        kal_count = 0
        W_interp = np.zeros([assimilation_positions.size, ens_size**2])
        for interp_position in assimilation_positions:
            local_positions = nearest_positions(interp_position, domain_shape,
                                                localization_length)
            local_ensemble = ensemble[local_positions]
            local_x_bar = x_bar[local_positions]
            local_obs = observations[local_positions] # assume H is I
            C = (local_ensemble.T)*R_inverse  # assume R_inverse is diag+const

            # This should be better, but I can't get it to work
            eig_value, eig_vector = np.linalg.eigh(
                (ens_size-1)*np.eye(ens_size)/inflation + C.dot(local_ensemble))
            P_tilde = eig_vector.copy()
            W_a = eig_vector.copy()
            for i, num in enumerate(eig_value):
                P_tilde[:, i] *= 1/num
                W_a[:, i] *= 1/np.sqrt(num)
            P_tilde = P_tilde.dot(eig_vector.T)
            W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)

            # P_tilde = np.linalg.inv(
            #     (ens_size - 1)*np.eye(ens_size)/inflation +
            #     C.dot(local_ensemble))
            # W_a = np.real(sp.linalg.sqrtm((ens_size - 1)*P_tilde))
            w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
            W_a += w_a_bar[:, None]
            W_interp[kal_count] = np.ravel(W_a) ## separate w_bar??
            kal_count += 1
        if assimilation_positions_2d.size != full_positions_2d.size:
            W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                     W_interp)
            W_fine_mesh = W_fun(full_positions_2d)
            W_fine_mesh = W_fine_mesh.reshape(domain_shape[0]*domain_shape[1],
                                              ens_size, ens_size)
        else:
            W_fine_mesh = W_interp.reshape(domain_shape[0]*domain_shape[1],
                                              ens_size, ens_size)
        ensemble = x_bar[:, None] + np.einsum(
            'ij, ijk->ik', ensemble, W_fine_mesh)

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
    CI_pert = np.random.normal(loc=0, scale=CI_sigma, size=ens_size)
    csi_min_pert = np.random.normal(loc=0, scale=CI_sigma, size=ens_size)
    csi_max_pert = np.random.normal(loc=1, scale=CI_sigma*.2, size=ens_size)
    ensemble[wind_size:] = ((csi_max_pert[None, :] - csi_min_pert[None, :])*
                            ensemble[wind_size:] + csi_min_pert[None, :])
    # ensemble[wind_size:] = ((1 - CI_pert[None, :])*ensemble[wind_size:] +
    #                         CI_pert[None, :])
    # ensemble[wind_size:] = ensemble[wind_size:] + CI_pert[None, :]
    return ensemble


def ensemble_creator_wind(sat_image, u, v, CI_sigma, wind_sigma, ens_size):
    """need to change later"""
    random_nums = np.random.normal(
        loc=0,
        scale=wind_sigma,
        size=ens_size)
    half_wind = int(round(wind_size/2))
    ens_wind = int(round(ens_size*half_wind))
    ensemble = u.ravel()[:, None] + random_nums[None, :]

    random_nums = np.random.normal(
        loc=0,
        scale=wind_sigma,
        size=ens_size)
    ensemble = np.concatenate(
        ensemble,
        v.ravel()[:, None] + random_nums[None, :],
        axis=0)
    ensemble = np.concatenate(
        [ensemble,
         np.repeat(sat_image.ravel()[:, None], ens_size, axis=1)],
        axis=0)
    wind_size = u.size + v.size
    CI_pert = np.random.normal(loc=0, scale=CI_sigma, size=ens_size)
    csi_min_pert = np.random.normal(loc=0, scale=CI_sigma, size=ens_size)
    csi_max_pert = np.random.normal(loc=1, scale=CI_sigma*.2, size=ens_size)
    ensemble[wind_size:] = ((csi_max_pert[None, :] - csi_min_pert[None, :])*
                            ensemble[wind_size:] + csi_min_pert[None, :])
    # ensemble[wind_size:] = ((1 - CI_pert[None, :])*ensemble[wind_size:] +
    #                         CI_pert[None, :])
    # ensemble[wind_size:] = ensemble[wind_size:] + CI_pert[None, :]
    return ensemble


def ensemble_creator_old(sat_image, CI_sigma, wind_size, wind_sigma, ens_size):
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
    CI_pert = np.random.normal(loc=0, scale=CI_sigma, size=ens_size)
    ensemble[wind_size:] = ((1 - CI_pert[None, :])*ensemble[wind_size:] +
                            CI_pert[None, :])
    return ensemble


def assimilation_position_generator(domain_shape, assimilation_grid_size):
    domain_size = domain_shape[0]*domain_shape[1]
    row_positions = np.arange(0, domain_shape[0], assimilation_grid_size)
    col_positions = np.arange(0, domain_shape[1], assimilation_grid_size)
    if row_positions[-1] != domain_shape[0] - 1:
        row_positions = np.concatenate((row_positions,
                                        np.array(domain_shape[0] - 1)[None]))
    if col_positions[-1] != domain_shape[1] - 1:
        col_positions = np.concatenate((col_positions,
                                        np.array(domain_shape[1] - 1)[None]))
    row_positions, col_positions = np.meshgrid(row_positions, col_positions)
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


def noise_fun(domain_shape):
    noise_init = np.zeros(domain_shape)
    noise_init[0:25, :] = 1
    noise_init[-25:, :] = 1
    noise_init[:, 0:25] = 1
    noise_init[:, -25:] = 1
    noise_init = sp.ndimage.gaussian_filter(noise_init, 12)
    return noise_init


def advect_5min(q, dt, U, dx, V, dy, T_steps):
    """Check back later"""
    for t in range(T_steps):
        q = time_deriv_3(q, dt, U, dx, V, dy)
    return q


def advect_5min_old(q, noise, ensemble, dt, U, dx, V, dy, T_steps, wind_size):
    """This may be used in old code, should use distributed"""
    domain_shape = q.shape
    domain_size = domain_shape[0]*domain_shape[1]
    ens_size = ensemble.shape[1]
    for t in range(T_steps):
        q = time_deriv_3(q, dt, U, dx, V, dy)
        noise = time_deriv_3(noise, dt, U, dx, V, dy)
        for ens_index in range(ens_size):
            ensemble[wind_size:, ens_index] = time_deriv_3(
                ensemble[wind_size:, ens_index].reshape(domain_shape), dt,
                U + ensemble[0, ens_index], dx,
                V + ensemble[1, ens_index], dy).reshape(domain_size)
    return q, noise, ensemble


def advect_5min_distributed(
        q, noise, ensemble, dt, U, dx, V, dy, T_steps, wind_size, client):

        """Check back later"""
        domain_shape = q.shape
        domain_size = domain_shape[0]*domain_shape[1]
        ens_size = ensemble.shape[1]

        def time_deriv_3_loop(CI_field, u_pert, v_pert):
            CI_field = CI_field.reshape(domain_shape)
            for t in range(T_steps):
                CI_field = time_deriv_3(CI_field, dt,
                                        U + u_pert, dx,
                                        V + v_pert, dy)
            return CI_field.ravel()
        CI_fields = ensemble[wind_size:].copy()
        CI_fields = np.concatenate(
            [q.ravel()[:, None], noise.ravel()[:, None], CI_fields], axis=1)
        CI_fields = CI_fields.T
        ## only works for constant wind perturbations
        u_perts = np.concatenate([[0], [0], ensemble[0]], axis=0)
        v_perts = np.concatenate([[0], [0], ensemble[1]], axis=0)
        futures = client.map(time_deriv_3_loop,
                             CI_fields, u_perts, v_perts)
        q = futures[0].result()
        q = q.reshape(domain_shape)
        noise = futures[1].result()
        noise = noise.reshape(domain_shape)
        temp = client.gather(futures[2:])
        temp = np.stack(temp, axis=1)
        ensemble[wind_size:] = temp
        return q, noise, ensemble


def advect_5min_distributed_coeff(
        q, noise, ensemble, dt, U, dx, V, dy, T_steps, coeff_size, wind_size, client):

        """Check back later"""
        domain_shape = q.shape
        domain_size = domain_shape[0]*domain_shape[1]
        ens_size = ensemble.shape[1]

        def time_deriv_3_loop(CI_field, u_pert, v_pert):
            CI_field = CI_field.reshape(domain_shape)
            for t in range(T_steps):
                CI_field = time_deriv_3(CI_field, dt,
                                        U + u_pert, dx,
                                        V + v_pert, dy)
            return CI_field.ravel()
        CI_fields = ensemble[(coeff_size + wind_size):].copy()
        CI_fields = np.concatenate(
            [q.ravel()[:, None], noise.ravel()[:, None], CI_fields], axis=1)
        CI_fields = CI_fields.T
        ## only works for constant wind perturbations
        u_perts = np.concatenate([[0], [0], ensemble[coeff_size + 0]], axis=0)
        v_perts = np.concatenate([[0], [0], ensemble[coeff_size + 1]], axis=0)
        futures = client.map(time_deriv_3_loop,
                             CI_fields, u_perts, v_perts)
        q = futures[0].result()
        q = q.reshape(domain_shape)
        noise = futures[1].result()
        noise = noise.reshape(domain_shape)
        temp = client.gather(futures[2:])
        temp = np.stack(temp, axis=1)
        ensemble[coeff_size + wind_size:] = temp
        return q, noise, ensemble



def find_flat_loc(sat_lat, sat_lon, sensor_loc):
    sat_lat = sat_lat[:, None]
    sat_lon = sat_lon[:, None]
    sensor_lat = sensor_loc['lat'].values[None, :]
    sensor_lon = sensor_loc['lon'].values[None, :]
    distance = np.sqrt((sat_lat - sensor_lat)**2 + (sat_lon - sensor_lon)**2)
    return np.argmin(distance, axis=0)


# # OLD VERSION
# def find_flat_loc(sat, sensor_loc):
#     sat_lat = sat.lat.values[:, 0]
#     sat_lon = sat.long.values[0, :]
#     shape = sat.lat.shape
#     sensor_lat = sensor_loc['lat'].values
#     sensor_lon = sensor_loc['lon'].values
#     sensor_lat_indices = to_nearest_indices(sat_lat, sensor_lat)
#     sensor_lon_indices = to_nearest_indices(sat_lon, sensor_lon)
#     lat_step = sat_lat[1] - sat_lat[0]
#     lon_step = sat_lon[1] - sat_lon[0]
#     sensor_indices_2d = np.stack([sensor_lat_indices, sensor_lon_indices])
#     sensor_indices_flat = np.ravel_multi_index(
#         multi_index=(sensor_lat_indices, sensor_lon_indices),
#         dims=shape)
#     return sensor_indices_flat, lat_step, lon_step


def get_flat_correct(
        cloud_height, dx, dy, domain_shape, sat_azimuth,
        sat_elevation, location, sensor_time):

    solar_position = location.get_solarposition(sensor_time)
    x_correct, y_correct = parallax_shift(
        cloud_height, sat_azimuth, sat_elevation,
        solar_position['azimuth'].values,
        solar_position['elevation'].values)
    x_correct = int(np.round(x_correct/dx))
    y_correct = int(np.round(y_correct/dy))
    flat_correct = x_correct + y_correct*domain_shape[1]
    return flat_correct


# OLD version
# def get_flat_correct(
#         cloud_height, lat_step, lon_step, domain_shape, sat_azimuth,
#         sat_elevation, location, sensor_time):


#     solar_position = location.get_solarposition(sensor_time)
#     x_correct, y_correct = parallax_shift(
#         cloud_height, sat_azimuth, sat_elevation,
#         solar_position['azimuth'].values,
#         solar_position['elevation'].values)
#     lat_correct, lon_correct = to_lat_lon(x_correct, y_correct,
#                                           location.latitude)
#     west_east_correct = int(np.round(lon_correct/lon_step))
#     south_north_correct = int(np.round(lat_correct/lat_step))
#     flat_correct = west_east_correct + south_north_correct*domain_shape[1] # should be 1??
#     return flat_correct


def preturb_irradiance(ensemble, domain_shape, edge_weight, pert_mean,
                       pert_sigma, rf_approx_var, rf_eig, rf_vectors):
    ens_size = ensemble.shape[1]
    average = ensemble.mean(axis=1)
    average = average.reshape(domain_shape)
    target = ski_filters.sobel(average)
    target = target/target.max()
    target[target < 0.1] = 0
    target = sp.ndimage.gaussian_filter(target, sigma=4)
    target = target/target.max()
    cloud_target = 1 - average
    cloud_target = (cloud_target/cloud_target.max()).clip(min=0,
                                                          max=1)
    target = np.maximum(cloud_target, target*edge_weight)
    target = target/target.max()
    target = sp.ndimage.gaussian_filter(target, sigma=5)
    target = target.ravel()
    sample = np.random.randn(rf_eig.size, ens_size)
    sample = rf_vectors.dot(np.sqrt(rf_eig[:, None])*sample)
    target_mean = target.mean()
    target_var = (target**2).mean()
    cor_mean = pert_mean/target_mean
    cor_sd = pert_sigma/np.sqrt(rf_approx_var*target_var)
    ensemble = (
        ensemble +
        (cor_sd*sample + cor_mean)*target[:, None])
    return ensemble


def coarsen(array, coarseness):
    old_shape = np.array(array.shape, dtype=float)
    new_shape = coarseness * np.ceil(old_shape / coarseness).astype(int)

    row_add = int(new_shape[0] - old_shape[0])
    col_add = int(new_shape[1] - old_shape[1])
    padded = np.pad(array, ((0, row_add),(0, col_add)), mode='edge')
    temp = padded.reshape(new_shape[0] // coarseness, coarseness,
                          new_shape[1] // coarseness, coarseness)
    array = np.sum(temp, axis=(1, 3))/coarseness**2
    return array


def divergence(u, v, dx, dy):
    dudy, dudx = np.gradient(u, dy, dx)
    dvdy, dvdx = np.gradient(v, dy, dx)
    return dudx + dvdy


def optical_flow(image0, image1, time0, time1, u, v):
    var_size = 7
    # win_var_size = 11
    var_sig = 2
    var_thresh = 300
    sd_num = 2                  # for removing u_of & v_of
    coarseness = 4
    feature_params = dict( maxCorners=5000,
                           qualityLevel=0.0001,
                           minDistance=10,
                           blockSize=4)
    winSize = (50, 50)
    # windSize = (int(round(80/coarseness)), int(round(80/coarseness)))
    maxLevel = 5
    lk_params = dict(winSize=winSize,
                     maxLevel=maxLevel,
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    image0 = coarsen(array=image0, coarseness=coarseness)
    image1 = coarsen(array=image1, coarseness=coarseness)

    c_min = min(image0.min(), image1.min())
    c_max = max(image0.max(), image1.max())
    image0 = (image0 - c_min)/(c_max - c_min)*255
    image1 = (image1 - c_min)/(c_max - c_min)*255
    image0 = image0.astype('uint8')
    image1 = image1.astype('uint8')

    u = coarsen(array=u, coarseness=coarseness)
    v = coarsen(array=v, coarseness=coarseness)
    U_median = np.median(u)
    V_median = np.median(v)

    x_step = (time1 - time0).seconds*U_median/(250*coarseness)
    y_step = (time1 - time0).seconds*V_median/(250*coarseness)
    x_step = int(np.round(x_step))
    y_step = int(np.round(y_step))

    p0 = cv2.goodFeaturesToTrack(
        image0,
        **feature_params)
    p0_resh = p0.reshape(p0.shape[0], p0.shape[2])

    means = ndimage.filters.uniform_filter(image0.astype('float'), (var_size, var_size))
    second_moments = ndimage.filters.uniform_filter(image0.astype('float')**2, (var_size, var_size))
    variances = second_moments - means**2
    win_vars = ndimage.filters.gaussian_filter(variances, sigma=var_sig)
    win_vars = win_vars[(p0[:, :, 1].astype('int'), p0[:, :, 0].astype('int'))].ravel()

    p0 = p0[win_vars > var_thresh]
    p0_resh = p0.reshape(p0.shape[0], p0.shape[2])

    p1_guess = p0 + np.array([x_step, y_step])[None, None, :]
    p1, status, err = cv2.calcOpticalFlowPyrLK(
        image0, image1, p0, p1_guess, **lk_params)
    status = status.ravel().astype('bool')
    p1 = p1[status, :, :]
    p0 = p0[status, :, :]
    in_domain = np.logical_and(p1>0, p1<image0.shape[0]).all(axis=-1) # assumes clouds0 is square
    in_domain = in_domain.ravel()
    p1 = p1[in_domain, :, :]
    p0 = p0[in_domain, :, :]

    err = err.ravel()[status]
    p0_resh = p0.reshape(p0.shape[0], p0.shape[2])
    p1_resh = p1.reshape(p1.shape[0], p1.shape[2])

    time_step0 = (time1 - time0).seconds

    u_of = (p1_resh[:, 0] - p0_resh[:, 0])*(250*coarseness/(time_step0))
    v_of = (p1_resh[:, 1] - p0_resh[:, 1])*(250*coarseness/(time_step0))

    u_mu = u.mean()
    u_sd = np.sqrt(u.var())
    v_mu = v.mean()
    v_sd = np.sqrt(v.var())
    good_wind = ((u > u_mu - u_sd*sd_num) & (u < u_mu + u_sd*sd_num) &
                 (v > v_mu - v_sd*sd_num) & (v < v_mu + v_sd*sd_num))
    u = u[good_wind]
    v = v[good_wind]
    p0_good = p0_resh[good_wind]
    return u, v, p0_good


def remove_divergence(u, v):
    ## this could be improved by increasing the order in V
    c_shape = u.shape
    mesh = fe.RectangleMesh(fe.Point(0,0),
                            fe.Point(c_shape[0] - 1, c_shape[1] - 1),
                            c_shape[0] - 1, c_shape[1] - 1)
    c_shape = u.shape
    mesh = fe.RectangleMesh(fe.Point(0,0),
                            fe.Point(c_shape[0] - 1, c_shape[1] - 1),
                            c_shape[0] - 1, c_shape[1] - 1)
    V = fe.FunctionSpace(mesh, 'P', 1)

    V_div = divergence(u, v, 1, 1)
    ff = fe.Function(V)
    d2v_map = fe.dof_to_vertex_map(V)
    array_ff = V_div.ravel()
    array_ff = array_ff[d2v_map]
    ff.vector().set_local(array_ff)
    uu = fe.TrialFunction(V)
    vv = fe.TestFunction(V)
    a = fe.dot(fe.grad(uu), fe.grad(vv))*fe.dx
    L = ff*vv*fe.dx
    uu = fe.Function(V)
    fe.solve(a == L, uu)
    phi = uu.compute_vertex_values().reshape(c_shape)
    grad_phi = np.gradient(phi, 1, 1)
    u_corrected = u + grad_phi[1]
    v_corrected = v + grad_phi[0]
    sigma = 2
    u_corrected = sp.filters.gaussian_filter(u_corrected, sigma=sigma)
    v_corrected = sp.filters.gaussian_filter(v_corrected, sigma=sigma)
    return u_corrected, v_corrected


def main_only_sat(sat, x, y, domain_shape, domain_crop_cols, domain_crop_shape,
                  U, U_shape, U_crop_cols, U_crop_shape,
                  V, V_shape, V_crop_cols, V_crop_shape,
                  start_time, end_time, dx, dy, C_max,
                  assimilation_grid_size,
                  localization_letkf, sat_inflation,
                  sat_sig, ens_size,
                  wind_sigma, wind_size,
                  CI_sigma,
                  client, flat_error_domain,
                  wind_inflation,
                  wind_sat_sig,
                  pert_sigma, pert_mean,
                  rf_eig, rf_vectors, rf_approx_var,
                  edge_weight,
                  l_sat=None, l_x=None, l_y=None, l_shape=None,
                  l_U=None, l_U_shape=None, l_V=None, l_V_shape=None):
    if (start_time is None) & (end_time is None):
        sat_time_range = sat.index
    else:
        sat_time_range = (pd.date_range(start_time, end_time, freq='15 min')
                          .tz_localize('MST'))
        sat_time_range = sat_time_range.intersection(sat.index)
    sat_crop = sat[domain_crop_cols]
    sat_crop.colums = np.arange(domain_crop_cols.size, dtype='int')
    x_crop = x[domain_crop_cols]
    y_crop = y[domain_crop_cols]
    U_crop = U[U_crop_cols]
    U_crop.columns = np.arange(U_crop_cols.size, dtype='int')
    V_crop = V[V_crop_cols]
    V_crop.columns = np.arange(V_crop_cols.size, dtype='int')
    assimilation_positions, assimilation_positions_2d, full_positions_2d = (
        assimilation_position_generator(domain_shape, assimilation_grid_size))
    q = sat_crop.loc[sat_time_range[0]].values.reshape(domain_shape)
    # ensemble = ensemble_creator(
    #     q, CI_sigma=CI_sigma, wind_size=wind_size,
    #     wind_sigma=wind_sigma, ens_size=ens_size)
    ensemble = ensemble_creator_wind(
        q, CI_sigma=CI_sigma, wind_size=wind_size,
        wind_sigma=wind_sigma, ens_size=ens_size)
    ens_shape = ensemble.shape
    ensemble_15 = pd.DataFrame(data=ensemble.ravel()[None, :]*np.nan,
                               index=[sat_time_range[0]])
    ensemble_30 = ensemble_15.copy()
    ensemble_45 = ensemble_15.copy()
    ensemble_analy = ensemble_15.copy()

    noise_init = noise_fun(domain_shape)
    noise = noise_init.copy()

    # background = pd.DataFrame(data=ensemble.mean(axis=1)[None, :]*np.nan,
    #                           index=[sat_time_range[0]])
    # analysis = background.copy()*np.nan
    advected_15 = pd.DataFrame(data=q.ravel()[None, :]*np.nan,
                               index=[sat_time_range[0]])
    advected_30 = advected_15.copy()
    advected_45 = advected_15.copy()
    # used_sat = advected_15.copy()
    # background_sd = pd.DataFrame(data=np.sqrt(ensemble.var(axis=1)[None, :]),
    #                              index=[sat_time_range[0]])
    # analysis_sd = background_sd.copy()*np.NaN

    # background_error = advected_15.copy()*np.nan
    # analysis_error = advected_15.copy()*np.nan
    # advected_error = advected_15.copy()*np.nan

    # col_names = ['back', 'analy', 'advected', 'back_sd', 'analy_sd']
    # error_stats = pd.DataFrame(data=np.ones(len(col_names))[None, :]*np.nan,
    #                            index=[sat_time_range[0]],
    #                            columns=col_names)

    # background_ni = background.copy()
    # background_error_ni = background_error.copy()
    # background_sd_ni = background_sd.copy()
    # advected_ni = advected.copy()
    # advected_error_ni = advected_error.copy()
    # used_sat_ni = used_sat.copy()

    # col_names = ['back', 'advected', 'back_sd']
    # error_stats_ni = pd.DataFrame(data=np.ones(len(col_names))[None, :]*np.nan,
    #                               index=[sat_time_range[0]],
    #                               columns=col_names)

    for time_index in range(sat_time_range.size - 1):
        sat_time = sat_time_range[time_index]
        print(sat_time)
        # *** manually adjust wind by .5***
        int_index_wind = U_crop.index.get_loc(sat_time, method='pad')
        this_U = U_crop.iloc[int_index_wind].values.reshape(U_crop_shape)
        this_V = V_crop.iloc[int_index_wind].values.reshape(V_crop_shape)
        cx = abs(U_crop.values).max()
        cy = abs(V_crop.values).max()
        T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)/C_max))
        dt = (5*60)/T_steps
        num_of_advec = int((
            (sat_time_range[time_index + 1].value) -
            (sat_time_range[time_index].value))*(10**(-9)/(60*5))/3)

        temp_ensemble = ensemble.copy()
        temp_noise = noise.copy()

        print('15 min')
        for n in range(3):
            # print('advection_number_15: ' + str(n))
            q, temp_noise, temp_ensemble = advect_5min_distributed(
                q, temp_noise, temp_ensemble, dt, this_U, dx,
                this_V, dy, T_steps, wind_size, client)

            # add pertubation
            if pert_sigma != 0:
                temp_ensemble[wind_size:] = preturb_irradiance(
                    temp_ensemble[wind_size:], domain_crop_shape,
                    edge_weight, pert_mean, pert_sigma,
                    rf_approx_var, rf_eig, rf_vectors)
                # average = temp_ensemble[wind_size:].mean(axis=1)
                # average = average.reshape(domain_shape)
                # target = ski_filters.sobel(average)
                # target = target/target.max()
                # target[target < 0.1] = 0
                # target = sp.ndimage.gaussian_filter(target, sigma=4)
                # target = target/target.max()
                # cloud_target = 1 - average
                # cloud_target = (cloud_target/cloud_target.max()).clip(min=0,
                #                                                       max=1)
                # target = np.maximum(cloud_target, target*edge_weight)
                # target = target/target.max()
                # target = sp.ndimage.gaussian_filter(target, sigma=5)
                # target = target.ravel()
                # sample = np.random.randn(rf_eig.size, ens_size)
                # sample = rf_vectors.dot(np.sqrt(rf_eig[:, None])*sample)
                # target_mean = target.mean()
                # target_var = (target**2).mean()
                # cor_mean = pert_mean/target_mean
                # cor_sd = pert_sigma/np.sqrt(rf_approx_var*target_var)
                # temp_ensemble[wind_size:] = (
                #     temp_ensemble[wind_size:] +
                #     (cor_sd*sample + cor_mean)*target[:, None])
        ensemble_15.loc[sat_time_range[time_index] + pd.Timedelta('15min')] = (
            temp_ensemble.ravel())
        advected_15.loc[sat_time_range[time_index] + pd.Timedelta('15min')] = (
            q.ravel())
        if num_of_advec == 1:
            ensemble = temp_ensemble.copy()
            noise = temp_noise.copy()

        print('30 min')
        for n in range(3):
            # print('advection_number_30: ' + str(n))
            q, temp_noise, temp_ensemble = advect_5min_distributed(
                q, temp_noise, temp_ensemble, dt, this_U, dx,
                this_V, dy, T_steps, wind_size, client)

            # add pertubation
            if pert_sigma != 0:
                temp_ensemble[wind_size:] = preturb_irradiance(
                    temp_ensemble[wind_size:], domain_crop_shape,
                    edge_weight, pert_mean, pert_sigma,
                    rf_approx_var, rf_eig, rf_vectors)
                # average = temp_ensemble[wind_size:].mean(axis=1)
                # average = average.reshape(domain_shape)
                # target = ski_filters.sobel(average)
                # target = target/target.max()
                # target[target < 0.1] = 0
                # target = sp.ndimage.gaussian_filter(target, sigma=4)
                # target = target/target.max()
                # cloud_target = 1 - average
                # cloud_target = (cloud_target/cloud_target.max()).clip(min=0,
                #                                                       max=1)
                # target = np.maximum(cloud_target, target*edge_weight)
                # target = target/target.max()
                # target = sp.ndimage.gaussian_filter(target, sigma=5)
                # target = target.ravel()
                # sample = np.random.randn(rf_eig.size, ens_size)
                # sample = rf_vectors.dot(np.sqrt(rf_eig[:, None])*sample)
                # target_mean = target.mean()
                # target_var = (target**2).mean()
                # cor_mean = pert_mean/target_mean
                # cor_sd = pert_sigma/np.sqrt(rf_approx_var*target_var)
                # temp_ensemble[wind_size:] = (
                #     temp_ensemble[wind_size:] +
                #     (cor_sd*sample + cor_mean)*target[:, None])
        ensemble_30.loc[sat_time_range[time_index] + pd.Timedelta('30min')] = (
            temp_ensemble.ravel())
        advected_30.loc[sat_time_range[time_index] + pd.Timedelta('30min')] = (
            q.ravel())
        if num_of_advec == 2:
            ensemble = temp_ensemble.copy()
            noise = temp_noise.copy()

        print('45 min')
        for n in range(3):
            # print('advection_number_45: ' + str(n))
            q, temp_noise, temp_ensemble = advect_5min_distributed(
                q, temp_noise, temp_ensemble, dt, this_U, dx,
                this_V, dy, T_steps, wind_size, client)

            # add pertubation
            if pert_sigma != 0:
                temp_ensemble[wind_size:] = preturb_irradiance(
                    temp_ensemble[wind_size:], domain_crop_shape,
                    edge_weight, pert_mean, pert_sigma,
                    rf_approx_var, rf_eig, rf_vectors)
                # average = temp_ensemble[wind_size:].mean(axis=1)
                # average = average.reshape(domain_shape)
                # target = ski_filters.sobel(average)
                # target = target/target.max()
                # target[target < 0.1] = 0
                # target = sp.ndimage.gaussian_filter(target, sigma=4)
                # target = target/target.max()
                # cloud_target = 1 - average
                # cloud_target = (cloud_target/cloud_target.max()).clip(min=0, max=1)
                # target = np.maximum(cloud_target, target*edge_weight)
                # target = target/target.max()
                # target = sp.ndimage.gaussian_filter(target, sigma=5)
                # target = target.ravel()
                # sample = np.random.randn(rf_eig.size, ens_size)
                # sample = rf_vectors.dot(np.sqrt(rf_eig[:, None])*sample)
                # target_mean = target.mean()
                # target_var = (target**2).mean()
                # cor_mean = pert_mean/target_mean
                # cor_sd = pert_sigma/np.sqrt(rf_approx_var*target_var)
                # temp_ensemble[wind_size:] = (
                #     temp_ensemble[wind_size:] +
                #     (cor_sd*sample + cor_mean)*target[:, None])
        ensemble_45.loc[sat_time_range[time_index] + pd.Timedelta('45min')] = (
            temp_ensemble.ravel())
        advected_45.loc[sat_time_range[time_index] + pd.Timedelta('45min')] = (
            q.ravel())
        if num_of_advec == 3:
            ensemble = temp_ensemble.copy()
            noise = temp_noise.copy()

        ## for whole image assimilation ##
        sat_time = sat_time_range[time_index + 1]
        print('Assimilation')
        print(sat_time)

        ## collect advected info

        # advected.loc[sat_time] = q.ravel()
        q = sat_crop.loc[sat_time].values.reshape(domain_shape)
        # advected_error.loc[sat_time] = (
        #     advected.loc[sat_time].values - q.ravel())
        # temp = advected_error.loc[sat_time].values[flat_error_domain]
        # error_stats.loc[sat_time, 'advected'] = np.sqrt((temp**2).mean())

        # used_sat.loc[sat_time] = q.ravel()


        # # collect background info
        # background.loc[sat_time] = ensemble.mean(axis=1)
        # background_error.loc[sat_time] = (
        #     ensemble[wind_size:].mean(axis=1) - q.ravel())
        # ens_test = ensemble[
        #     flat_error_domain + wind_size].mean(axis=1).ravel()
        # temp = ens_test - q.ravel()[flat_error_domain]
        # error_stats.loc[sat_time, 'back'] = np.sqrt((temp**2).mean())
        # temp = np.sqrt(ensemble.var(axis=1))
        # background_sd.loc[sat_time] = temp
        # error_stats.loc[sat_time, 'back_sd'] = temp[
        #     flat_error_domain + wind_size].mean()

        # replace noisy areas (Does not maintain variace)
        noise = (noise - noise.min())
        noise = noise/noise.max()
        noise = noise.ravel()
        ensemble[wind_size:] = (q.ravel()[:, None]*noise[:, None] +
                                ensemble[wind_size:, :]*(1 - noise[:, None]))

        # assimilate satellite image into wind portion of ensemble
        wind_assimilate_positions = np.concatenate(
            [np.arange(wind_size), flat_error_domain])
        ensemble[:wind_size] = assimilate_wind(
            ensemble=ensemble[wind_assimilate_positions],
            observations=q.ravel()[flat_error_domain],
            flat_sensor_indices=None,
            R_inverse=1/wind_sat_sig**2,
            inflation=wind_inflation,
            wind_size=wind_size)

        # assimilate satellite image into satellite portion of ensemble
        ensemble[wind_size:] = assimilate(
            ensemble=ensemble[wind_size::],
            observations=q.ravel(),
            flat_sensor_indices=None, R_inverse=1/sat_sig**2,
            inflation=sat_inflation,
            domain_shape=domain_shape,
            localization_length=localization_letkf,
            assimilation_positions=assimilation_positions,
            assimilation_positions_2d=assimilation_positions_2d,
            full_positions_2d=full_positions_2d)

        # collect analysis info
        ensemble_analy.loc[sat_time] = ensemble.ravel()
        # analysis.loc[sat_time] = ensemble.mean(axis=1)
        # analysis_error.loc[sat_time] = (
        #     ensemble[wind_size:].mean(axis=1) - q.ravel())
        # ens_test = ensemble[
        #     flat_error_domain + wind_size].mean(axis=1)
        # temp = ens_test - q.ravel()[flat_error_domain]
        # error_stats.loc[sat_time, 'analy'] = np.sqrt((temp**2).mean())
        # temp = np.sqrt(ensemble.var(axis=1))
        # analysis_sd.loc[sat_time]= temp
        # error_stats.loc[sat_time, 'analy_sd'] = temp[
        #     flat_error_domain + wind_size].mean()

    # ensemble_15 = ensemble_15.dropna()
    # ensemble_30 = ensemble_30.dropna()
    # ensemble_45 = ensemble_45.dropna()
    # ensemble_analy = ensemble_analy.dropna()
    # advected_15 = advected_15.dropna()
    # advected_30 = advected_30.dropna()
    # advected_45 = advected_45.dropna()
    to_return = (ensemble_15, ensemble_30, ensemble_45,
                 ensemble_analy, ens_shape,
                 advected_15, advected_30, advected_45)
    return to_return


## need to put everything into a pandas objects
def only_durring_sat(sat, x, y, domain_shape,
                     U, U_shape, V, V_shape,
                     start_time, end_time, dx, dy, C_max,
                 assimilation_grid_size,
                     localization_letkf, sat_inflation,
                     sat_sig, ens_size,
                     wind_sigma, wind_size,
                     client,
                     oi_sensor_sig,
                     oi_sat_sig,
                     oi_localization,
                     oi_inflation,
                     sensor_data,
                     sensor_loc,
                     sensor_sig,
                     sensor_inflation,
                     CI_sigma, location,
                     cloud_height, sat_azimuth,
                     sat_elevation):
    sat_time_range = (pd.date_range(start_time, end_time, freq='15 min')
                      .tz_localize('MST'))
    sat_time_range = sat_time_range.intersection(sat.index)
    sat_lats, sat_lons = prep.lcc_to_sphere(x, y)
    sensor_loc.sort_values(by='id', inplace=True)
    sensor_loc_test = sensor_loc[sensor_loc.test==True]
    sensor_loc_assim = sensor_loc[sensor_loc.test==False]
    sensor_data_test = sensor_data[sensor_loc_test.id.values]
    sensor_data_assim = sensor_data[sensor_loc_assim.id.values]
    flat_sensor_loc_test = find_flat_loc(
        sat_lats, sat_lons, sensor_loc_test)
    flat_sensor_loc_assim = find_flat_loc(
        sat_lats, sat_lons, sensor_loc_assim)
    assimilation_positions, assimilation_positions_2d, full_positions_2d = (
        assimilation_position_generator(domain_shape, assimilation_grid_size))
    q = sat.loc[sat_time_range[0]].values.reshape(domain_shape)
    ensemble = ensemble_creator(
        q, CI_sigma=CI_sigma, wind_size=wind_size,
        wind_sigma=wind_sigma, ens_size=ens_size)

    noise_init = noise_fun(domain_shape)
    noise = noise_init.copy()

    background = pd.DataFrame(data=ensemble.mean(axis=1)[None, :],
                              index=[sat_time_range[0]])
    analysis = background.copy()
    advected = pd.DataFrame(data=q.ravel()[None, :],
                            index=[sat_time_range[0]])
    OI = advected.copy()
    used_sat = advected.copy()

    col_names = ['sat', 'back', 'analy_sat', 'analy_sat_sens', 'oi',
                 'trace_back', 'trace_analy']
    error_stats = pd.DataFrame(data=np.ones(len(col_names))[None, :]*np.nan,
                               index=[sat_time_range[0]],
                               columns=col_names)

    # background_error = np.zeros([1, sensor_loc_test.id.size])
    # analysis_error_sat = background_error.copy()
    # analysis_error_sat_sens = background_error.copy()
    # sat_error = background_error.copy()
    # oi_error = background_error.copy()

    # trace_back =  np.array([ensemble.var(axis=1).mean()])
    # trace_analy = np.array([np.nan])

    for time_index in range(sat_time_range.size - 1):
        sat_time = sat_time_range[time_index] # time before advection
        print('sat_time: ' + str(sat_time))
        # *** manually adjust wind by .5***
        int_index_wind = U.index.get_loc(sat_time, method='pad')
        this_U = U.iloc[int_index_wind].values.reshape(U_shape) # + .5
        this_V = V.iloc[int_index_wind].values.reshape(V_shape)
        cx = abs(U.values).max()
        cy = abs(V.values).max()
        T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)/C_max))
        dt = (5*60)/T_steps
        advection_number = int((
            (sat_time_range[time_index + 1].value) -
            (sat_time_range[time_index].value))*(10**(-9)/(60*5)))
        for n in range(advection_number):
            print('advection_number: ' + str(n))
            q, noise, ensemble = advect_5min_distributed(
                q, noise, ensemble, dt, this_U, dx,
                this_V, dy, T_steps, wind_size, client)
        # for whole image assimilation

        sat_time = sat_time_range[time_index + 1] # time after advection

        flat_correct = get_flat_correct(
            cloud_height=cloud_height, dx=dx, dy=dy,
            domain_shape=domain_shape, sat_azimuth=sat_azimuth,
            sat_elevation=sat_elevation,
            location=location, sensor_time=sat_time)
        this_flat_sensor_loc_test = flat_sensor_loc_test + flat_correct
        this_flat_sensor_loc_assim = flat_sensor_loc_assim + flat_correct

        # error from background
        ens_test = ensemble[
            this_flat_sensor_loc_test + wind_size].mean(axis=1)
        temp = ens_test - sensor_data_test.ix[sat_time].values
        error_stats.loc[sat_time, 'back'] = np.sqrt((temp**2).mean())
        # background_error = np.concatenate(
        #     [background_error, temp[None, :]], axis=0)

        advected.loc[sat_time] = q.ravel()
        q = sat.loc[sat_time].values.reshape(domain_shape)
        used_sat.loc[sat_time] = q.ravel()

        # error from just using sat
        temp = (q.ravel()[this_flat_sensor_loc_test] -
                sensor_data_test.ix[sat_time].values)
        error_stats.loc[sat_time, 'sat'] = np.sqrt((temp**2).mean())
        # sat_error = np.concatenate(
        #     [sat_error, temp[None, :]], axis=0)

        noise = (noise - noise.min())
        noise = noise/noise.max()
        noise = noise.ravel()
        background.loc[sat_time] = ensemble.mean(axis=1)

        error_stats.loc[sat_time, 'trace_back'] = ensemble.var(axis=1).mean()
        # trace_back = np.concatenate(
        #     [trace_back,  np.array([ensemble.var(axis=1).mean()])])
        ensemble[wind_size:] = (q.ravel()[:, None]*noise[:, None] +
                                 ensemble[wind_size:, :]*(1 - noise[:, None]))
        ensemble[wind_size:] = assimilate(
            ensemble=ensemble[wind_size:],
            observations=q.ravel(),
            flat_sensor_indices=None, R_inverse=1/sat_sig**2,
            inflation=sat_inflation,
            domain_shape=domain_shape,
            localization_length=localization_letkf,
            assimilation_positions=assimilation_positions,
            assimilation_positions_2d=assimilation_positions_2d,
            full_positions_2d=full_positions_2d)

        # error from assimilating new sat
        ens_test = ensemble[
            this_flat_sensor_loc_test + wind_size].mean(axis=1)
        temp = ens_test - sensor_data_test.ix[sat_time].values
        error_stats.loc[sat_time, 'analy_sat'] = np.sqrt((temp**2).mean())
        # analysis_error_sat = np.concatenate(
        #     [analysis_error_sat, temp[None, :]], axis=0)

        # for sensor assimilation
        ensemble = assimilate(ensemble, sensor_data_assim.ix[sat_time],
                              this_flat_sensor_loc_assim + wind_size,
                              1/sensor_sig**2, inflation=sensor_inflation)
        error_stats.loc[sat_time, 'trace_analy'] = ensemble.var(axis=1).mean()
        # trace_analy = np.concatenate(
        #     [trace_analy,  np.array([ensemble.var(axis=1).mean()])])
        analysis.loc[sat_time] = ensemble.mean(axis=1)
        # analysis = np.concatenate(
        #     [analysis, ensemble.mean(axis=1)[None, :]], axis=0)

        noise = noise_init.copy()

        # error from assimilating new sensor data as well
        ens_test = ensemble[
            this_flat_sensor_loc_test + wind_size].mean(axis=1)
        temp = ens_test - sensor_data_test.ix[sat_time].values
        error_stats.loc[sat_time, 'analy_sat_sens'] = np.sqrt((temp**2).mean())
        # analysis_error_sat_sens = np.concatenate(
        #     [analysis_error_sat_sens, temp[None, :]], axis=0)

        # for optimal interpolation
        this_OI = optimal_interpolation(
            q.ravel(), oi_sat_sig, sensor_data_assim.ix[sat_time], oi_sensor_sig,
            q.ravel(), this_flat_sensor_loc_assim,
            oi_localization, oi_inflation)
        OI.loc[sat_time] = this_OI
        # OI = np.concatenate(
        #     [OI, this_OI[None, :]], axis=0)

        # error from oi only
        temp = (this_OI[this_flat_sensor_loc_test] -
                sensor_data_test.ix[sat_time].values)
        error_stats.loc[sat_time, 'oi'] = np.sqrt((temp**2).mean())
        # oi_error = np.concatenate(
        #     [oi_error, temp[None, :]], axis=0)

    to_return = (analysis, background, advected, used_sat, OI, sat_time_range,
                 error_stats)
    return to_returnn

def only_oi(sat, x, y, domain_shape,
            start_time, end_time,
            oi_sensor_sig,
            oi_sat_sig,
            oi_localization,
            oi_inflation,
            sensor_data,
            sensor_loc,
            location,
            cloud_height, sat_azimuth,
            sat_elevation):
    sat_time_range = (pd.date_range(start_time, end_time, freq='15 min')
                      .tz_localize('MST'))
    sat_time_range = sat_time_range.intersection(sat.index)
    sat_lats, sat_lons = prep.lcc_to_sphere(x, y)
    sensor_loc = sensor_loc.sort_values(by='id', inplace=False)
    sensor_loc_test = sensor_loc[sensor_loc.test==True]
    sensor_loc_assim = sensor_loc[sensor_loc.test==False]
    sensor_data_test = sensor_data[sensor_loc_test.id.values]
    sensor_data_assim = sensor_data[sensor_loc_assim.id.values]
    flat_sensor_loc_test = find_flat_loc(
        sat_lats, sat_lons, sensor_loc_test)
    flat_sensor_loc_assim = find_flat_loc(
        sat_lats, sat_lons, sensor_loc_assim)
    assimilation_positions, assimilation_positions_2d, full_positions_2d = (
        assimilation_position_generator(domain_shape, assimilation_grid_size))
    q = sat.loc[sat_time_range[0]].values.reshape(domain_shape)
    # small CI_sigma not interested in fixing irradiance
    used_sat = q[None, :, :].copy()
    OI = advected.copy().ravel()[None, :]





    background_error = np.zeros([1, sensor_loc_test.id.size])
    analysis_error = background_error.copy()




    for time_index in range(sat_time_range.size - 1):
        sat_time = sat_time_range[time_index]
        print('sat_time: ' + str(sat_time))
        # *** manually adjust wind by .5***
        int_index_wind = U.index.get_loc('2014-04-15 13:45:00', method='pad')
        this_U = U.iloc[int_index_wind].values.reshape(U_shape) + .5
        this_V = V.iloc[int_index_wind].values.reshape(V_shape)
        cx = abs(U.values).max()
        cy = abs(V.values).max()
        T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)/C_max))
        dt = (5*60)/T_steps
        advection_number = int((
            (sat_time_range[time_index + 1].value) -
            (sat_time_range[time_index].value))*(10**(-9)/(60*5)))
        for n in range(advection_number):
            print('advection_number: ' + str(n))
            q, noise, ensemble = advect_5min_distributed(
                q, noise, ensemble, dt, this_U, dx,
                this_V, dy, T_steps, wind_size, client)
        # for whole image assimilation
        advected = np.concatenate([advected, q[None, :, :]], axis=0)
        q = sat.loc[sat_time_range[time_index + 1]].values.reshape(domain_shape)
        used_sat = np.concatenate([used_sat, q[None, :, :]], axis=0)
        noise = (noise - noise.min())
        noise = noise/noise.max()
        noise = noise.ravel()
        background = np.concatenate(
                [background, ensemble.mean(axis=1)[None,:]], axis=0)
        trace_back = np.concatenate(
            [trace_back,  np.array([ensemble.var(axis=1).mean()])])
        ensemble[wind_size:] = (q.ravel()[:, None]*noise[:, None] +
                                 ensemble[wind_size:, :]*(1 - noise[:, None]))
        ensemble[wind_size:] = assimilate(
            ensemble=ensemble[wind_size:],
            observations=q.ravel(),
            flat_sensor_indices=None, R_inverse=1/sat_sig**2,
            inflation=sat_inflation,
            domain_shape=domain_shape,
            localization_length=localization_letkf,
            assimilation_positions=assimilation_positions,
            assimilation_positions_2d=assimilation_positions_2d,
            full_positions_2d=full_positions_2d)
        # for optimal interpolation
        flat_correct = get_flat_correct(
            cloud_height=cloud_height, dx=dx, dy=dy,
            domain_shape=domain_shape, sat_azimuth=sat_azimuth,
            sat_elevation=sat_elevation,
            location=location, sensor_time=sat_time)
        this_flat_sensor_loc_test = flat_sensor_loc_test + flat_correct
        this_flat_sensor_loc_assim = flat_sensor_loc_assim + flat_correct
        this_OI = optimal_interpolation(
            q.ravel(), oi_sat_sig, sensor_data_assim.ix[sat_time], oi_sensor_sig,
            q.ravel(), this_flat_sensor_loc_assim,
            oi_localization, oi_inflation)
        OI = np.concatenate(
            [OI, this_OI[None, :]], axis=0)
        # for sensor assimilation
        ensemble = assimilate(ensemble, sensor_data_assim.ix[sat_time],
                              this_flat_sensor_loc_assim + wind_size,
                              1/sensor_sig**2, inflation=sensor_inflation)
        trace_analy = np.concatenate(
            [trace_analy,  np.array([ensemble.var(axis=1).mean()])])
        analysis = np.concatenate(
            [analysis, ensemble.mean(axis=1)[None, :]], axis=0)
        noise = noise_init.copy()

    to_return = (analysis, background, advected, used_sat, OI, sat_time_range,
                 trace_back, trace_analy)
    return to_return


def main(
        sat, x, y, domain_shape,
        U, U_shape, V, V_shape,
        start_time, end_time, dx, dy, C_max,
        assimilation_grid_size,
        localization_letkf, sat_inflation,
        sat_sig, ens_size,
        wind_sigma, wind_size,
        client,
        sensor_data,
        sensor_loc,
        sensor_sig,
        sensor_inflation,
        CI_sigma, coeff_sigma, location,
        cloud_height, sat_azimuth,
        sat_elevation):
    sat_lats, sat_lons = prep.lcc_to_sphere(x, y)
    sat_time_range = (pd.date_range(start_time, end_time, freq='15 min')
                      .tz_localize('MST'))
    sat_time_range = sat_time_range.intersection(sat.index)
    assimilation_positions, assimilation_positions_2d, full_positions_2d = (
        assimilation_position_generator(domain_shape, assimilation_grid_size))
    q = sat.loc[sat_time_range[0]].values.reshape(domain_shape)
    sensor_loc = sensor_loc.sort_values(by='id', inplace=False)
    sensor_loc_test = sensor_loc[sensor_loc.test==True]
    sensor_loc_assim = sensor_loc[sensor_loc.test==False]
    sensor_data_test = sensor_data[sensor_loc_test.id.values]
    sensor_data_assim = sensor_data[sensor_loc_assim.id.values]
    flat_sensor_loc_test = find_flat_loc(
        sat_lats, sat_lons, sensor_loc_test)
    flat_sensor_loc_assim = find_flat_loc(
        sat_lats, sat_lons, sensor_loc_assim)
    ensemble = ensemble_creator(
        q, CI_sigma=CI_sigma, wind_size=wind_size,
        wind_sigma=wind_sigma, ens_size=ens_size)
    init_coeff = np.array([2.36, -6.2, 6.22, -2.63, -0.58, 1])
    coeff_size = coeff_sigma.size
    rand_coeffs = np.random.normal(loc=0, scale=coeff_sigma[0], size=ens_size)
    rand_coeffs = np.concatenate(
        [rand_coeffs[None, :],
         np.random.normal(loc=0, scale=coeff_sigma[1], size=ens_size)[None, :]],
        axis=0)
    ensemble = np.concatenate([rand_coeffs, ensemble], axis=0)
    csi_ensemble = ensemble[(coeff_size + wind_size):].copy()
    for ens_num in range(ens_size):
        this_coeff_preturb = np.array(
            [0, 0, 0, 0,
             csi_ensemble[1, ens_num] - csi_ensemble[0, ens_num],
             csi_ensemble[0, ens_num]])
        csi_ensemble[:, ens_num] = np.polyval(
            init_coeff + this_coeff_preturb, csi_ensemble[:, ens_num])
    csi_ensemble = np.concatenate(
        [ensemble[:(coeff_size + wind_size)], csi_ensemble], axis=0)
    ####
    plt.figure()
    im = plt.pcolormesh(
        csi_ensemble[(coeff_size + wind_size):].mean(axis=1).reshape(domain_shape),
        cmap='Blues')
    plt.colorbar(im)
    plt.title('initial: ')
    plt.axis('equal')
    ####
    trace_back =  np.array([csi_ensemble.var(axis=1).mean()])
    trace_analy = np.array([np.nan])
    noise_init = noise_fun(domain_shape)
    noise = noise_init.copy()
    background = ensemble.mean(axis=1)[None, :]
    analysis = background.copy()
    advected = q[None, :, :].copy()
    background_error = np.zeros([1, sensor_loc_test.id.size])
    analysis_error = background_error.copy()
    for time_index in range(sat_time_range.size - 1):
        sat_time = sat_time_range[time_index]
        print('sat_time: ' + str(sat_time))
        # *** manually adjust wind by .5***
        int_index_wind = U.index.get_loc('2014-04-15 13:45:00', method='pad')
        this_U = U.iloc[int_index_wind].values.reshape(U_shape) + .5
        this_V = V.iloc[int_index_wind].values.reshape(V_shape)
        cx = abs(U.values).max()
        cy = abs(V.values).max()
        T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)/C_max))
        dt = (5*60)/T_steps
        advection_number = int((
            (sat_time_range[time_index + 1].value) -
            (sat_time_range[time_index].value))*(10**(-9)/(60*5)))
        for n in range(advection_number):
            sensor_time = sat_time + pd.Timedelta('5 min')*n
            print('sensor_time: ' + str(sensor_time))
            q, noise, ensemble = advect_5min_distributed_coeff(
                q, noise, ensemble, dt, this_U, dx,
                this_V, dy, T_steps, coeff_size, wind_size, client)
            flat_correct = get_flat_correct(
                cloud_height=cloud_height, dx=dx, dy=dy,
                domain_shape=domain_shape, sat_azimuth=sat_azimuth,
                sat_elevation=sat_elevation,
                location=location, sensor_time=sat_time)
            this_flat_sensor_loc_test = flat_sensor_loc_test + flat_correct
            csi_ensemble = ensemble[(coeff_size + wind_size):].copy()
            for ens_num in range(ens_size):
                this_coeff_preturb = np.array(
                    [0, 0, 0, 0,
                     csi_ensemble[1, ens_num] - csi_ensemble[0, ens_num],
                     csi_ensemble[0, ens_num]])
                csi_ensemble[:, ens_num] = np.polyval(
                    init_coeff + this_coeff_preturb, csi_ensemble[:, ens_num])
            csi_ensemble = np.concatenate(
                [ensemble[:(coeff_size + wind_size)], csi_ensemble], axis=0)
            trace_back = np.concatenate(
                [trace_back,  np.array([csi_ensemble.var(axis=1).mean()])])
            ens_test = csi_ensemble[
                this_flat_sensor_loc_test + wind_size + coeff_size].mean(axis=1)
            temp = ens_test - sensor_data_test.ix[sensor_time].values
            background_error = np.concatenate(
                [background_error, temp[None, :]], axis=0)
            background = np.concatenate(
                [background, ensemble.mean(axis=1)[None,:]], axis=0)
            this_flat_sensor_loc_assim = flat_sensor_loc_assim + flat_correct
            if n != advection_number - 1:

                # csi_ensemble = ensemble[(coeff_size + wind_size):].copy()
                # for ens_num in range(ens_size):
                #     csi_ensemble[:, ens_num] = np.polyval(
                #         ensemble[:coeff_size, ens_num], csi_ensemble[:, ens_num])
                # csi_ensemble = np.concatenate(
                #     [ensemble[:(coeff_size + wind_size)], csi_ensemble], axis=0)
                ####
                plt.figure()
                im = plt.pcolormesh(
                    csi_ensemble[(coeff_size + wind_size):].mean(axis=1).reshape(domain_shape),
                    cmap='Blues')
                plt.colorbar(im)
                plt.title('background: ')
                plt.axis('equal')
                ####
                csi_ensemble = assimilate(csi_ensemble, sensor_data_assim.ix[sensor_time],
                                  this_flat_sensor_loc_assim + wind_size + coeff_size,
                                  1/sensor_sig**2, inflation=sensor_inflation)
                ####
                plt.figure()
                im = plt.pcolormesh(
                    csi_ensemble[(coeff_size + wind_size):].mean(axis=1).reshape(domain_shape),
                    cmap='Blues')
                plt.colorbar(im)
                plt.title('Analysis: ')
                plt.axis('equal')
                ####
                # for ens_num in range(ens_size):
                #     ensemble[:coeff_size, ens_num] = np.polyfit(
                #         ensemble[(coeff_size + wind_size):, ens_num].ravel(),
                #         csi_ensemble[(coeff_size + wind_size):, ens_num].ravel(),
                #         coeff_size - 1)
                trace_analy = np.concatenate(
                    [trace_analy,  np.array([csi_ensemble.var(axis=1).mean()])])
                ensemble[:(coeff_size + wind_size)] = csi_ensemble[
                    :(coeff_size + wind_size)]
                analysis = np.concatenate(
                    [analysis, ensemble.mean(axis=1)[None, :]], axis=0)
                advected = np.concatenate([advected, q[None, :, :]], axis=0)
                ens_test = csi_ensemble[
                    this_flat_sensor_loc_test + wind_size].mean(axis=1)
                temp = ens_test - sensor_data_test.ix[sensor_time].values
                analysis_error = np.concatenate(
                    [analysis_error, temp[None, :]], axis=0)

        # for whole image assimilation
        advected = np.concatenate([advected, q[None, :, :]], axis=0)
        q = sat.loc[sat_time_range[time_index + 1]].values.reshape(domain_shape)
        # noise = (noise - noise.min())
        # noise = noise/noise.max()
        # noise = noise.ravel()
        # ensemble[wind_size::] = (q.ravel()[:, None]*noise[:, None] +
        #                          ensemble[wind_size:, :]*(1 - noise[:, None]))
        # ensemble[wind_size::] = assimilate(
        #     ensemble=ensemble[wind_size::],
        #     observations=q.ravel(),
        #     flat_sensor_indices=None, R_inverse=1/sat_sig**2,
        #     inflation=sat_inflation,
        #     domain_shape=domain_shape,
        #     localization_length=localization_letkf,
        #     assimilation_positions=assimilation_positions,
        #     assimilation_positions_2d=assimilation_positions_2d,
        #     full_positions_2d=full_positions_2d)
        ensemble[(coeff_size + wind_size):] = q.ravel()[:, None]
        # NOT doing any assimilation right now
        csi_ensemble = ensemble[(coeff_size + wind_size):].copy()
        for ens_num in range(ens_size):
            this_coeff_preturb = np.array(
                [0, 0, 0, 0,
                 csi_ensemble[1, ens_num] - csi_ensemble[0, ens_num],
                 csi_ensemble[0, ens_num]])
            csi_ensemble[:, ens_num] = np.polyval(
                init_coeff + this_coeff_preturb, csi_ensemble[:, ens_num])
        csi_ensemble = np.concatenate(
            [ensemble[:(coeff_size + wind_size)], csi_ensemble], axis=0)
        trace_analy = np.concatenate(
            [trace_analy,  np.array([csi_ensemble.var(axis=1).mean()])])
        analysis = np.concatenate(
            [analysis, ensemble.mean(axis=1)[None, :]], axis=0)
        advected = np.concatenate([advected, q[None, :, :]], axis=0)
        ens_test = csi_ensemble[
            this_flat_sensor_loc_test + wind_size].mean(axis=1)
        temp = ens_test - sensor_data_test.ix[sensor_time].values
        analysis_error = np.concatenate(
            [analysis_error, temp[None, :]], axis=0)
        noise = noise_init.copy()
    time_range = pd.date_range(sat_time_range[0], sat_time_range[-1], freq='5min')
    return analysis, analysis_error, background, background_error, advected, time_range, trace_back, trace_analy

def main_poly_interp(
        sat, x, y, domain_shape,
        U, U_shape, V, V_shape,
        start_time, end_time, dx, dy, C_max,
        assimilation_grid_size,
        localization_letkf, sat_inflation,
        sat_sig, ens_size,
        wind_sigma, wind_size,
        client,
        sensor_data,
        sensor_loc,
        sensor_sig,
        sensor_inflation,
        CI_sigma, location,
        cloud_height, sat_azimuth,
        sat_elevation):
    sat_lats, sat_lons = prep.lcc_to_sphere(x, y)
    sat_time_range = (pd.date_range(start_time, end_time, freq='15 min')
                      .tz_localize('MST'))
    sat_time_range = sat_time_range.intersection(sat.index)
    assimilation_positions, assimilation_positions_2d, full_positions_2d = (
        assimilation_position_generator(domain_shape, assimilation_grid_size))
    q = sat.loc[sat_time_range[0]].values.reshape(domain_shape)
    sensor_loc = sensor_loc.sort_values(by='id', inplace=False)
    sensor_loc_test = sensor_loc[sensor_loc.test==True]
    sensor_loc_assim = sensor_loc[sensor_loc.test==False]
    sensor_data_test = sensor_data[sensor_loc_test.id.values]
    sensor_data_assim = sensor_data[sensor_loc_assim.id.values]
    flat_sensor_loc_test = find_flat_loc(
        sat_lats, sat_lons, sensor_loc_test)
    flat_sensor_loc_assim = find_flat_loc(
        sat_lats, sat_lons, sensor_loc_assim)
    ensemble = ensemble_creator(
        q, CI_sigma=CI_sigma, wind_size=wind_size,
        wind_sigma=wind_sigma, ens_size=ens_size)
    init_coeff = np.array([2.36, -6.2, 6.22, -2.63, -0.58, 1])
    coeff_size = 6
    ensemble = np.concatenate(
        [np.repeat(init_coeff[:, None], ens_size, axis=1),
        ensemble], axis=0)
    noise_init = noise_fun(domain_shape)
    noise = noise_init.copy()
    background = ensemble.mean(axis=1)[None, :]
    analysis = background.copy()
    advected = q[None, :, :].copy()
    background_error = np.zeros([1, sensor_loc_test.id.size])
    OI_error = background_error.copy()
    analysis_error = background_error.copy()
    for time_index in range(sat_time_range.size - 1):
        sat_time = sat_time_range[time_index]
        print('sat_time: ' + str(sat_time))
        # *** manually adjust wind by .5***
        int_index_wind = U.index.get_loc('2014-04-15 13:45:00', method='pad')
        this_U = U.iloc[int_index_wind].values.reshape(U_shape) + .5
        this_V = V.iloc[int_index_wind].values.reshape(V_shape)
        cx = abs(U.values).max()
        cy = abs(V.values).max()
        T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)/C_max))
        dt = (5*60)/T_steps
        advection_number = int((
            (sat_time_range[time_index + 1].value) -
            (sat_time_range[time_index].value))*(10**(-9)/(60*5)))
        for n in range(advection_number):
            sensor_time = sat_time + pd.Timedelta('5 min')*n
            print('sensor_time: ' + str(sensor_time))
            q, noise, ensemble = advect_5min_distributed_coeff(
                q, noise, ensemble, dt, this_U, dx,
                this_V, dy, T_steps, coeff_size, wind_size, client)
            flat_correct = get_flat_correct(
                cloud_height=cloud_height, dx=dx, dy=dy,
                domain_shape=domain_shape, sat_azimuth=sat_azimuth,
                sat_elevation=sat_elevation,
                location=location, sensor_time=sat_time)
            this_flat_sensor_loc_test = flat_sensor_loc_test + flat_correct
            ens_test = ensemble[
                this_flat_sensor_loc_test + wind_size + coeff_size].mean(axis=1)
            temp = ens_test - sensor_data_test.ix[sensor_time].values
            background_error = np.concatenate(
                [background_error, temp[None, :]], axis=0)
            background = np.concatenate(
                [background, ensemble.mean(axis=1)[None,:]], axis=0)
            this_flat_sensor_loc_assim = flat_sensor_loc_assim + flat_correct
            if n != advection_number - 1:
                csi_ensemble = ensemble[(coeff_size + wind_size):].copy()
                for ens_num in range(ens_size):
                    csi_ensemble[:, ens_num] = np.polyval(
                        ensemble[:coeff_size, ens_num], csi_ensemble[:, ens_num])
                csi_ensemble = np.concatenate(
                    [ensemble[:(coeff_size + wind_size)], csi_ensemble], axis=0)
                ####
                plt.figure()
                im = plt.pcolormesh(
                    csi_ensemble[(coeff_size + wind_size):, 0].reshape(domain_shape),
                    cmap='Blues')
                plt.colorbar(im)
                plt.title('background: ')
                plt.axis('equal')
                ####
                csi_ensemble = assimilate(csi_ensemble, sensor_data_assim.ix[sensor_time],
                                  this_flat_sensor_loc_assim + wind_size,
                                  1/sensor_sig**2, inflation=sensor_inflation)
                ####
                plt.figure()
                im = plt.pcolormesh(
                    csi_ensemble[(coeff_size + wind_size):, 0].reshape(domain_shape),
                    cmap='Blues')
                plt.colorbar(im)
                plt.title('Analysis: ')
                plt.axis('equal')
                ####


                for ens_num in range(ens_size):
                    ensemble[:coeff_size, ens_num] = np.polyfit(
                        ensemble[(coeff_size + wind_size):, ens_num].ravel(),
                        csi_ensemble[(coeff_size + wind_size):, ens_num].ravel(),
                        coeff_size - 1)
                #coeff_size + wind        ensemble[coeff_size:wind_size, ens_num] = csi_ensemble[
                        #coeff_size:wind_size, ens_num]
                analysis = np.concatenate(
                    [analysis, ensemble.mean(axis=1)[None, :]], axis=0)
                advected = np.concatenate([advected, q[None, :, :]], axis=0)
                ens_test = ensemble[
                    this_flat_sensor_loc_test + wind_size].mean(axis=1)
                temp = ens_test - sensor_data_test.ix[sensor_time].values
                analysis_error = np.concatenate(
                    [analysis_error, temp[None, :]], axis=0)

        # for whole image assimilation
        advected = np.concatenate([advected, q[None, :, :]], axis=0)
        q = sat.loc[sat_time_range[time_index + 1]].values.reshape(domain_shape)
        # noise = (noise - noise.min())
        # noise = noise/noise.max()
        # noise = noise.ravel()
        # ensemble[wind_size::] = (q.ravel()[:, None]*noise[:, None] +
        #                          ensemble[wind_size:, :]*(1 - noise[:, None]))
        # ensemble[wind_size::] = assimilate(
        #     ensemble=ensemble[wind_size::],
        #     observations=q.ravel(),
        #     flat_sensor_indices=None, R_inverse=1/sat_sig**2,
        #     inflation=sat_inflation,
        #     domain_shape=domain_shape,
        #     localization_length=localization_letkf,
        #     assimilation_positions=assimilation_positions,
        #     assimilation_positions_2d=assimilation_positions_2d,
        #     full_positions_2d=full_positions_2d)
        ensemble[(coeff_size + wind_size):] = q.ravel()[:, None]
        analysis = np.concatenate(
            [analysis, ensemble.mean(axis=1)[None, :]], axis=0)
        noise = noise_init.copy()
    time_range = pd.date_range(sat_time_range[0], sat_time_range[-1], freq='5min')
    return analysis, analysis_error, background, background_error, advected, time_range



# Only ends at satellite times
def main_old(sat, wind, sensor_data, sensor_loc, start_time,
         end_time, dx, dy, C_max, assimilation_grid_size,
         localization_length, sensor_inflation, sat_inflation,
         sat_sig, sensor_sig, ens_size,
         wind_sigma, wind_size, CI_sigma, location, cloud_height,
         sat_azimuth, sat_elevation, client):
    """Check back later."""
    ## NEED: Incorporate IO? Would need to reformulate so that P is smaller.
    time_range = (pd.date_range(start_time, end_time, freq='15 min')
                  .tz_localize('MST').astype(int))
    all_time = sat.time.values
    time_range = np.intersect1d(time_range, all_time)
    domain_shape = sat['clear_sky_good'].isel(time=0).shape
    noise_init = noise_fun(domain_shape)
    assimilation_positions, assimilation_positions_2d, full_positions_2d = (
        assimilation_position_generator(domain_shape, assimilation_grid_size))
    ## This is only for now. Eventually H will be a function of time and cloud height.
    # H, delete = forward_obs_mat(sensor_loc[['lat', 'lon']].values, sat_loc)
    # H = np.concatenate((np.zeros((H.shape[0], 2)), H), axis=1)
    sensor_loc = sensor_loc.sort_values(by='id', inplace=False)
    sensor_loc_test = sensor_loc[sensor_loc.test==True]
    sensor_loc_assim = sensor_loc[sensor_loc.test==False]
    sensor_data_test = sensor_data[sensor_loc_test.id.values]
    sensor_data_assim = sensor_data[sensor_loc_assim.id.values]
    flat_sensor_loc_test, lat_step, lon_step = find_flat_loc(
        sat, sensor_loc_test)
    flat_sensor_loc_assim, lat_step, lon_step = find_flat_loc(
        sat, sensor_loc_assim)
    ensemble = ensemble_creator(
        sat['clear_sky_good'].sel(time=time_range[0]).values,
        CI_sigma=CI_sigma, wind_size=wind_size,
        wind_sigma=wind_sigma, ens_size=ens_size)
    q = sat['clear_sky_good'].sel(time=time_range[0]).values
    noise = noise_init.copy()
    background = ensemble.mean(axis=1)[None, :]
    analysis = background.copy()
    advected = q[None, :, :].copy()
    background_error = np.zeros([1, sensor_loc_test.id.size])
    analysis_error = background_error.copy()
    for time_index in range(time_range.size - 1):
        sat_time = time_range[time_index]
        print('time_index: ' + str(time_index))
        # *** manually adjust wind by .5***
        U = wind.sel(time=sat_time, method='pad').U.values + .5
        V = wind.sel(time=sat_time, method='pad').V.values
        cx = abs(U).max()
        cy = abs(V).max()
        T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)/C_max))
        dt = (5*60)/T_steps
        advection_number = int((time_range[time_index + 1] -
                                time_range[time_index])*(10**(-9)/(60*5)))
        for n in range(advection_number):
            sensor_time = pd.Timestamp(
                sat_time + (n + 1)*5*60*10**9).tz_localize('UTC'
                ).tz_convert('MST')
            print('advection_number: ' + str(n))
            q, noise, ensemble = advect_5min_distributed(
                q, noise, ensemble, dt, U, dx,
                V, dy, T_steps, wind_size, client)
            background = np.concatenate(
                [background, ensemble.mean(axis=1)[None,:]], axis=0)
            flat_correct = get_flat_correct(
                cloud_height=cloud_height, lat_step=lat_step, lon_step=lon_step,
                domain_shape=domain_shape, sat_azimuth=sat_azimuth,
                sat_elevation=sat_elevation,
                location=location, sensor_time=sensor_time)
            this_flat_sensor_loc_test = flat_sensor_loc_test + flat_correct
            ens_test = ensemble[
                this_flat_sensor_loc_test + wind_size].mean(axis=1)
            temp = ens_test - sensor_data_test.ix[sensor_time].values
            background_error = np.concatenate(
                [background_error, temp[None, :]], axis=0)
            this_flat_sensor_loc_assim = flat_sensor_loc_assim + flat_correct
            ensemble = assimilate(ensemble, sensor_data_assim.ix[sensor_time],
                                  this_flat_sensor_loc_assim + wind_size,
                                  1/sensor_sig**2, inflation=sensor_inflation)
            if n != advection_number-1:
                analysis = np.concatenate(
                    [analysis, ensemble.mean(axis=1)[None, :]], axis=0)
                advected = np.concatenate([advected, q[None, :, :]], axis=0)
                ens_test = ensemble[
                    this_flat_sensor_loc_test + wind_size].mean(axis=1)
                temp = ens_test - sensor_data_test.ix[sensor_time].values
                analysis_error = np.concatenate(
                    [analysis_error, temp[None, :]], axis=0)


        # for whole image assimilation
        q = sat['clear_sky_good'].sel(time=time_range[time_index + 1]).values
        advected = np.concatenate([advected, q[None, :, :]], axis=0)
        noise = (noise - noise.min())
        noise = noise/noise.max()
        noise = noise.ravel()
        ensemble[wind_size::] = (q.ravel()[:, None]*noise[:, None] +
                                 ensemble[wind_size:, :]*(1 - noise[:, None]))
        ensemble[wind_size::] = assimilate(
            ensemble=ensemble[wind_size::],
            observations=sat['clear_sky_good'].sel(
                time=time_range[time_index + 1]).values.ravel(),
            flat_sensor_indices=None, R_inverse=1/sat_sig**2,
            inflation=sat_inflation,
            domain_shape=domain_shape,
            localization_length=localization_length,
            assimilation_positions=assimilation_positions,
            assimilation_positions_2d=assimilation_positions_2d,
            full_positions_2d=full_positions_2d)
        analysis = np.concatenate(
            [analysis, ensemble.mean(axis=1)[None, :]], axis=0)
        ens_test = ensemble[
            this_flat_sensor_loc_test + wind_size].mean(axis=1)
        temp = ens_test - sensor_data_test.ix[sensor_time].values
        analysis_error = np.concatenate(
            [analysis_error, temp[None, :]], axis=0)
        noise = noise_init.copy()
    begining = time_range[0]
    end = time_range[-1]
    time_range = (pd.date_range(begining, end, freq='5 min')
                  .tz_localize('UTC').tz_convert('MST'))
    return analysis, analysis_error, background, background_error, advected, time_range


def main_oi(sat, wind, sensor_data, sensor_loc, start_time,
            end_time, dx, dy, C_max, assimilation_grid_size,
            localization_length, sensor_inflation, sat_inflation,
            sat_sig, sensor_sig, ens_size,
            wind_sigma, wind_size, CI_sigma, CI_localization, sat_oi_inflation,
            location, cloud_height,
            sat_azimuth, sat_elevation, client):
    """Check back later."""
    ## NEED: Incorporate IO? Would need to reformulate so that P is smaller.
    time_range = (pd.date_range(start_time, end_time, freq='15 min')
                  .tz_localize('MST').astype(int))
    all_time = sat.time.values
    time_range = np.intersect1d(time_range, all_time)
    domain_shape = sat['clear_sky_good'].isel(time=0).shape
    noise_init = noise_fun(domain_shape)
    assimilation_positions, assimilation_positions_2d, full_positions_2d = (
        assimilation_position_generator(domain_shape, assimilation_grid_size))
    ## This is only for now. Eventually H will be a function of time and cloud height.
    # H, delete = forward_obs_mat(sensor_loc[['lat', 'lon']].values, sat_loc)
    # H = np.concatenate((np.zeros((H.shape[0], 2)), H), axis=1)
    sensor_loc = sensor_loc.sort_values(by='id', inplace=False)
    sensor_loc_test = sensor_loc[sensor_loc.test==True]
    sensor_loc_assim = sensor_loc[sensor_loc.test==False]
    sensor_data_test = sensor_data[sensor_loc_test.id.values]
    sensor_data_assim = sensor_data[sensor_loc_assim.id.values]
    flat_sensor_loc_test, lat_step, lon_step = find_flat_loc(
        sat, sensor_loc_test)
    flat_sensor_loc_assim, lat_step, lon_step = find_flat_loc(
        sat, sensor_loc_assim)
    ensemble = ensemble_creator(
        sat['clear_sky_good'].sel(time=time_range[0]).values,
        CI_sigma=CI_sigma, wind_size=wind_size,
        wind_sigma=wind_sigma, ens_size=ens_size)
    q = sat['clear_sky_good'].sel(time=time_range[0]).values
    noise = noise_init.copy()
    background = ensemble.mean(axis=1)[None, :]
    analysis = background.copy()
    advected = q[None, :, :].copy()
    background_error = np.zeros([1, sensor_loc_test.id.size])
    OI_error = background_error.copy()
    analysis_error = background_error.copy()
    for time_index in range(time_range.size - 1):
        sat_time = time_range[time_index]
        print('time_index: ' + str(time_index))
        U = wind.sel(time=sat_time, method='pad').U.values
        V = wind.sel(time=sat_time, method='pad').V.values
        cx = abs(U).max()
        cy = abs(V).max()
        T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)/C_max))
        dt = (5*60)/T_steps
        advection_number = int((time_range[time_index + 1] -
                                time_range[time_index])*(10**(-9)/(60*5)))
        for n in range(advection_number):
            sensor_time = pd.Timestamp(
                sat_time + (n + 1)*5*60*10**9).tz_localize('UTC'
                ).tz_convert('MST')
            print('advection_number: ' + str(n))
            q, noise, ensemble = advect_5min_distributed(
                q, noise, ensemble, dt, U, dx,
                V, dy, T_steps, wind_size, client)
            background = np.concatenate(
                [background, ensemble.mean(axis=1)[None,:]], axis=0)
            flat_correct = get_flat_correct(
                cloud_height=cloud_height, lat_step=lat_step, lon_step=lon_step,
                domain_shape=domain_shape, sat_azimuth=sat_azimuth,
                sat_elevation=sat_elevation,
                location=location, sensor_time=sensor_time)
            this_flat_sensor_loc_test = flat_sensor_loc_test + flat_correct
            ens_test = ensemble[
                this_flat_sensor_loc_test + wind_size].mean(axis=1)
            temp = ens_test - sensor_data_test.ix[sensor_time].values
            background_error = np.concatenate(
                [background_error, temp[None, :]], axis=0)
            this_flat_sensor_loc_assim = flat_sensor_loc_assim + flat_correct
            if n != advection_number-1:
                ensemble = assimilate(ensemble,
                                      sensor_data_assim.ix[sensor_time],
                                      this_flat_sensor_loc_assim + wind_size,
                                      1/sensor_sig**2,
                                      inflation=sensor_inflation)
                analysis = np.concatenate(
                    [analysis, ensemble.mean(axis=1)[None, :]], axis=0)
                advected = np.concatenate([advected, q[None, :, :]], axis=0)
                ens_test = ensemble[
                    this_flat_sensor_loc_test + wind_size].mean(axis=1)
                temp = ens_test - sensor_data_test.ix[sensor_time].values
                analysis_error = np.concatenate(
                    [analysis_error, temp[None, :]], axis=0)
                OI_error = np.concatenate(
                    [OI_error, np.zeros_like(temp[None, :])], axis=0)


        # for whole image assimilation

        q = sat['clear_sky_good'].sel(time=time_range[time_index + 1]).values

        # ####
        # plt.figure()
        # im = plt.pcolormesh(q)
        # plt.colorbar(im)
        # ####

        q = optimal_interpolation(
            sat['clear_sky_good'].sel(
                time=time_range[time_index + 1]).values.ravel(), sat_sig,
            sensor_data_assim.ix[sensor_time].values, sensor_sig,
            sat['clear_sky_good'].sel(time=time_range[time_index + 1]).values.ravel(),
            this_flat_sensor_loc_assim, CI_localization, sat_oi_inflation).reshape(domain_shape)
        temp = q.ravel()[this_flat_sensor_loc_test] - sensor_data_test.ix[sensor_time].values
        OI_error = np.concatenate([OI_error, temp[None, :]], axis=0)
        # ####
        # plt.figure()
        # im = plt.pcolormesh(q)
        # plt.colorbar(im)
        # ####


        advected = np.concatenate([advected, q[None, :, :]], axis=0)
        noise = (noise - noise.min())
        noise = noise/noise.max()
        noise = noise.ravel()
        ensemble[wind_size::] = (q.ravel()[:, None]*noise[:, None] +
                                 ensemble[wind_size:, :]*(1 - noise[:, None]))
        ensemble[wind_size::] = assimilate(
            ensemble=ensemble[wind_size::],
            observations=sat['clear_sky_good'].sel(
                time=time_range[time_index + 1]).values.ravel(),
            flat_sensor_indices=None, R_inverse=1/sat_sig**2,
            inflation=sat_inflation,
            domain_shape=domain_shape,
            localization_length=localization_length,
            assimilation_positions=assimilation_positions,
            assimilation_positions_2d=assimilation_positions_2d,
            full_positions_2d=full_positions_2d)
        analysis = np.concatenate(
            [analysis, ensemble.mean(axis=1)[None, :]], axis=0)
        ens_test = ensemble[
            this_flat_sensor_loc_test + wind_size].mean(axis=1)
        temp = ens_test - sensor_data_test.ix[sensor_time].values
        analysis_error = np.concatenate(
            [analysis_error, temp[None, :]], axis=0)
        noise = noise_init.copy()
    begining = time_range[0]
    end = time_range[-1]
    time_range = (pd.date_range(begining, end, freq='5 min')
                  .tz_localize('UTC').tz_convert('MST'))
    return analysis, analysis_error, background, background_error, advected, time_range, OI_error


def just_advection(sat, wind, u_pert, v_pert,
                   start_time, end_time, dx, dy, C_max):
    """Check back later."""
    ## NEED: Incorporate IO? Would need to reformulate so that P is smaller.
    time_range = (pd.date_range(start_time, end_time, freq='15 min')
                  .tz_localize('MST').astype(int))
    all_time = sat.time.values
    time_range = np.intersect1d(time_range, all_time)
    q = sat['clear_sky_good'].sel(time=time_range[0]).values
    q_rmse = np.zeros(time_range.size - 1)
    count=0
    for time_index in range(time_range.size - 1):
        sat_time = time_range[time_index]
        # print('time_index: ' + str(time_index))
        U = wind.sel(time=sat_time, method='pad').U.values + u_pert
        V = wind.sel(time=sat_time, method='pad').V.values + v_pert
        cx = abs(U).max()
        cy = abs(V).max()
        T_steps = int(np.ceil((5*60)*(cx/dx+cy/dy)/C_max))
        dt = (5*60)/T_steps
        advection_number = int((time_range[time_index + 1] -
                                time_range[time_index])*(10**(-9)/(60*5)))
        for n in range(advection_number):
            # print('advection_number: ' + str(n))
            for t in range(T_steps):
                q = time_deriv_3(q, dt, U, dx, V, dy)

        q_new = sat['clear_sky_good'].sel(time=time_range[time_index + 1]).values
        q_error = (q - q_new)**2
        q_rmse[count] = np.sqrt(q_error.mean())
        # print(q_rmse[count])

        # verr = np.abs(q_error).max()
        # vmin = 0
        # vmax = 1.1

        # plt.figure()
        # im = plt.pcolormesh(
        #     q,
        #     cmap='Blues', vmin=vmin, vmax=vmax)
        # plt.colorbar(im)
        # plt.title('advected:')
        # plt.axis('equal')

        # plt.figure()
        # im = plt.pcolormesh(
        #     q_new,
        #     cmap='Blues', vmin=vmin, vmax=vmax)
        # plt.colorbar(im)
        # plt.title('new sat: ')
        # plt.axis('equal')

        # plt.figure()
        # im = plt.pcolormesh(
        #     q_error,
        #     cmap='bwr', vmin=-verr, vmax=verr)
        # plt.colorbar(im)
        # plt.title('new sat: ')
        # plt.axis('equal')

        q = q_new
        count += 1
    print(q_rmse.mean())
    # return q_rmse.mean()


def test_parallax(sat, domain_shape, dx, dy, lats, lons, sensor_data,
                  sensor_loc, start_time,
                  end_time, location, cloud_height,
                  sat_azimuth, sat_elevation,
                  oi_sat_sig, oi_sensor_sig, oi_localization, oi_inflation):
    """Check back later."""
    ## NEED: Incorporate OI? Would need to reformulate so that P is smaller.
    sensor_loc = sensor_loc.sort_values(by='id', inplace=False)
    time_range = (pd.date_range(start_time, end_time, freq='15 min')
                  .tz_localize('MST'))
    all_time = sat.index
    time_range = np.intersect1d(time_range, all_time)

    sensor_loc_test = sensor_loc[sensor_loc.test==True]
    sensor_loc_assim = sensor_loc[sensor_loc.test==False]
    sensor_data_test = sensor_data[sensor_loc_test.id.values]
    sensor_data_assim = sensor_data[sensor_loc_assim.id.values]

    flat_sensor_loc_test = find_flat_loc(
        lats, lons, sensor_loc_test)
    flat_sensor_loc_assim = find_flat_loc(
        lats, lons, sensor_loc_assim)

    sat_error = np.ones([time_range.size, flat_sensor_loc_test.size])*np.nan
    oi_error = np.ones([time_range.size, flat_sensor_loc_test.size])*np.nan
    lat_correction = np.ones(time_range.size)*np.nan
    lon_correction = np.ones(time_range.size)*np.nan
    for time_index in range(time_range.size):
        sat_time = time_range[time_index]
        q = sat.ix[sat_time].values
        flat_correct = get_flat_correct(
                cloud_height=cloud_height, dx=dx, dy=dy,
                domain_shape=domain_shape, sat_azimuth=sat_azimuth,
                sat_elevation=sat_elevation,
                location=location, sensor_time=sat_time)
        this_flat_sensor_loc_test = flat_sensor_loc_test - flat_correct # changed to -
        sat_error[time_index] = (q[this_flat_sensor_loc_test] -
                                 sensor_data_test.ix[sat_time].values)

        this_flat_sensor_loc_assim = flat_sensor_loc_assim - flat_correct #changed to -
        this_OI = optimal_interpolation(
            q.ravel(), oi_sat_sig, sensor_data_assim.ix[sat_time], oi_sensor_sig,
            q.ravel(), this_flat_sensor_loc_assim,
            oi_localization, oi_inflation)
        oi_error[time_index] = (this_OI[this_flat_sensor_loc_test] -
                                sensor_data_test.ix[sat_time].values)

        solar_position = location.get_solarposition(sat_time)
        x_correct, y_correct = parallax_shift(
            cloud_height, sat_azimuth, sat_elevation,
            solar_position['azimuth'].values,
            solar_position['elevation'].values)
        lat_correct, lon_correct = to_lat_lon(x_correct, y_correct,
                                              location.latitude)
        lat_correction[time_index] = lat_correct
        lon_correction[time_index] = lon_correct

        # for whole image assimilation
    return oi_error, sat_error, lat_correction, lon_correction, time_range
