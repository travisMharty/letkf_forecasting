import logging
import numpy as np
from distributed import Client
import pandas as pd
import scipy as sp
import scipy.interpolate as interpolate
from skimage import filters as ski_filters
import fenics as fe
from netCDF4 import Dataset, num2date

import letkf_forecasting.random_functions as rf
import letkf_forecasting.letkf_io as letkf_io
from letkf_forecasting.optical_flow import optical_flow

# average radius of earth when modeled as a sphere From Wikipedia
a = 6371000


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
        satellite_displacement*np.cos(-np.pi/2 -
                                      satellite_azimuth*2*np.pi/360))
    y_correction = (
        solar_displacement*np.sin(-np.pi/2 - solar_azimuth*2*np.pi/360) -
        satellite_displacement*np.sin(-np.pi/2 -
                                      satellite_azimuth*2*np.pi/360))

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
    https://stackoverflow.com/questions/2566412/
    find-nearest-value-in-numpy-array"""
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


def nearest_positions(loc, shape, dist, stag=None):
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
    if stag is None:
        row_min = (position[0] - dist).clip(min=0)
        row_max = (position[0] + dist).clip(max=(shape[0] - 1))
        col_min = (position[1] - dist).clip(min=0)
        col_max = (position[1] + dist).clip(max=(shape[1] - 1))

    else:
        row_min = (position[0] - dist + stag[0]).clip(min=0)
        row_max = (position[0] + dist + stag[0]).clip(max=(shape[0] - 1))
        col_min = (position[1] - dist + stag[1]).clip(min=0)
        col_max = (position[1] + dist + stag[1]).clip(max=(shape[1] - 1))
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
    K = sp.linalg.solve(
        (PHT[flat_locations, :] + o_sig**2*np.eye(flat_locations.size)),
        PHT.T).T
    to_return = background + K.dot(observations - background[flat_locations])

    return to_return


def reduced_enkf(ensemble,
                 observations, R_sig,
                 flat_locations, inflation,
                 localization=0, x=None, y=None):
    ens_num = ensemble.shape[1]
    obs_size = observations.size
    PHT = ensemble.dot(ensemble[flat_locations].T)
    if localization < 10**50:
        rhoHT = ((x[:, None] - x[None, flat_locations])**2 +
                 (y[:, None] - y[None, flat_locations])**2)
        rhoHT = np.exp(-rhoHT/(2*localization**2))
        PHT = PHT*rhoHT
    K = sp.linalg.solve(
        (PHT[flat_locations, :] + R_sig**2*np.eye(flat_locations.size)),
        PHT.T).T
    rand_obs = np.random.normal(
        loc=0.0, scale=R_sig, size=ens_num*obs_size)
    rand_obs = rand_obs.reshape(obs_size, ens_num)
    rand_obs += observations[:, None]
    analysis = ensemble + K.dot(rand_obs - ensemble[flat_locations])

    return analysis


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
    # Change to allow for R to not be pre-inverted?
    if localization_length is False:

        # LETKF without localization
        Y_b = ensemble[flat_sensor_indices, :]
        y_b_bar = Y_b.mean(axis=1)
        Y_b -= y_b_bar[:, None]
        x_bar = ensemble.mean(axis=1)
        ensemble -= x_bar[:, None]
        ens_size = ensemble.shape[1]
        # C = (Y_b.T).dot(R_inverse)
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
        w_a_bar = P_tilde.dot(C.dot(observations - y_b_bar))
        W_a += w_a_bar[:, None]
        ensemble = x_bar[:, None] + ensemble.dot(W_a)
        return ensemble

    else:
        # LETKF with localization assumes H is I
        # NEED: to include wind in ensemble will require reworking due to
        # new H and different localization.
        # NEED: Change to include some form of H for paralax correction??
        # Maybe: ^ not if paralax is only corrected when moving ...
        # to ground sensors.
        # SHOULD: Will currently write as though R_inverse is a scalar.
        # May need to change at some point but will likely need to do
        # something clever since R_inverse.size is 400 billion
        # best option: form R_inverse inside of localization routine
        # good option: assimilate sat images at low resolution ...
        # (probabily should do this either way)
        x_bar = ensemble.mean(axis=1)  # Need to bring this back
        ensemble -= x_bar[:, None]
        ens_size = ensemble.shape[1]
        kal_count = 0
        W_interp = np.zeros([assimilation_positions.size, ens_size**2])
        for interp_position in assimilation_positions:
            local_positions = nearest_positions(interp_position, domain_shape,
                                                localization_length)
            local_ensemble = ensemble[local_positions]
            local_x_bar = x_bar[local_positions]
            local_obs = observations[local_positions]  # assume H is I
            # assume R_inverse is diag*const
            C = (local_ensemble.T)*R_inverse

            # This should be better, but I can't get it to work
            eig_value, eig_vector = np.linalg.eigh(
                (ens_size-1)*np.eye(ens_size)/inflation +
                C.dot(local_ensemble))
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
            W_interp[kal_count] = np.ravel(W_a)  # separate w_bar??
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


def assimilate_full_wind(ensemble, observations, flat_sensor_indices,
                         R_inverse, R_inverse_wind, inflation, wind_inflation,
                         domain_shape=False, U_shape=False, V_shape=False,
                         localization_length=False,
                         localization_length_wind=False,
                         assimilation_positions=False,
                         assimilation_positions_2d=False,
                         full_positions_2d=False):
    # seperate out localization in irradiance and wind
    """
    *** NEED TO REWRITE Documentation
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

    # LETKF with localization assumes H is I
    # NEED: to include wind in ensemble will require reworking due to
    # new H and different localization.
    # NEED: Change to include some form of H for paralax correction??
    # Maybe: ^ not if paralax is only corrected when moving to ground sensors.
    # SHOULD: Will currently write as though R_inverse is a scalar.
    # May need to change at some point but will likely need to do
    # something clever since R_inverse.size is 400 billion
    # best option: form R_inverse inside of localization routine
    # good option: assimilate sat images at low resolution (probabily should...
    # do this either way)

    U_size = U_shape[0]*U_shape[1]
    V_size = V_shape[0]*V_shape[1]
    ensemble_U = ensemble[:U_size]
    x_bar_U = ensemble_U.mean(axis=1)
    ensemble_U -= x_bar_U[:, None]
    ensemble_V = ensemble[U_size:V_size + U_size]
    x_bar_V = ensemble_V.mean(axis=1)
    ensemble_V -= x_bar_V[:, None]
    ensemble_csi = ensemble[U_size + V_size:]
    x_bar_csi = ensemble_csi.mean(axis=1)
    ensemble_csi -= x_bar_csi[:, None]
    ens_size = ensemble.shape[1]
    kal_count = 0
    W_interp = np.zeros([assimilation_positions.size, ens_size**2])
    W_interp_wind = W_interp.copy()*np.nan
    # bad_count = 0
    for interp_position in assimilation_positions:
        # for the irradiance portion of the ensemble
        local_positions = nearest_positions(interp_position, domain_shape,
                                            localization_length)
        local_ensemble = ensemble_csi[local_positions]
        local_x_bar = x_bar_csi[local_positions]
        local_obs = observations[local_positions]  # assume H is I
        C = (local_ensemble.T)*R_inverse  # assume R_inverse is diag+const

        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/inflation + C.dot(local_ensemble))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        W_a += w_a_bar[:, None]
        W_interp[kal_count] = np.ravel(W_a)  # separate w_bar??

        # should eventually change to assimilate on coarser wind grid
        local_positions = nearest_positions(interp_position, domain_shape,
                                            localization_length_wind)
        local_ensemble = ensemble_csi[local_positions]
        local_x_bar = x_bar_csi[local_positions]
        local_obs = observations[local_positions]  # assume H is I
        C = (local_ensemble.T)*R_inverse_wind  # assume R_inverse is diag+const
        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/wind_inflation +
            C.dot(local_ensemble))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        W_a += w_a_bar[:, None]
        W_interp_wind[kal_count] = np.ravel(W_a)  # separate w_bar??
        kal_count += 1
    if assimilation_positions_2d.size != full_positions_2d.size:
        W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                 W_interp)
        W_interp = W_fun(full_positions_2d)

        W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                 W_interp_wind)
        W_interp_wind = W_fun(full_positions_2d)

    W_fine_mesh = W_interp.reshape(domain_shape[0]*domain_shape[1],
                                   ens_size, ens_size)
    # change this to its own variable
    W_interp = W_interp_wind.reshape(domain_shape[0], domain_shape[1],
                                     ens_size, ens_size)
    ensemble_csi = x_bar_csi[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_csi, W_fine_mesh)
    W_fine_mesh = (np.pad(W_interp, [(0, 0), (0, 1), (0, 0), (0, 0)],
                          mode='edge')).reshape(U_size, ens_size, ens_size)
    ensemble_U = x_bar_U[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_U, W_fine_mesh)
    W_fine_mesh = (np.pad(W_interp, [(0, 1), (0, 0), (0, 0), (0, 0)],
                          mode='edge')).reshape(V_size, ens_size, ens_size)
    ensemble_V = x_bar_V[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_V, W_fine_mesh)
    ensemble = np.concatenate([ensemble_U, ensemble_V, ensemble_csi], axis=0)

    return ensemble


def assimilate_sat_to_wind(ensemble, observations,
                           R_inverse_wind, wind_inflation,
                           domain_shape=False, U_shape=False, V_shape=False,
                           localization_length_wind=False,
                           assimilation_positions=False,
                           assimilation_positions_2d=False,
                           full_positions_2d=False):
    # separate out localization in irradiance and wind
    """
    *** NEED TO REWRITE Documentation
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

    # LETKF with localization assumes H is I
    # NEED: to include wind in ensemble will require reworking due to
    # new H and different localization.
    # NEED: Change to include some form of H for paralax correction??
    # Maybe: ^ not if paralax is only corrected when moving to ground sensors.
    # SHOULD: Will currently write as though R_inverse is a scalar.
    # May need to change at some point but will likely need to do
    # something clever since R_inverse.size is 400 billion
    # best option: form R_inverse inside of localization routine
    # good option: assimilate sat images at low resolution ...
    # (probabily should do this either way)

    U_size = U_shape[0]*U_shape[1]
    V_size = V_shape[0]*V_shape[1]
    ensemble_U = ensemble[:U_size]
    x_bar_U = ensemble_U.mean(axis=1)
    ensemble_U -= x_bar_U[:, None]
    ensemble_V = ensemble[U_size:V_size + U_size]
    x_bar_V = ensemble_V.mean(axis=1)
    ensemble_V -= x_bar_V[:, None]
    ensemble_csi = ensemble[U_size + V_size:]
    x_bar_csi = ensemble_csi.mean(axis=1)
    ensemble_csi -= x_bar_csi[:, None]
    ens_size = ensemble.shape[1]
    kal_count = 0
    W_interp = np.zeros([assimilation_positions.size, ens_size**2])
    W_interp_wind = W_interp.copy()*np.nan
    # bad_count = 0
    for interp_position in assimilation_positions:
        # # for the irradiance portion of the ensemble
        # local_positions = nearest_positions(interp_position, domain_shape,
        #                                     localization_length)
        # local_ensemble = ensemble_csi[local_positions]
        # local_x_bar = x_bar_csi[local_positions]
        # local_obs = observations[local_positions] # assume H is I
        # C = (local_ensemble.T)*R_inverse  # assume R_inverse is diag+const

        # eig_value, eig_vector = np.linalg.eigh(
        #     (ens_size-1)*np.eye(ens_size)/inflation + C.dot(local_ensemble))
        # P_tilde = eig_vector.copy()
        # W_a = eig_vector.copy()
        # for i, num in enumerate(eig_value):
        #     P_tilde[:, i] *= 1/num
        #     W_a[:, i] *= 1/np.sqrt(num)
        # P_tilde = P_tilde.dot(eig_vector.T)
        # W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        # w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        # W_a += w_a_bar[:, None]
        # W_interp[kal_count] = np.ravel(W_a) # separate w_bar??

        # should eventually change to assimilate on coarser wind grid
        local_positions = nearest_positions(interp_position, domain_shape,
                                            localization_length_wind)
        local_ensemble = ensemble_csi[local_positions]
        local_x_bar = x_bar_csi[local_positions]
        local_obs = observations[local_positions]  # assume H is I
        C = (local_ensemble.T)*R_inverse_wind  # assume R_inverse is diag+const
        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/wind_inflation +
            C.dot(local_ensemble))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        W_a += w_a_bar[:, None]
        W_interp_wind[kal_count] = np.ravel(W_a)  # separate w_bar??
        kal_count += 1
    if assimilation_positions_2d.size != full_positions_2d.size:
        W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                 W_interp_wind)
        W_interp_wind = W_fun(full_positions_2d)
    # change this to its own variable
    W_interp = W_interp_wind.reshape(domain_shape[0], domain_shape[1],
                                     ens_size, ens_size)
    W_fine_mesh = (np.pad(W_interp, [(0, 0), (0, 1), (0, 0), (0, 0)],
                          mode='edge')).reshape(U_size, ens_size, ens_size)
    ensemble_U = x_bar_U[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_U, W_fine_mesh)
    W_fine_mesh = (np.pad(W_interp, [(0, 1), (0, 0), (0, 0), (0, 0)],
                          mode='edge')).reshape(V_size, ens_size, ens_size)
    ensemble_V = x_bar_V[:, None] + np.einsum(
        'ij, ijk->ik', ensemble_V, W_fine_mesh)
    # leave csi unchanged
    ensemble_csi += x_bar_csi[:, None]

    ensemble = np.concatenate([ensemble_U, ensemble_V, ensemble_csi], axis=0)

    return ensemble


def assimilate_wrf(ensemble, observations,
                   R_inverse, wind_inflation,
                   wind_shape,
                   localization_length_wind=False,
                   assimilation_positions=False,
                   assimilation_positions_2d=False,
                   full_positions_2d=False):
    # Currently doing U and V separately
    """
    *** NEED TO REWRITE Documentation
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

    # LETKF with localization assumes H is I
    # NEED: to include wind in ensemble will require reworking due to
    # new H and different localization.
    # NEED: Change to include some form of H for paralax correction??
    # Maybe: ^ not if paralax is only corrected when moving to ground sensors.
    # SHOULD: Will currently write as though R_inverse is a scalar.
    # May need to change at some point but will likely need to do
    # something clever since R_inverse.size is 400 billion
    # best option: form R_inverse inside of localization routine
    # good option: assimilate sat images at low resolution...
    # (probabily should do this either way)

    x_bar = ensemble.mean(axis=1)
    ensemble -= x_bar[:, None]
    ens_size = ensemble.shape[1]
    kal_count = 0
    W_interp = np.zeros([assimilation_positions.size, ens_size**2])
    for interp_position in assimilation_positions:
        # for the irradiance portion of the ensemble
        local_positions = nearest_positions(interp_position, wind_shape,
                                            localization_length_wind)
        local_ensemble = ensemble[local_positions]
        local_x_bar = x_bar[local_positions]
        local_obs = observations[local_positions]  # assume H is I
        C = (local_ensemble.T)*R_inverse  # assume R_inverse is diag+const
        eig_value, eig_vector = np.linalg.eigh(
            (ens_size-1)*np.eye(ens_size)/wind_inflation +
            C.dot(local_ensemble))
        P_tilde = eig_vector.copy()
        W_a = eig_vector.copy()
        for i, num in enumerate(eig_value):
            P_tilde[:, i] *= 1/num
            W_a[:, i] *= 1/np.sqrt(num)
        P_tilde = P_tilde.dot(eig_vector.T)
        W_a = W_a.dot(eig_vector.T)*np.sqrt(ens_size - 1)
        w_a_bar = P_tilde.dot(C.dot(local_obs - local_x_bar))
        W_a += w_a_bar[:, None]
        W_interp[kal_count] = np.ravel(W_a)  # separate w_bar??
        kal_count += 1
    if assimilation_positions_2d.size != full_positions_2d.size:
        W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                 W_interp)
        W_interp = W_fun(full_positions_2d)

    W_fine_mesh = W_interp.reshape(wind_shape[0]*wind_shape[1],
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
    x_bar = ensemble.mean(axis=1)  # Need to bring this back
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
    w_a_bar = P_tilde.dot(C.dot(observations - y_b_bar))
    W_a += w_a_bar[:, None]
    ensemble = x_bar[:, None] + ensemble.dot(W_a)
    return ensemble[:wind_size]


def ensemble_creator(sat_image, u, v, CI_sigma, wind_sigma, ens_size):
    """need to change later"""
    random_nums = np.random.normal(
        loc=0,
        scale=wind_sigma[0],
        size=ens_size)
    ensemble = u.ravel()[:, None] + random_nums[None, :]
    random_nums = np.random.normal(
        loc=0,
        scale=wind_sigma[1],
        size=ens_size)
    ensemble = np.concatenate(
        [ensemble,
         v.ravel()[:, None] + random_nums[None, :]],
        axis=0)
    ensemble = np.concatenate(
        [ensemble,
         np.repeat(sat_image.ravel()[:, None], ens_size, axis=1)],
        axis=0)
    wind_size = u.size + v.size

    csi_min_pert = np.random.normal(loc=0, scale=CI_sigma, size=ens_size)
    csi_max_pert = np.random.normal(loc=1, scale=CI_sigma*.2, size=ens_size)
    ensemble[wind_size:] = (
        (csi_max_pert[None, :] - csi_min_pert[None, :])
        * ensemble[wind_size:] + csi_min_pert[None, :])
    # CI_pert = np.random.normal(loc=0, scale=CI_sigma, size=ens_size)
    # ensemble[wind_size:] = ((1 - CI_pert[None, :])*ensemble[wind_size:] +
    #                         CI_pert[None, :])
    # ensemble[wind_size:] = ensemble[wind_size:] + CI_pert[None, :]
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


def find_flat_loc(sat_lat, sat_lon, sensor_loc):
    sat_lat = sat_lat[:, None]
    sat_lon = sat_lon[:, None]
    sensor_lat = sensor_loc['lat'].values[None, :]
    sensor_lon = sensor_loc['lon'].values[None, :]
    distance = np.sqrt((sat_lat - sensor_lat)**2 + (sat_lon - sensor_lon)**2)
    return np.argmin(distance, axis=0)


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


def perturb_irradiance(ensemble, domain_shape, edge_weight, pert_mean,
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


def logistic(array, L, k, x0):
    return L/(1 + np.exp(-k*(array - x0)))


def perturb_irradiance_new(ensemble, domain_shape, edge_weight, pert_mean,
                           pert_sigma, rf_approx_var, rf_eig, rf_vectors):
    L = 1
    k = 20
    x0 = 0.2
    ens_size = ensemble.shape[1]
    average = ensemble.mean(axis=1)
    average = average.reshape(domain_shape)
    cloud_target = 1 - average
    cloud_target = logistic(cloud_target, L=L, k=k, x0=x0)
    cloud_target = sp.ndimage.maximum_filter(cloud_target, size=9)
    cloud_target = sp.ndimage.gaussian_filter(cloud_target, sigma=5)
    cloud_target = cloud_target/cloud_target.max()
    cloud_target = cloud_target.clip(min=0, max=1)
    cloud_target = cloud_target.ravel()

    sample = np.random.randn(rf_eig.size, ens_size)
    sample = rf_vectors.dot(np.sqrt(rf_eig[:, None])*sample)
    target_mean = cloud_target.mean()
    target_var = (cloud_target**2).mean()
    cor_mean = pert_mean/target_mean
    cor_sd = pert_sigma/np.sqrt(rf_approx_var*target_var)
    # cor_sd = pert_sigma/np.sqrt(rf_approx_var)
    ensemble = (
        ensemble +
        (cor_sd*sample + cor_mean)*cloud_target[:, None])
    ensemble = ensemble.clip(min=ensemble.min(), max=1)
    return ensemble


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


def test_parallax(sat, domain_shape, dx, dy, lats, lons, sensor_data,
                  sensor_loc, start_time,
                  end_time, location, cloud_height,
                  sat_azimuth, sat_elevation,
                  oi_sat_sig, oi_sensor_sig, oi_localization, oi_inflation):
    """Check back later."""
    # NEED: Incorporate OI? Would need to reformulate so that P is smaller.
    sensor_loc = sensor_loc.sort_values(by='id', inplace=False)
    time_range = (pd.date_range(start_time, end_time, freq='15 min')
                  .tz_localize('MST'))
    all_time = sat.index
    time_range = np.intersect1d(time_range, all_time)

    sensor_loc_test = sensor_loc[sensor_loc.test is True]
    sensor_loc_assim = sensor_loc[sensor_loc.test is False]
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
        this_flat_sensor_loc_test = (flat_sensor_loc_test
                                     - flat_correct)  # changed to-
        sat_error[time_index] = (q[this_flat_sensor_loc_test] -
                                 sensor_data_test.ix[sat_time].values)

        this_flat_sensor_loc_assim = (flat_sensor_loc_assim
                                      - flat_correct)  # changed to -
        this_OI = optimal_interpolation(
            q.ravel(), oi_sat_sig, sensor_data_assim.ix[sat_time],
            oi_sensor_sig, q.ravel(), this_flat_sensor_loc_assim,
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
