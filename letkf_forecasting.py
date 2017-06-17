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
    in the units of cloud_height.v

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
            idx[i] =  idx[i] - 1
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
                          flat_locations, localization, sat_inflation):

    PHT = np.abs(P_field[:, None] - P_field[None, flat_locations])
    PHT = np.exp(-PHT**2/(localization**2))
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

        W_fun = interpolate.LinearNDInterpolator(assimilation_positions_2d,
                                                 W_interp)
        W_fine_mesh = W_fun(full_positions_2d)
        W_fine_mesh = W_fine_mesh.reshape(domain_shape[0]*domain_shape[1],
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


def advect_5min(q, noise, ensemble, dt, U, dx, V, dy, T_steps, wind_size):
    """Check back later"""
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


def find_flat_loc(sat, sensor_loc):
    sat_lat = sat.lat.values[:, 0]
    sat_lon = sat.long.values[0, :]
    shape = sat.lat.shape
    sensor_lat = sensor_loc['lat'].values
    sensor_lon = sensor_loc['lon'].values
    sensor_lat_indices = to_nearest_indices(sat_lat, sensor_lat)
    sensor_lon_indices = to_nearest_indices(sat_lon, sensor_lon)
    lat_step = sat_lat[1] - sat_lat[0]
    lon_step = sat_lon[1] - sat_lon[0]
    sensor_indices_2d = np.stack([sensor_lat_indices, sensor_lon_indices])
    sensor_indices_flat = np.ravel_multi_index(
        multi_index=(sensor_lat_indices, sensor_lon_indices),
        dims=shape)
    return sensor_indices_flat, lat_step, lon_step


def get_flat_correct(
        cloud_height, lat_step, lon_step, domain_shape, sat_azimuth,
        sat_elevation, location, sensor_time):


    solar_position = location.get_solarposition(sensor_time)
    x_correct, y_correct = parallax_shift(
        cloud_height, sat_azimuth, sat_elevation,
        solar_position['azimuth'].values,
        solar_position['elevation'].values)
    lat_correct, lon_correct = to_lat_lon(x_correct, y_correct,
                                          location.latitude)
    west_east_correct = int(np.round(lon_correct/lon_step))
    south_north_correct = int(np.round(lat_correct/lat_step))
    flat_correct = west_east_correct + south_north_correct*domain_shape[0]
    return flat_correct


# Only ends at satellite times
def main(sat, wind, sensor_data, sensor_loc, start_time,
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
    return analysis, analysis_error, background, background_error, advected, time_range

def test_parallax(sat, sensor_data, sensor_loc, start_time,
                       end_time, location, cloud_height,
                       sat_azimuth, sat_elevation):
    """Check back later."""
    ## NEED: Incorporate IO? Would need to reformulate so that P is smaller.
    sensor_loc = sensor_loc.sort_values(by='id', inplace=False)
    time_range = (pd.date_range(start_time, end_time, freq='15 min')
                  .tz_localize('MST').astype(int))
    all_time = sat.time.values
    time_range = np.intersect1d(time_range, all_time)
    sat_loc = np.concatenate(
        (sat['lat'].values.ravel()[:, None],
         sat['long'].values.ravel()[:, None]), axis=1)
    domain_shape = sat['clear_sky_good'].isel(time=0).shape
    flat_sensor_loc, lat_step, lon_step = find_flat_loc(
        sat, sensor_loc)
    error = np.ones([time_range.size, flat_sensor_loc.size])*np.nan
    lat_correction = np.ones(time_range.size)*np.nan
    lon_correction = np.ones(time_range.size)*np.nan
    for time_index in range(time_range.size):
        sat_int_time = time_range[time_index]
        q = sat['clear_sky_good'].sel(time=sat_int_time).values.ravel()
        sat_time = pd.Timestamp(sat_int_time).tz_localize('UTC').tz_convert('MST')
        flat_correct = get_flat_correct(
                cloud_height=cloud_height, lat_step=lat_step, lon_step=lon_step,
                domain_shape=domain_shape, sat_azimuth=sat_azimuth,
                sat_elevation=sat_elevation,
                location=location, sensor_time=sat_time)
        this_flat_sensor_loc = flat_sensor_loc + flat_correct
        error[time_index] = (q[this_flat_sensor_loc] -
                             sensor_data.ix[sat_time].values)
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
    return error, lat_correction, lon_correction, time_range
