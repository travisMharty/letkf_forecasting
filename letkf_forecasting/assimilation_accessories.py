import numpy as np

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
