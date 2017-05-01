import numpy as np
import pandas as pd


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
                   satellite_altitude,
                   solar_azimuth,
                   solar_altitude):
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
    satellite_displacement = cloud_height*cot(satellite_altitude)
    solar_displacement = cloud_height*cot(solar_altitude)
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
    print(sensor_loc.shape)
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
