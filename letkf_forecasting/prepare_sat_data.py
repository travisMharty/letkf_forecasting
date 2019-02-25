import numpy as np
import pandas as pd
import xarray as xr
import pvlib as pv
from glob import glob
from scipy import interpolate


def get_all_files(start_hour=11, end_hour=3):
    """
    start_hour - in UTC
    end_hour - in UTC"""
    if end_hour < start_hour:
        hour_range = np.union1d(np.arange(start_hour, 23 + 1),
                                np.arange(0, end_hour + 1))
    else:
        hour_range = np.arange(start_hour, end_hour)
    files = []
    for hour in hour_range:
        temp = glob(
            ('/a2/uaren/goes_images/april/goes15.2014.*.{hour:02d}*.BAND_01.nc'
             ).format(hour=hour))
        files = files + temp
        temp = glob(
            ('/a2/uaren/goes_images/may/goes15.2014.*.{hour:02d}*.BAND_01.nc'
             ).format(hour=hour))
        files = files + temp
        temp = glob(
            ('/a2/uaren/goes_images/june/goes15.2014.*.{hour:02d}*.BAND_01.nc'
             ).format(hour=hour))
        files = files + temp
    files.sort(key=lambda x: x.split('.2014.')[1])
    return files


def sphere_to_lcc(lats, lons, R=6370, truelat0=31.7, truelat1=31.7,
                  ref_lat=31.68858, stand_lon=-113.7):
    """
    Taken from Tony Lorenzo's repository at:
    https://github.com/alorenzo175/
         satellite_irradiance_optimal_interpolation.git.
    Convert from spherical lats/lons like what comes out of WRF to the WRF
    Lambert Conformal x/y coordinates. Defaults are what
    are generally used for the AZ domain
    """
    phis = np.radians(lats)
    lambdas = np.radians(lons)
    phi0 = np.radians(ref_lat)
    phi1 = np.radians(truelat0)
    phi2 = np.radians(truelat1)
    lambda0 = np.radians(stand_lon)

    if truelat0 == truelat1:
        n = np.sin(phi0)
    else:
        n = (np.log(np.cos(phi1) / np.cos(phi2)) /
             np.log(np.tan(np.pi / 4 + phi2 / 2) /
                    np.tan(np.pi / 4 + phi1 / 2)
                    ))
    F = (np.cos(phi1) * np.power(np.tan(np.pi / 4 + phi1 / 2), n) / n)
    rho0 = F / np.power(np.tan(np.pi / 4 + phi0 / 2), n)
    rho = F / np.power(np.tan(np.pi / 4 + phis / 2), n)
    x = R * rho * np.sin(n * (lambdas - lambda0))
    y = R * (rho0 - rho * np.cos(n * (lambdas - lambda0)))

    return x, y


def lcc_to_sphere(x, y, R=6370, truelat0=31.7, truelat1=31.7,
                  ref_lat=31.68858, stand_lon=-113.7):
    """
    Taken from Tony Lorenzo's repository at:
    https://github.com/alorenzo175/
         satellite_irradiance_optimal_interpolation.git.
    Convert from spherical lats/lons like what comes out of WRF to the WRF
    Lambert Conformal x/y coordinates. Defaults are what
    are generally used for the AZ domain
    """
    phi0 = np.radians(ref_lat)
    phi1 = np.radians(truelat0)
    phi2 = np.radians(truelat1)
    lambda0 = np.radians(stand_lon)

    if truelat0 == truelat1:
        n = np.sin(phi0)
    else:
        n = (np.log(np.cos(phi1) / np.cos(phi2)) /
             np.log(np.tan(np.pi / 4 + phi2 / 2) /
                    np.tan(np.pi / 4 + phi1 / 2)
                    ))
    F = (np.cos(phi1) * np.power(np.tan(np.pi / 4 + phi1 / 2), n) / n)
    rho0 = F / np.power(np.tan(np.pi / 4 + phi0 / 2), n)
    x = x / R
    y = y / R
    rho = np.sqrt(x**2 + (y - rho0)**2)
    phis = 2 * (np.arctan2(F**(1.0 / n), rho**(1.0 / n)) - np.pi / 4)
    lambdas = np.arcsin(x / rho) / n + lambda0

    return np.degrees(phis), np.degrees(lambdas)


def fine_crop(lat, lon, sat, x_slice, y_slice):
    sat = sat.isel(yc=y_slice, xc=x_slice)
    data = sat.data.values[0]
    sat_lat = sat.lat.values
    sat_lon = sat.lon.values
    data = data.ravel()
    sat_lat = sat_lat.ravel()
    sat_lon = sat_lon.ravel()
    data_positions = np.stack([sat_lat, sat_lon], axis=1)
    f = interpolate.NearestNDInterpolator(data_positions, data)
    shape = lat.shape
    interp_lat = lat.ravel()
    interp_lon = lon.ravel()
    interp_positions = np.stack([interp_lat, interp_lon], axis=1)
    return f(interp_positions).reshape(shape)


def midday(data, loc, max_zenith):
    if isinstance(data, pd.DataFrame):
        times = data.index
        zenith = loc.get_solarposition(times).loc[:, 'apparent_zenith']
        night = np.where(zenith >= max_zenith)[0]
        data.ix[night] = np.NAN
        data = data.dropna()
        return data

    if isinstance(data, xr.Dataset):
        times = (pd.to_datetime(data.time.values)
                 .tz_localize('UTC').tz_convert('MST'))
        zenith = loc.get_solarposition(times).loc[:, 'apparent_zenith']
        midday_times = np.where(zenith < max_zenith)
        data = data.isel(time=midday_times[0])
        return data


def get_cloudiness_index(pixel, lats, lons, elevation):
    # based on suny_ghi in input_data.py by Tony Lorenzo
    # https://github.com/alorenzo175/satellite_irradiance_optimal_interpolation/blob/master/satoi/input_data.py
    # pressure is in pascals, should be mb for solar_position
    pressure = pv.atmosphere.alt2pres(elevation)
    app_zenith = None
    for time in pixel.index:
        temp = pv.spa.solar_position(time.value/10**9, lats, lons, elevation,
                                     pressure/100, 12, 67.0, 0.)
        if app_zenith is None:
            app_zenith = pd.DataFrame({time: temp[0]})
        else:
            app_zenith[time] = temp[0]

    am = pv.atmosphere.absoluteairmass(pv.atmosphere.relativeairmass(
        app_zenith.T), pressure)
    soldist = pv.solarposition.pyephem_earthsun_distance(pixel.index).T.pow(2)
    norpix = (pixel * am).mul(soldist, axis=0)
    upper_bound = []
    lower_bound = []
    for c in norpix.columns:
        upper_bound.append(norpix[c].nlargest(20).mean())
        lower_bound.append(norpix[c].nsmallest(40).mean())
    low = pd.Series(lower_bound)
    up = pd.Series(upper_bound)

    cloudiness_index = (norpix - low) / (up - low)
    return low, up, cloudiness_index


def get_clearsky(times, elevation, lats, lons):
    pressure = pv.atmosphere.alt2pres(elevation)
    app_zenith = None
    zenith = None
    for time in times:
        temp = pv.spa.solar_position(time.value/10**9, lats, lons, elevation,
                                     pressure, 12, 67.0, 0.)
        if app_zenith is None:
            app_zenith = pd.DataFrame({time: temp[0]})
        else:
            app_zenith[time] = temp[0]

        if zenith is None:
            zenith = pd.DataFrame({time: temp[1]})
        else:
            zenith[time] = temp[1]
    am = pv.atmosphere.absoluteairmass(pv.atmosphere.relativeairmass(
        app_zenith.T), pressure)
    cos_zen = np.cos(np.radians(zenith.T))
    LT = pv.clearsky.lookup_linke_turbidity(times, lats.mean(), lons.mean())
    ones = pd.DataFrame(np.ones([times.size, lats.size]),
                        index=times, columns=np.arange(lats.size))
    LT = ones.mul(LT, axis=0)
    elev = ones.mul(elevation, axis=1)
    cg1 = (0.0000509 * elev + 0.868)
    cg2 = (0.0000392 * elev + 0.0387)
    I0 = ones.mul(pv.irradiance.extraradiation(times), axis=0)
    fh1 = np.exp(-1.0 / 8000 * elev)
    fh2 = np.exp(-1.0 / 1250 * elev)
    clearsky = cg1 * I0 * cos_zen * np.exp(-1.0 * cg2 * am * (
        fh1 + fh2 * (LT - 1))) * np.exp(0.01 * am**1.8)

    return clearsky


def main(dist_from_center, dx, x_slice, y_slice,
         tus_lat=32.2217, tus_lon=-110.9265):
    tus_x, tus_y = np.array(sphere_to_lcc(tus_lat, tus_lon))
    start = np.floor(tus_x - dist_from_center)
    end = np.ceil(tus_x + dist_from_center)
    x = np.arange(start, end + dx, dx)
    west_east = np.arange(x.size)
    start = np.floor(tus_y - dist_from_center)
    end = np.ceil(tus_y + dist_from_center)
    y = np.arange(start, end + dx, dx)
    south_north = np.arange(y.size)
    x, y = np.meshgrid(x, y)
    lat, lon = lcc_to_sphere(x, y)

    files = get_all_files()
    sat = xr.open_dataset(files[0])
    data = fine_crop(lat, lon, sat, x_slice, y_slice)[None, :, :]
    sat_dataset = xr.Dataset(
        {'data': (['time', 'south_north', 'west_east'], data)},
        coords={'west_east': west_east,
                'south_north': south_north,
                'lon': (['south_north', 'west_east'], lon),
                'lat': (['south_north', 'west_east'], lat),
                'x': (['south_north', 'west_east'], x),
                'y': (['south_north', 'west_east'], y),
                'time': sat.time.values.astype('int')})

    count = 0
    for file in files[1:]:
        sat = xr.open_dataset(file)
        data = fine_crop(lat, lon, sat, x_slice, y_slice)[None, :, :]
        temp = xr.Dataset(
            {'data': (['time', 'south_north', 'west_east'], data)},
            coords={'west_east': west_east,
                    'south_north': south_north,
                    'lon': (['south_north', 'west_east'], lon),
                    'lat': (['south_north', 'west_east'], lat),
                    'x': (['south_north', 'west_east'], x),
                    'y': (['south_north', 'west_east'], y),
                    'time': sat.time.values.astype('int')})
        sat_dataset = xr.concat([sat_dataset, temp], dim='time')
        count += 1
        if count % 500 == 0:
            sat_dataset.to_netcdf(
                path='/a2/uaren/goes_images/crop/sat_images.nc')
            print(count)
            print(pd.Timestamp(int(sat.time.values.astype('int')))
                  .tz_localize('UTC').tz_convert('MST'))
    sat_dataset.to_netcdf(path='/a2/uaren/goes_images/crop/sat_images.nc')
    print('finished')


if __name__ == '__main__':
    dist_from_center = 180  # km
    dx = 1  # km
    y_slice = slice(900 - 100, 1090 + 100)
    x_slice = slice(2200 - 200, 2570 + 220)
    main(dist_from_center, dx, x_slice, y_slice)
