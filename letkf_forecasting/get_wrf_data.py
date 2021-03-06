import os
import numpy as np
import pandas as pd
import xarray as xr
import letkf_forecasting.prepare_sat_data as prep


def rh_calc(temps, pressure, qvapor):
    """
    Calculate rh from arrays
    """
    T1 = 273.16
    # Groff-Gratch equation
    ew = 10**(10.78574 * (1 - T1/temps)
              - 5.028 * np.log10(temps/T1)
              + 1.50475*10**-4 * (1 - 10**(-8.2969 * (temps/T1 - 1)))
              + 0.42873*10**-3 * (10**(4.76955 * (1 - T1/temps)) - 1)
              + 0.78614)d
    es = ew*(1.0016+3.15*10**-6*pressure-0.074/pressure)
    e = qvapor/(qvapor+0.62198)*pressure
    rh = e/es*100
    return rh


def dewpoint_calc(qvapor, pressure):
    """
    Calculate tdew from arrays
    """
    e = qvapor/(qvapor+0.62198)*pressure
    # Bolton's fit
    a = 6.112  # millibars
    b = 17.67
    c = 243.5  # deg C
    logterm = np.log(e/a)
    tdew = (c*logterm)/(b-logterm)
    return tdew


def main(time_range, wrf_path, interpolated_ci):
    '''
    time_range: pd.DatetimeIndex
    file_path: location of wrf dataset
    '''
    sat_x = interpolated_ci['x_coarse'].ravel()
    sat_y = interpolated_ci['y_coarse'].ravel()
    sat_lat, sat_lon = prep.lcc_to_sphere(sat_x, sat_y)
    lat_min = sat_lat.min()
    lat_max = sat_lat.max()
    lon_min = sat_lon.min()
    lon_max = sat_lon.max()

    dataset = xr.open_dataset(wrf_path)
    wrf_lat = dataset.XLAT[:, 0]
    wrf_lon = dataset.XLONG[0, :]
    we_min = np.searchsorted(wrf_lon, lon_min, side='left')
    we_max = np.searchsorted(wrf_lon, lon_max, side='left')
    if we_max == wrf_lon.size:
        we_slice = slice(we_min, we_max)
    else:
        we_slice = slice(we_min, we_max + 1)
    we_stag_slice = slice(we_min, we_slice.stop + 1)

    sn_min = np.searchsorted(wrf_lat, lat_min, side='left')
    sn_max = np.searchsorted(wrf_lat, lat_max, side='left')
    if sn_max == wrf_lat.size:
        sn_slice = slice(sn_min, sn_max)
    else:
        sn_slice = slice(sn_min, sn_max + 1)
    sn_stag_slice = slice(sn_min, sn_slice.stop + 1)

    dataset = dataset.sel(west_east=we_slice,
                          south_north=sn_slice,
                          west_east_stag=we_stag_slice,
                          south_north_stag=sn_stag_slice)

    data_times = dataset['Times'].to_pandas().apply(
        lambda x: pd.to_datetime(x.decode("utf-8").replace('_', ' ')))
    data_times = pd.Index(data_times).tz_localize('UTC').tz_convert('MST')
    start_time = time_range[0]
    end_time = time_range[-1]
    times_int = dataset.Time.values
    times_int = times_int[(data_times >= start_time)
                          & (data_times <= end_time)]
    data_times = data_times[times_int]
    num_of_times = times_int.size
    dataset = dataset.sel(Time=times_int)
    times_int = dataset.Time.values
    dataset['pressure'] = dataset['P'] + dataset['PB']
    dataset['temp'] = (dataset['T']+300)*(dataset['pressure']/100000)**0.2854
    dataset['rh'] = rh_calc(
        dataset['temp'], dataset['pressure']/100, dataset['QVAPOR'])
    dims = list(dataset['rh'].dims)
    dims.remove('Time')
    dims.remove('bottom_top')
    dataset['average_rh'] = dataset['rh'].mean(dim=dims)
    temp = dataset.sel(Time=times_int[0])
    bottom_top = pd.Series()
    bottom_top[data_times[0]] = temp['bottom_top'].where(
        temp['average_rh'] == temp['average_rh'].max(), drop=True).item()
    U = pd.DataFrame(
        data=temp['U'].sel(
            bottom_top=int(bottom_top[data_times[0]])).values.ravel()[None, :],
        index=[data_times[0]])
    V = pd.DataFrame(
        data=temp['V'].sel(
            bottom_top=int(bottom_top[data_times[0]])).values.ravel()[None, :],
        index=[data_times[0]])
    for t in np.arange(num_of_times - 1) + 1:
        temp = dataset.isel(Time=t)
        bottom_top[data_times[t]] = temp['bottom_top'].where(
            temp['average_rh'] == temp['average_rh'].max(), drop=True).item()
        U_temp = pd.DataFrame(
            data=temp['U'].sel(
                bottom_top=int(
                    bottom_top[data_times[t]])).values.ravel()[None, :],
            index=[data_times[t]])
        V_temp = pd.DataFrame(
            data=temp['V'].sel(
                bottom_top=int(
                    bottom_top[data_times[t]])).values.ravel()[None, :],
            index=[data_times[t]])
        U = U.append(U_temp)
        V = V.append(V_temp)
    wind_lats = dataset['XLAT'].values.ravel()
    wind_lons = dataset['XLONG'].values.ravel()
    U_shape = dataset['U'].isel(Time=0, bottom_top=0).shape
    V_shape = dataset['V'].isel(Time=0, bottom_top=0).shape
    to_return = {'U': U, 'V': V, 'bottom_top': bottom_top,
                 'wind_lats': wind_lats, 'wind_lons': wind_lons,
                 'U_shape': U_shape, 'V_shape': V_shape}
    return to_return
