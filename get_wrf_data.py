import os
import numpy as np
import pandas as pd
import xarray as xr


def rh_calc(temps, pressure, qvapor):

    """
    Actually calculate rh from arrays
    """
    T1 = 273.16
    # Groff-Gratch equation
    ew = 10**(10.78574 * (1 - T1/temps)+
              - 5.028 * np.log10(temps/T1)+
              + 1.50475*10**-4 * (1 - 10**(-8.2969 * (temps/T1 - 1)))+
              + 0.42873*10**-3 * (10**(4.76955 * (1 - T1/temps)) - 1)+
              + 0.78614)
    # Bolton Equation
    # ew = ne.evaluate(
    #     '6.112*exp(17.62*(temps-273.15)/(243.21 + temps - 273.15))')
    es = ew*(1.0016+3.15*10**-6*pressure-0.074/pressure)
    e = qvapor/(qvapor+0.62198)*pressure
    rh = e/es*100
    return rh


def dewpoint_calc(qvapor, pressure):
    """
    Actually calculate tdew from arrays
    """
    e = qvapor/(qvapor+0.62198)*pressure
    # Bolton's fit
    a = 6.112  # millibars
    b = 17.67
    c = 243.5  # deg C
    logterm = np.log(e/a)
    tdew = (c*logterm)/(b-logterm)
    return tdew


def main(time_range, wrf_path, save_path):
    '''
    time_range: pd.DatetimeIndex
    file_path: location of wrf dataset
    '''
    dataset = xr.open_dataset(file_path)
    dataset['pressure'] = dataset['P'] + dataset['PB']
    dataset['temp'] = (dataset['T']+300)*(dataset['pressure']/100000)**0.2854
    dataset['rh'] = rh_calc(
        dataset['temp'], dataset['pressure']/100, dataset['QVAPOR'])
    dims = list(dataset['rh'].dims)
    dims.remove('Time')
    dims.remove('bottom_top')
    dataset['average_rh'] = dataset['rh'].mean(dim=dims)
    data_times = dataset['Times'].to_pandas().apply(
        lambda x: pd.to_datetime(x.decode("utf-8").replace('_', ' ')))
    data_times = pd.Index(data_times).tz_localize('UTC').tz_convert('MST')
    start_time = time_range[0]
    end_time = time_range[1]
    times_int = dataset.Time.values
    times_int = times_int[(data_times >= start_time) & (data_times <= end_time)]
    num_of_times = times_int.size()
    temp = dataset.sel(Time=Times_int[0])
    bottom_top = pd.Series()
    bottom_top[Times[0]] = temp['bottom_top'].where(
        temp['average_rh'] == temp['average_rh'].max(), drop=True).item()
    U = pd.DataFrame(
        data=temp['U'].sel(
            bottom_top = bottom_top[Times[0]]).values.ravel()[None, :],
        index=[Times[0]])
    V = pd.DataFrame(
        data=temp['V'].sel(
            bottom_top = bottom_top[Times[0]]).values.ravel()[None, :],
        index=[Times[0]])
    for t in np.arange(num_of_times - 1) + 1:
        temp = dataset.isel(Time=t)
        bottom_top[Times[t]] = temp['bottom_top'].where(
            temp['average_rh'] == temp['average_rh'].max(), drop=True).item()
        U_temp = pd.DataFrame(
            data=temp['U'].sel(bottom_top = bottom_top[Times[t]]).values.ravel()[None, :],
            index=[Times[t]])
        V_temp = pd.DataFrame(
            data=temp['V'].sel(bottom_top = bottom_top[Times[t]]).values.ravel()[None, :],
            index=[Times[t]])
        U = U.append(U_temp)
        V = V.append(V_temp)
    wind_lats = dataset['XLAT'].values.ravel()
    wind_lons = dataset['XLONG'].values.ravel()
    U_shape = dataset['U'].isel(Time=0, bottom_top=0).shape
    V_shape = dataset['V'].isel(Time=0, bottom_top=0).shape

    suffix = '_' + str(start_time.month) + '_' + str(start_time.day)
    save_path = save_path + 'for' + suffix + '/'
    print(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file = save_path + '{variable}{suffix}.h5'
    U.to_hdf(file.format(variable='U', suffix=suffix), 'U'+suffix)
    V.to_hdf(file.format(variable='V', suffix=suffix), 'V'+suffix)
    bottom_top.to_hdf(
        file.format(variable='bottom_top', suffix=suffix), 'bottom_top'+suffix)
    np.save(file.format(variable='wind_lats', suffix=suffix)[:-3],
            wind_lats)
    np.save(file.format(variable='wind_lons', suffix=suffix)[:-3],
            wind_lons)
    np.save(file.format(variable='U_shape', suffix=suffix)[:-3],
            U_shape)
    np.save(file.format(variable='V_shape', suffix=suffix)[:-3],
            V_shape)