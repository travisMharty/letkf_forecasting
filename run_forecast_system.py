import numpy as np
import os
import pandas as pd
import logging
import time as time_py
import sys

sys.path.append('/home/travis/python_code/letkf_forecasting/')
import letkf_forecasting as lf

year = 2014
month = 5
day = 29

home = os.path.expanduser("~")
file_path = f'{home}/data/{year:04}/{month:02}/{day:02}/'
if not os.path.exists(file_path):
    os.makedirs(file_path)
file_path_ci = os.path.join(file_path, 'ci.h5')
print(file_path_ci)
file_path_winds = os.path.join(file_path, 'winds.h5')
print(file_path_winds)
param_dic = {
    #data_paths
    'ci_file_path':file_path_ci,
    'winds_file_path':file_path_winds,

    #Switches
    'assim_test':True,
    'perturbation_test':True,
    'div_test':True,
    'assim_of_test':True,
    'assim_sat2sat_test':False,
    'assim_sat2wind_test':False,
    'assim_wrf_test':True,

    #advection_params
    'start_time':None,
    'end_time':None,
    'C_max':0.7,
    'max_horizon': pd.Timedelta('1hour'),
    'n_workers':20,
    'scheduler_port':0,
    'diagnostics_port':0,

    #assimilation_params
    #assim_sat2sat
    'sig_sat2sat' : 0.05,
    'loc_sat2sat' : 1*4,
    'infl_sat2sat' : 1.5,
    'assim_gs_sat2sat':20, #if assim_sat is false, this is for sat into winds
    #assim_sat2wind
    'sig_sat2wind':1,
    'loc_sat2wind':30*4,
    'infl_sat2wind':4,
    'assim_gs_sat2wind':20,
    #assim_wrf
    'sig_wrf':0.5,
    'infl_wrf':1,
    'loc_wrf':1*4,
    'assim_gs_wrf':5,
    #assim_OF
    'sig_of':1,
    'loc_of':20, #in km not grid spaces,
    'infl_of':4, # 10 # was 1

    #ensemble_params
    'ens_num':20,
    'winds_sigma':(1, 1),
    'ci_sigma':.4,

    #perturbation_params
    'Lx':5,
    'Ly':5,
    'tol':0.005,
    'pert_sigma':0.15/3,
    'pert_mean':0,
    'edge_weight':1,
    }

# forecast_system
time0 = time_py.time()
# importlib.reload(logging)
# logging.basicConfig(filename='./logs/letkf.log', filemode='w', level=logging.DEBUG)
# logging.info('Started')

returned = lf.forecast_system(param_dic, **param_dic)
logging.info('Ended')
time1 = time_py.time()
print('It took: ' + str((time1 - time0)/60))
