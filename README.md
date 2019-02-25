# letkf_forecasting
This package was used to generate cloud index forecasts for the paper "Intra-hour cloud index forecasting with data assimilation." The cloud index and cloud motion vector fields used in this paper are available on [zenodo](https://zenodo.org/record/2574203#.XHQb1VNKh24).

There are 5 scripts in this package:
1. create_data.py
1. generate_plots.py
1. optimize.py
1. run_create_data.sh
1. run_forecast_system.py
1. run_many_config.sh
1. run_many_days.sh

The script create_data.py reads satellite data from:
/a2/uaren/data/satellite_data/cloudiness_index.h5
and
/a2/uaren/2014/*/*/solar_3/wrfsolar_d02_hourly.nc
and saves interpolated data to:
/a2/uaren/travis/data/2014/*/*/data.nc

The script generate_plots.py generates the plots used in "Intra-hour cloud index forecasting with data assimilation."

The script optimize.py runs several different parameter values for localization and inflation in order to determine which of the different sets produces forecasts with the lowest error.

The script run_create_data.sh is used to run several dates through
create_data.py.

The script run_forecast_system.py is used to run the forecast_system function
in letkf_forecasting.py. This script must be called with an argument which
is the path to a yaml configuration file such as config_example.py. The
conifig file must importantly point to the location of 2d wind fields and
satellite images for the day to be forecasted. Optionally,
run_forecast_system.py can also include the year, month, and day so that the
dates in the configuration file will be ignored and the provided date will be
used instead.

The script run_many_config.sh is used to run many different configuration files
on the same date.

The script run_many_days.sh is used to run many different dates on the same
configuration files.

There are 8 configuration files in this package:
1. config_analysis_fore.yml (ANOC control forecast)
1. config_opt_flow.yml (dense optical flow no divergence)
1. config_opt_flow_with_div.yml (dense optical flow with divergence)
1. config_owp_opt.yml (ANOC forecast)
1. config_radio.yml (Radiosonde forecast)
1. config_wrf_mean.yml (NWP mean wind field)
1. config_wrf_no_div.yml (NWP winds no divergence)
1. config_wrf.yml (NWP winds with divergence)

Theses configuration files include the values of many different parameters
including flags which determine which (if any) assimilations take place,
the locations to read data from, and the values of parameters for assimilation
and perturbation.

There are 13 modules in this package:
1. advection.py
1. advect.pyx
1. analyse_results.py
1. assimilation_accessories.py
1. assimilation.py
1. get_wrf_data.py
1. interpolate_data.py
1. letkf_forecasting.py
1. letkf_io.py
1. optical_flow.py
1. prepare_sat_data.py
1. random_functions.py
1. tmh_plot.py

The module advection.py contains functions required for the advection scheme.
This includes the functions for removing divergence.

The module advect.pyx is cython code used to speed up advection.

The module analyse_results.py contains functions useful in analyzing the
forecasts.

The modules assimilation_accessories.py contains functions needed for the
assimilation process, such as ensemble creation or creating a forward
observation matrix, but are not part of the actual assimilation step.

The module assimilation.py contains the core assimilation functions needed for
the LETKF, EnKF, and optimal interpolation.

The module get_wrf_data.py includes functions used to open WRF output and save
the level of this data which has the highest relative humidity.

The module interpolate_data.py includes functions to interpolate satellite and
WRF data onto a fine grid which is required for our advection scheme.

The module letkf_forecasting.py is the core module. It contains the functions
needed to run the forecast system.

The module letkf_io.py includes functions for opening and saving netcdf files.

The module optical_flow.py contains one very large function for calculating
optical flow. This function needs to be broken up and simplified.

The module prepare_sat_data.py includes functions which are used to read
separate GOES-15 netcdf files and combine them into one file. This includes
things such as selecting all times of day for which the sun is up, map projection,
interpolating satellite data on a square grid, and selecting all positions which
are within some distance from Tucson.

The module random_functions.py includes the functions needed to randomly perturb
the ensemble.

The module tmh_plots.py includes functions which plot satellite and wind field
data.

There are 6 notebooks in this package:
1. analyze_results_all_days_calc_errors.ipynb
1. domains_figure.ipynb
1. error_field_plots.ipynb
1. analyze_results_calc_day_average_error.ipynb
1. ensemble_stamps.ipynb
1. run_prepare_data.ipynb

The notebooks analyze_results_all_days_calc_errors.ipynb and analyze_results_calc_day_average_error.ipynb are used to calculate error statistics of the cloud index forecasts.

The notebooks domains_figure.ipynb, error_field_plots.ipynb, and ensemble_stamps.ipynb generate figures.

The notebook run_prepare_data.ipynb calculates and saves CI fields based on geostationary satellite images.

## Packaging

To use ``letkf_forecasting`` now, run ``pip install -e .`` in this directory
in your conda env to install the packaged in the editable mode. This means
any changes you make to the code are automatically reflected in the package.
Then you can ``import letkf_forecasting``.
You may need to reload the module/ipython kernel for changes in the library
code to be reflected in the notebook.
