# letkf_forecasting

There are currently 12 modules in this package:
1. advection.py
1. analyse_results.py
1. assimilation_accessories.py
1. assimilation.py *
1. get_wrf_data.py *
1. interpolate_data.py *
1. letkf_forecasting.py
1. letkf_io.py
1. optical_flow.py
1. prepare_sat_data.py *
1. random_functions.py
1. tmh_plot.py

Assimilation.py should be refactored in order to simplify functions and
eliminate duplicate code. The modules get_wrf_data.py, interpolate_data.py, and
prepare_sat_data.py require rewriting and need to be combined with code in
notebooks. This will be needed before a large statistical analysis.
optical_flow.py could also use some refactoring to increase readability.

The script run_forecast_system.py must be called with an argument which
is the path to a yaml configuration file such as config_example.py. The
conifig file must importantly point to the location of 2d wind fields and
satellite images for the day to be forecasted. Optionally,
run_forecast_system.py can also include the year, month, and day so that the
dates in the configuration file will be ignored and the provided date will be
used instead. There is also a bash file badly named run_bash.sh which can be
called to run several different days in unison.

The module advection.py contains functions required for the advection scheme.
This includes the functions for removing divergence since these functions fit
best hear, and didn't seem to contain enough to warrant their own module.

The module analyse_results.py contains functions useful in analyzing the
forecasts.

The modules assimilation_accessories.py contains functions needed for the
assimilation process, such as ensemble creation or creating a forward
observation matrix, but are not part of the actual assimilation step.

The module assimilation.py contains the core assimilation functions needed for
the LETKF, EnKF, and optimal interpolation. It needs some attention.

The module get_wrf_data.py includes functions used to open WRF output and save
the level of this data which has the highest relative humidity. It needs some
attention.

The module interpolate_data.py includes functions to interpolate satellite and
WRF data onto a fine grid which is required for our advection scheme. It needs
some attention.

The module letkf_forecasting.py is the core module. It contains the functions
needed to run the forecast system.

The module letkf_io.py includes functions for opening and saving netcdf files.

The module optical_flow.py contains one very large function for calculating
optical flow. This function needs to be broken up and simplified. It needs some
attention.


The module prepare_sat_data.py includes functions which are used to read
separate GOES-15 netcdf files and combine them into one file. This includes
things such as selecting all times of day for which the sun is up, converting
from longitude and latitude coordinates to coordinates based on distance,
interpolating satellite data on a square grid, and selecting all positions which
are within some distance from Tucson. It needs some attention.

The module random_functions.py includes the functions needed to randomly perturb
the ensemble's scalar field.

The module tmh_plots.py includes functions which plot satellite and wind field
data.

## Packaging

To use ``letkf_forecasting`` now, run ``pip install -e .`` in this directory
in your conda env to install the packaged in the editable mode. This means
any changes you make to the code are automatically reflected in the package.
Then you can ``import letkf_forecasting`` instead of doing a sys.path.append.
You may need to reload the module/ipython kernel for changes in the library
code to be reflected in the notebook.
