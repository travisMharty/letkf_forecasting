# letkf_forecasting

The script run_forecast_system.py must be called with an argument which
is the path to a yaml configuration file such as config_example.py. The
conifig file must importantly point to the location of 2d wind fields and
satellite images for the day to be forecasted. Optionally,
run_forecast_system.py can also include the year, month, and day so that the
dates in the configuration file will be ignored and the provided date will be
used instead. There is also a bash file badly named run_bash.sh which can be
called to run several different days in unison.


The module letkf_forecasting.py is the core module. It will likely need to be
broken up into more modules. It includes functions for:
1. different methods of assimilation
1. ensemble creation
1. advection
1. removing divergence from wind fields
1. calculating optical flow vectors
1. the forecast_system function which runs the forecasting system.

The module letkf_io.py includes functions for opening and saving netcdf files.

The module tmh_plots.py includes functions which plot satellite and wind field
data.

The module interpolate_data.py includes functions to interpolate satellite and
WRF data onto a fine grid which is required for our advection scheme.

The module get_wrf_data.py includes functions used to open WRF output and save
the level of this data which has the highest relative humidity.

The module prepare_sat_data.py includes functions which are used to read
separate GOES-15 netcdf files and combine them into one file. This includes
things such as selecting all times of day for which the sun is up, converting
from longitude and latitude coordinates to coordinates based on distance,
interpolating satellite data on a square grid, and selecting all positions which
are within some distance from Tucson.

The module random_functions.py includes functions to create random fields used
to randomly perturb cloudiness index and wind fields.

## Packaging

To use ``letkf_forecasting`` now, run ``pip install -e .`` in this directory
in your conda env to install the packaged in the editable mode. This means
any changes you make to the code are automatically reflected in the package.
Then you can ``import letkf_forecasting`` instead of doing a sys.path.append.
You may need to reload the module/ipython kernel for changes in the library
code to be reflected in the notebook.
