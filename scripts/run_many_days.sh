#!/bin/bash

year=2014

# First batch
# declare -a month_day=("4 9" "4 15" "4 18" "5 6" "5 9" "5 29" "6 11" "6 12")

# Second batch
# declare -a month_day=("4 2" "4 5" "4 19" "5 7" "5 8" "5 19" "6 03" "6 10" "6 14" "6 15")
# declare -a month_day=("4 9" "4 15" "4 18" "5 6" "5 9" "5 29" "6 11" "6 12" "4 2" "4 5" "4 19" "5 7" "5 8" "5 19" "6 03" "6 10" "6 14" "6 15")

# Third batch First part
# declare -a month_day=("4 10" "4 11" "4 12" "4 20" "4 21" "4 22" "4 25" "4 26")
# Third batch Second part
# declare -a month_day=("5 5" "5 20" "5 21" "5 22" "5 23" "5 24" "5 25" "5 30" "6 16" "6 17" "6 18" "6 19" "6 22")
declare -a month_day=("4 10" "4 11" "4 12" "4 20" "4 21" "4 22" "4 25" "4 26" "5 5" "5 20" "5 21" "5 22" "5 23" "5 24" "5 25" "5 30" "6 16" "6 17" "6 18" "6 19" "6 22")

for md in "${month_day[@]}"
do
    date=($md)
    echo "${date[@]}"
    echo "opt_flow"
    python run_forecast_system.py config_opt_flow.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "ow_15"
    # python run_forecast_system.py ow_15.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "owp_opt.yml"
    # python run_forecast_system.py config_owp_opt.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "radiosonde"
    # python run_forecast_system.py config_radio.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "wrf_no_div"
    # python run_forecast_system.py config_wrf_no_div.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "wrf"
    # python run_forecast_system.py config_wrf.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "wrf_mean.yml"
    # python run_forecast_system.py config_wrf_mean.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "ow_15_wp3"
    # python run_forecast_system.py ow_15_wp3.yml -y $year -m ${date[0]} -d ${date[1]}
done
