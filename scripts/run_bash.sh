#!/bin/bash

year=2014
declare -a month_day=("4 9" "4 15" "4 18" "5 6" "5 9" "5 29" "6 11" "6 12")
#declare -a month_day=("4 9" "5 9" "5 29" "6 11" "6 12")
# declare -a month_day=("6 11" "6 12")

for md in "${month_day[@]}"
do
    date=($md)
    # echo "${date[@]}"
    # echo "wrf_no_div"
    # python run_forecast_system.py config_wrf_no_div.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "radiosonde"
    # python run_forecast_system.py config_radio.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "ow_15"
    # python run_forecast_system.py ow_15.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "ow_15_wp3"
    # python run_forecast_system.py ow_15_wp3.yml -y $year -m ${date[0]} -d ${date[1]}

    echo "${date[@]}"
    echo "opt_flow"
    python run_forecast_system.py config_opt_flow.yml -y $year -m ${date[0]} -d ${date[1]}
done
