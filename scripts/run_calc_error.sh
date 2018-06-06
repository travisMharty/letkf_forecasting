#!/bin/bash

year=2014
declare -a month_day=("4 9" "4 15" "4 18" "5 6" "5 9" "5 29" "6 11" "6 12")
# declare -a month_day=("4 9" "5 6" "5 9" "5 29" "6 11" "6 12")
# declare -a month_day=("5 19" "6 03" "6 10" "6 14" "6 15")

for md in "${month_day[@]}"
do
    date=($md)
    # echo "${date[@]}"
    # echo "wrf_no_div"
    # python calc_error.py config_wrf_no_div.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "radiosonde"
    # python calc_error.py config_radio.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "config_wrf_mean"
    # python calc_error.py config_wrf_mean.yml -y $year -m ${date[0]} -d ${date[1]}

    # echo "${date[@]}"
    # echo "config_opt_flow"
    # python calc_error.py config_opt_flow.yml -y $year -m ${date[0]} -d ${date[1]}
    echo "${date[@]}"
    echo "config_persistence"
    python calc_error.py config_persistence.yml -y $year -m ${date[0]} -d ${date[1]}
done
