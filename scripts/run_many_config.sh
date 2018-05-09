#!/bin/bash

year=2014
month=5
day=29
# declare -a config_list=("ow_1.yml" "ow_2.yml" "ow_3.yml" "ow_4.yml" "ow_5.yml")
# declare -a config_list=("ow_10.yml" "ow_11.yml" "ow_12.yml" "ow_13.yml")
declare -a config_list=("ow_15_wp2.yml" "ow_15_wp3.yml" "ow_15_wp4.yml")

for config in "${config_list[@]}"
do
    # config=($c)
    echo "$config"
    python run_forecast_system.py $config -y $year -m $month -d $day
done
