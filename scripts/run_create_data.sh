#!/bin/bash

year=2014
# declare -a month_day=("4 9" "4 15" "4 18" "5 6" "5 9" "5 29" "6 11" "6 12")
# declare -a month_day=("5 6" "5 9" "5 29" "6 11" "6 12")
# No Solar3 wrf hourly data: "4 3" "4 4" "4 16" "6 13" "6 20" "6 24" "6 25" "6 26"
# declare -a month_day=("4 2" "4 4" "4 5" "5 7" "5 8" "5 19" "6 3" "6 10" "6 3" "6 10" "6 13" "6 14")
declare -a month_day=("6 15")


for md in "${month_day[@]}"
do
    date=($md)
    echo "${date[@]}"
    python create_data.py -y $year -m ${date[0]} -d ${date[1]}
done
