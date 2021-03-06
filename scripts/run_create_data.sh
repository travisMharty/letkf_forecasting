#!/bin/bash

# No Solar3 wrf hourly data: "4 3" "4 4" "4 16" "6 13" "6 20" "6 24" "6 25" "6 26"

year=2014
# declare -a month_day=("4 9" "4 15" "4 18" "5 6" "5 9" "5 29" "6 11" "6 12")
# declare -a month_day=("4 2" "4 4" "4 5" "5 7" "5 8" "5 19" "6 3" "6 10" "6 3" "6 10" "6 13" "6 14")
# declare -a month_day=("4 10" "4 11" "4 12" "4 20" "4 21" "4 22" "4 25" "4 26")

declare -a month_day=("5 5" "5 20" "5 21" "5 22" "5 23" "5 24" "5 25" "5 30" "6 16" "6 17" "6 18" "6 19" "6 22")

for md in "${month_day[@]}"
do
    date=($md)
    echo "${date[@]}"
    python create_data.py -y $year -m ${date[0]} -d ${date[1]}
done
