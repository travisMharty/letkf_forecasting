#!/bin/bash

year=2014
declare -a month_day=("4 9" "4 15" "4 18" "5 6" "5 9" "5 29" "6 11" "6 12")
# declare -a month_day=("5 6" "5 9" "5 29" "6 11" "6 12")


for md in "${month_day[@]}"
do
    date=($md)
    echo "${date[@]}"
    python create_data.py -y $year -m ${date[0]} -d ${date[1]}
done
