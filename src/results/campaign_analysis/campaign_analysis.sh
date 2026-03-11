#!/bin/bash

for i in $(seq 0 11)
do
    taskset -c $((i)),$((i+24)) python campaign_analysis.py $((i)) &
done 
