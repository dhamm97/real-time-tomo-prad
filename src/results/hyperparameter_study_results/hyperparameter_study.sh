#!/bin/bash

for i in $(seq 0 9);
do
    python hyperparameter_study.py $((i * 100)) $(((i+1) * 100))  &
done
