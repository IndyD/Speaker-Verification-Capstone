#!/bin/bash

now=$(date +"%T")

for config in "$@"; do
  python3 generate_data.py "$config" 2>&1 | tee "output/log_$now.txt"
  python3 run_model.py "$config" 2>&1 | tee -a "output/log_$now.txt"
done