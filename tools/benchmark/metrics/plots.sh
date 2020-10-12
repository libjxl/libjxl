#!/bin/bash

"$(dirname "$0")/run_all_metrics.sh" "$@" | sed -n '/```/q;p' > results.csv
mkdir -p plots/
rm -rf plots/*
python3 "$(dirname "$0")/plots.py" results.csv

