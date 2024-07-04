#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

self=$(realpath "$0")
mydir=$(dirname "${self}")

"${mydir}/run_all_sdr_metrics.sh" "$@" | sed -n '/```/q;p' > sdr_results.csv
mkdir -p sdr_plots/
rm -rf sdr_plots/*
python3 "${mydir}/plots.py" sdr_results.csv sdr_plots
