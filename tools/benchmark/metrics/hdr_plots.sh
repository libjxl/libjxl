#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

self=$(realpath "$0")
mydir=$(dirname "${self}")

"${mydir}/run_all_hdr_metrics.sh" "$@" | sed -n '/```/q;p' > hdr_results.csv
mkdir -p hdr_plots/
rm -rf hdr_plots/*
python3 "${mydir}/plots.py" hdr_results.csv hdr_plots
