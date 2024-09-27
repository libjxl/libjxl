#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

self=$(realpath "$0")
mydir=$(dirname "${self}")

./compute_octave_metric.sh "$@" \
  --path "${mydir}"/../../../third_party/hdr_metrics/ \
  "${mydir}"/compute-pumetrics.m 'ssim'
