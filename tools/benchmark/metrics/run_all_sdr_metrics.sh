#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -eu

self=$(realpath "$0")
mydir=$(dirname "${self}")

main() {
  local metrics=(
    FSIM-Y:"${mydir}"/fsim-y.sh
    FSIM-RGB:"${mydir}"/fsim-rgb.sh
    LPIPS:"${mydir}"/lpips-rgb.sh
    MS-SSIM-Y:"${mydir}"/msssim-y.sh
    NLPD:"${mydir}"/nlpd-y.sh
    SSIMULACRA:"${mydir}"/ssimulacra.sh
    VIF:"${mydir}"/vif-rgb.sh
    VMAF:"${mydir}"/vmaf.sh
  )
  # other metrics, not in core experiments:
#    VSI:"${mydir}"/vsi-rgb.sh
#    SSIM-RGB:"${mydir}"/ssim-rgb.sh
#    SSIM-Y:"${mydir}"/ssim-y.sh
#    GMSD:"${mydir}"/gmsd.sh
#    DISTS:"${mydir}"/dists-rgb.sh
#    MS-SSIM-RGB:"${mydir}"/msssim-rgb.sh

  local metrics_args=$(printf '%s' "${metrics[@]/#/,}")
  metrics_args=${metrics_args:1}


  "${mydir}/../../../build/tools/benchmark_xl" \
    --print_details_csv \
    --num_threads=1 \
    --error_pnorm=6 \
    --extra_metrics ${metrics_args} \
    "$@"
}

main "$@"
