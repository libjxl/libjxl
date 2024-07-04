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
    HDR-VDP:"${mydir}"/hdrvdp.sh
    MRSE:"${mydir}"/mrse.sh
    puPSNR:"${mydir}"/pupsnr.sh
    puSSIM:"${mydir}"/pussim.sh
  )

  local metrics_args=$(printf '%s' "${metrics[@]/#/,}")
  metrics_args=${metrics_args:1}


  "${mydir}/../../../build/tools/benchmark_xl" \
    --print_details_csv \
    --num_threads=32 \
    --error_pnorm=6 \
    --extra_metrics ${metrics_args} \
    "$@"
}

main "$@"
