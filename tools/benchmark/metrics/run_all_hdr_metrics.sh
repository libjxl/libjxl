#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eu
dir="$(dirname "$0")"

main() {
  local metrics=(
    HDR-VDP:"${dir}"/hdrvdp.sh
    MRSE:"${dir}"/mrse.sh
    puPSNR:"${dir}"/pupsnr.sh
    puSSIM:"${dir}"/pussim.sh
  )

  local metrics_args=$(printf '%s' "${metrics[@]/#/,}")
  metrics_args=${metrics_args:1}


  "${dir}/../../../build/tools/benchmark_xl" \
    --print_details_csv \
    --num_threads=32 \
    --error_pnorm=6 \
    --extra_metrics ${metrics_args} \
    "$@"
}

main "$@"
