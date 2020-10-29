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

MYDIR=$(dirname $(realpath "$0"))


main() {
  cd "${MYDIR}/../../../third_party"
  local zipurl
  for zipurl in \
    'https://sourceforge.net/projects/hdrvdp/files/hdrvdp/2.2.2/hdrvdp-2.2.2.zip' \
    'https://sourceforge.net/projects/hdrvdp/files/simple_metrics/1.0/hdr_metrics.zip'
  do
    local zipfile="$(basename "${zipurl}")"
    local dirname="$(basename "${zipfile}" '.zip')"
    rm -fr "${dirname}"
    if [[ ! -e "${zipfile}" ]]; then
      wget -O "${zipfile}.tmp" "${zipurl}"
      mv "${zipfile}.tmp" "${zipfile}"
    fi
    unzip "${zipfile}" "${dirname}"/'*'
  done

  pushd hdrvdp-2.2.2
  patch -p1 < ../../tools/benchmark/metrics/hdrvdp-fixes.patch
  pushd matlabPyrTools_1.4_fixed
  mkoctfile --mex MEX/corrDn.c MEX/convolve.c MEX/wrap.c MEX/edges.c
  mkoctfile --mex MEX/pointOp.c
  mkoctfile --mex MEX/upConv.c
  popd
  popd


  pushd difftest_ng
  ./configure
  make
  popd


  pushd vmaf/libvmaf
  rm -rf build
  meson build --buildtype release
  ninja -vC build
  popd
}
main "$@"

