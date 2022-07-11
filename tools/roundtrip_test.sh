#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# End-to-end roundtrip tests for cjxl and djxl tools.

MYDIR=$(dirname $(realpath "$0"))
JPEGXL_TEST_DATA_PATH="${MYDIR}/../testdata"

set -eux

EMULATOR=${EMULATOR:-}

# Temporary files cleanup hooks.
CLEANUP_FILES=()
cleanup() {
  if [[ ${#CLEANUP_FILES[@]} -ne 0 ]]; then
    rm -rf "${CLEANUP_FILES[@]}"
  fi
}
trap 'retcode=$?; { set +x; } 2>/dev/null; cleanup' INT TERM EXIT

roundtrip_test() {
  local infn="${JPEGXL_TEST_DATA_PATH}/$1"
  local encargs="$2"
  local maxdist="$3"
  
  local encoder="${EMULATOR} ${build_dir}/tools/cjxl"
  local decoder="${EMULATOR} ${build_dir}/tools/djxl"
  local comparator="${EMULATOR} ${build_dir}/tools/ssimulacra_main"
  local jxlfn="$(mktemp -p "$tmpdir")"

  "${encoder}" "${infn}" "${jxlfn}" $encargs

  if [ "${infn: -3}" == "jpg" ]; then
      local outfn="$(mktemp -p "$tmpdir").jpg"

      # Test losless jpeg reconstruction.
      "${decoder}" "${jxlfn}" "${outfn}" --num_reps 2
      diff "${infn}" "${outfn}"

      # Test decoding to pixels.
      "${decoder}" "${jxlfn}" "${outfn}" --num_reps 2 --pixels_to_jpeg
      local dist="$("${comparator}" "${infn}" "${outfn}")"
      python3 -c "import sys; sys.exit(not ${dist} > 0.0)"
      python3 -c "import sys; sys.exit(not ${dist} < 0.005)"
      
      # Test decoding to pixels by setting the --jpeg_quality flag.
      "${decoder}" "${jxlfn}" "${outfn}" --num_reps 2 --jpeg_quality 100
      local dist="$("${comparator}" "${infn}" "${outfn}")"
      python3 -c "import sys; sys.exit(not ${dist} > 0.0)"
      python3 -c "import sys; sys.exit(not ${dist} < 0.005)"

      # Test decoding to pixels by writing to a png.
      outfn="$(mktemp -p "$tmpdir").png"
      "${decoder}" "${jxlfn}" "${outfn}" --num_reps 2
      local dist="$("${comparator}" "${infn}" "${outfn}")"
      python3 -c "import sys; sys.exit(not ${dist} > 0.0)"
      python3 -c "import sys; sys.exit(not ${dist} < 0.005)"
  else
      # Test decoding to png.
      local outfn="$(mktemp -p "$tmpdir").png"
      "${decoder}" "${jxlfn}" "${outfn}" --num_reps 2
      local dist="$("${comparator}" "${infn}" "${outfn}")"
      python3 -c "import sys; sys.exit(not ${dist} <= ${maxdist})"

      # Test decoding to jpg.
      outfn="$(mktemp -p "$tmpdir").jpg"
      "${decoder}" "${jxlfn}" "${outfn}" --num_reps 2
      local dist="$("${comparator}" "${infn}" "${outfn}")"
      python3 -c "import sys; sys.exit(not ${dist} <= ${maxdist} + 0.05)"
  fi
}

main() {
  local tmpdir=$(mktemp -d)
  CLEANUP_FILES+=("${tmpdir}")

  local build_dir="${1:-}"
  if [[ -z "${build_dir}" ]]; then
    build_dir=$(realpath "${MYDIR}/../../build")
  fi

  roundtrip_test "jxl/flower/flower.png" "-e 1" 0.02
  roundtrip_test "jxl/flower/flower.png" "-e 1 -d 0.0" 0.0
  roundtrip_test "jxl/flower/flower_cropped.jpg" "-e 1" 0.0
}

main "$@"
