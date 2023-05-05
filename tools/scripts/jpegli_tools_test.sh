#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# End-to-end roundtrip tests for cjpegli and djpegli tools.

set -eux

MYDIR=$(dirname $(realpath "$0"))
JPEGXL_TEST_DATA_PATH="${MYDIR}/../../testdata"

# Temporary files cleanup hooks.
CLEANUP_FILES=()
cleanup() {
  if [[ ${#CLEANUP_FILES[@]} -ne 0 ]]; then
    rm -rf "${CLEANUP_FILES[@]}"
  fi
}
trap 'retcode=$?; { set +x; } 2>/dev/null; cleanup' INT TERM EXIT

cjpegli_test() {
  local infn="${JPEGXL_TEST_DATA_PATH}/$1"
  local encargs="$2"
  local minscore="$3"
  local maxbpp="$4"
  local jpgfn="$(mktemp -p "$tmpdir")"
  local outfn="$(mktemp -p "${tmpdir}").png"

  "${encoder}" "${infn}" "${jpgfn}" $encargs
  "${decoder}" "${jpgfn}" "${outfn}"
  local score="$("${comparator}" "${infn}" "${outfn}")"
  python3 -c "import sys; sys.exit(not ${score} >= ${minscore})"
  local size="$(wc -c "${jpgfn}" | cut -d' ' -f1)"
  local pixels=$(( "$(identify "${infn}" | cut -d' ' -f3 | tr 'x' '*')" ))
  python3 -c "import sys; sys.exit(not ${size} * 8 <= ${maxbpp} * ${pixels})"
}

cjpegli_test_target_size() {
  local infn="${JPEGXL_TEST_DATA_PATH}/$1"
  local encargs="$2"
  local target_size="$3"
  local jpgfn="$(mktemp -p "$tmpdir")"
  local outfn="$(mktemp -p "${tmpdir}").png"

  "${encoder}" "${infn}" "${jpgfn}" $encargs --target_size "${target_size}"
  "${decoder}" "${jpgfn}" "${outfn}"
  local size="$(wc -c "${jpgfn}" | cut -d' ' -f1)"
  python3 -c "import sys; sys.exit(not ${target_size} * 0.996 <= ${size})"
  python3 -c "import sys; sys.exit(not ${target_size} * 1.004 >= ${size})"
}

djpegli_test() {
  local infn="${JPEGXL_TEST_DATA_PATH}/$1"
  local encargs="$2"
  local minscore="$3"
  local jpgfn="$(mktemp -p "$tmpdir")"

  "${encoder}" "${infn}" "${jpgfn}" $encargs

  # Test that disabling output works.
  "${decoder}" "${jpgfn}" --disable_output
  for ext in png pgm ppm pfm pnm baz; do
    "${decoder}" "${jpgfn}" /foo/bar.$ext --disable_output
  done

  # Test decoding to PNG, PPM, PNM, PFM
  for ext in png ppm pnm pfm; do
    local outfn="$(mktemp -p "${tmpdir}").${ext}"
    "${decoder}" "${jpgfn}" "${outfn}" --num_reps 2
    local score="$("${comparator}" "${infn}" "${outfn}")"
    python3 -c "import sys; sys.exit(not ${score} >= ${minscore})"
  done

  # Test decoding to PGM (for grayscale input)
  if [[ "${infn: -6}" == ".g.png" ]]; then
    local outfn="$(mktemp -p "${tmpdir}").pgm"
    "${decoder}" "${jpgfn}" "${outfn}" --quiet
    local score="$("${comparator}" "${infn}" "${outfn}")"
    python3 -c "import sys; sys.exit(not ${score} >= ${minscore})"
  fi

  # Test decoding to 16 bit
  for ext in png pnm; do
    local outfn8="$(mktemp -p "${tmpdir}").8.${ext}"
    local outfn16="$(mktemp -p "${tmpdir}").16.${ext}"
    "${decoder}" "${jpgfn}" "${outfn8}"
    "${decoder}" "${jpgfn}" "${outfn16}" --bitdepth 16
    local score8="$("${comparator}" "${infn}" "${outfn8}")"
    local score16="$("${comparator}" "${infn}" "${outfn16}")"
    python3 -c "import sys; sys.exit(not ${score16} > ${score8})"
  done
}

main() {
  local tmpdir=$(mktemp -d)
  CLEANUP_FILES+=("${tmpdir}")

  local build_dir="${1:-}"
  if [[ -z "${build_dir}" ]]; then
    build_dir=$(realpath "${MYDIR}/../../build")
  fi

  local encoder="${build_dir}/tools/cjpegli"
  local decoder="${build_dir}/tools/djpegli"
  local comparator="${build_dir}/tools/ssimulacra2"

  local rgb_in="jxl/flower/flower_small.rgb.png"
  local gray_in="jxl/flower/flower_small.g.png"

  cjpegli_test "${rgb_in}" "" 89 1.7
  cjpegli_test "${rgb_in}" "-q 80" 84 1.2
  cjpegli_test "${rgb_in}" "-q 95" 92 2.4
  cjpegli_test "${rgb_in}" "-d 0.5" 93 2.6
  cjpegli_test "${rgb_in}" "--chroma_subsampling 420" 86 1.5
  cjpegli_test "${rgb_in}" "--chroma_subsampling 440" 88 1.6
  cjpegli_test "${rgb_in}" "--chroma_subsampling 422" 88 1.6
  cjpegli_test "${rgb_in}" "--xyb" 87 1.5
  cjpegli_test "${rgb_in}" "--std_quant" 91 2.2
  cjpegli_test "${rgb_in}" "--noadaptive_quantization" 90 1.8
  cjpegli_test "${rgb_in}" "-p 1" 89 1.72
  cjpegli_test "${rgb_in}" "-p 0" 89 1.75
  cjpegli_test "${rgb_in}" "-p 0 --fixed_code" 89 1.8
  cjpegli_test "${gray_in}" "" 92 1.4

  cjpegli_test_target_size "${rgb_in}" "" 10000
  cjpegli_test_target_size "${rgb_in}" "" 50000
  cjpegli_test_target_size "${rgb_in}" "" 100000
  cjpegli_test_target_size "${rgb_in}" "--chroma_subsampling 420" 20000
  cjpegli_test_target_size "${rgb_in}" "--xyb" 20000
  cjpegli_test_target_size "${rgb_in}" "-p 0 --fixed_code" 20000

  cjpegli_test "jxl/flower/flower_small.rgb.depth8.ppm" "" 89 1.7
  cjpegli_test "jxl/flower/flower_small.rgb.depth16.ppm" "" 89 1.7
  cjpegli_test "jxl/flower/flower_small.g.depth8.pgm" "" 89 1.7
  cjpegli_test "jxl/flower/flower_small.g.depth16.pgm" "" 89 1.7

  djpegli_test "${rgb_in}" "-q 95" 92
  djpegli_test "${gray_in}" "-q 95" 92
}

main "$@"
