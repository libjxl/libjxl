#!/bin/bash
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

# Conformance test tooling test. This is not the JPEG XL conformance test
# runner. This test that the tooling to generate the conformance test and the
# conformance test runner work together.

MYDIR=$(dirname $(realpath "$0"))

set -eux

# Temporary files cleanup hooks.
CLEANUP_FILES=()
cleanup() {
  if [[ ${#CLEANUP_FILES[@]} -ne 0 ]]; then
    rm -rf "${CLEANUP_FILES[@]}"
  fi
}
trap 'retcode=$?; { set +x; } 2>/dev/null; cleanup' INT TERM EXIT

main() {
  local tmpdir=$(mktemp -d)
  CLEANUP_FILES+=("${tmpdir}")

  if ! python3 -c 'import yaml'; then
    echo "Missing yaml, skipping test." >&2
    exit 254  # Signals ctest that we should mark this test as skipped.
  fi

  local build_dir="${1:-}"
  if [[ -z "${build_dir}" ]]; then
    build_dir=$(realpath "${MYDIR}/../../build")
  fi

  local decoder="${build_dir}/tools/conformance/djxl_conformance"
  "${MYDIR}/generator.py" \
    --decoder="${decoder}" \
    --output="${tmpdir}" \
    "${MYDIR}/../../third_party/testdata/jxl/blending/cropped_traffic_light.jxl"

  # List the contents of the corpus dir.
  tree "${tmpdir}" || true

  "${MYDIR}/conformance.py" \
    --decoder="${decoder}" \
    --corpus="${tmpdir}"
}

main "$@"
