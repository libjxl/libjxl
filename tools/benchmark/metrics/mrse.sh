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

set -euo pipefail

original="$1"
decoded="$2"
output="$3"
intensity_target="$4"

tmpdir="$(mktemp --directory)"

linearized_original="$(mktemp --tmpdir="$tmpdir" --suffix='.pfm')"
linearized_decoded="$(mktemp --tmpdir="$tmpdir" --suffix='.pfm')"

cleanup() {
  rm -- "$linearized_original" "$linearized_decoded"
  rmdir --ignore-fail-on-non-empty -- "$tmpdir"
}
trap cleanup EXIT

linearize() {
  local input="$1"
  local output="$2"
  convert "$input" -set colorspace sRGB -colorspace RGB -evaluate multiply "$intensity_target" "$output"
}

linearize "$original" "$linearized_original"
linearize "$decoded" "$linearized_decoded"

"$(dirname "$0")"/../../../third_party/difftest_ng/difftest_ng --mrse "$linearized_original" "$linearized_decoded" \
  | sed -e 's/^MRSE:\s*//' \
  > "$output"
