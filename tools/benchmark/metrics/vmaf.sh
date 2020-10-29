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

tmpdir="$(mktemp --directory)"

exr_original="$(mktemp --tmpdir="$tmpdir" --suffix='.exr')"
exr_decoded="$(mktemp --tmpdir="$tmpdir" --suffix='.exr')"

yuv_original="$(mktemp --tmpdir="$tmpdir" --suffix='.yuv')"
yuv_decoded="$(mktemp --tmpdir="$tmpdir" --suffix='.yuv')"

vmaf_csv="$(mktemp --tmpdir="$tmpdir" --suffix='.csv')"

cleanup() {
  rm -- "$exr_original" "$exr_decoded" "$yuv_original" "$yuv_decoded" "$vmaf_csv"
  rmdir --ignore-fail-on-non-empty -- "$tmpdir"
}
trap cleanup EXIT

convert "$original" "$exr_original"
convert "$decoded" "$exr_decoded"

srgb=(-colorspace bt709 -color_primaries bt709 -color_trc iec61966-2-1)
ffmpeg "${srgb[@]}" -i "$exr_original" -pix_fmt yuv444p10le "${srgb[@]}" -y "$yuv_original" &>/dev/null
ffmpeg "${srgb[@]}" -i "$exr_decoded" -pix_fmt yuv444p10le "${srgb[@]}" -y "$yuv_decoded" &>/dev/null

"$(dirname "$0")"/../../../third_party/vmaf/libvmaf/build/tools/vmafossexec \
  yuv444p10le \
  "$(identify -format '%w' "$original")" "$(identify -format '%h' "$original")" \
  "$yuv_original" "$yuv_decoded" \
  ../../../third_party/vmaf/model/vmaf_v0.6.1.pkl \
  --log-fmt csv --log "$vmaf_csv" &>/dev/null

read_csv="$(cat <<'END'
import csv
import sys
reader = csv.DictReader(sys.stdin)
for row in reader:
  print(row['vmaf'])
END
)"

python -c "$read_csv" < "$vmaf_csv" > "$output"
