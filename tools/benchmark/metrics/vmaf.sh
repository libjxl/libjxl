#!/bin/bash

set -euo pipefail

original="$1"
decoded="$2"
output="$3"

tmpdir="$(mktemp --directory)"

normalized_original="$(mktemp --tmpdir="$tmpdir" --suffix='.exr')"
normalized_decoded="$(mktemp --tmpdir="$tmpdir" --suffix='.exr')"

yuv_original="$(mktemp --tmpdir="$tmpdir" --suffix='.yuv')"
yuv_decoded="$(mktemp --tmpdir="$tmpdir" --suffix='.yuv')"

vmaf_csv="$(mktemp --tmpdir="$tmpdir" --suffix='.csv')"

cleanup() {
  rm -- "$normalized_original" "$normalized_decoded" "$yuv_original" "$yuv_decoded" "$vmaf_csv"
  rmdir --ignore-fail-on-non-empty -- "$tmpdir"
}
trap cleanup EXIT

convert "$original" -evaluate divide 255 "$normalized_original"
convert "$decoded" -evaluate divide 255 "$normalized_decoded"

srgb=(-colorspace bt709 -color_primaries bt709 -color_trc iec61966-2-1)
ffmpeg "${srgb[@]}" -i "$normalized_original" -pix_fmt yuv444p10le "${srgb[@]}" -y "$yuv_original"
ffmpeg "${srgb[@]}" -i "$normalized_decoded" -pix_fmt yuv444p10le "${srgb[@]}" -y "$yuv_decoded"

"$(dirname "$0")"/../../../third_party/vmaf/libvmaf/build/tools/vmafossexec \
  yuv444p10le \
  "$(identify -format '%w' "$original")" "$(identify -format '%h' "$original")" \
  "$yuv_original" "$yuv_decoded" \
  ../../../third_party/vmaf/model/vmaf_v0.6.1.pkl \
  --log-fmt csv --log "$vmaf_csv"

read_csv="$(cat <<'END'
import csv
import sys
reader = csv.DictReader(sys.stdin)
for row in reader:
  print(row['vmaf'])
END
)"

python -c "$read_csv" < "$vmaf_csv" > "$output"
