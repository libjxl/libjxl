#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Regression test for jxl_from_tree's HiddenChannel bound.
#
# The "max 16" check used to compare the number of *characters* parsed against
# 16 instead of the parsed value, so:
#   - "HiddenChannel 100" (3 chars) was accepted and drove the tool into a
#     multi-hundred-megabyte hidden-channel allocation, and
#   - "HiddenChannel 00000000000000001" (17 chars, value 1) was rejected.
# The fix bounds the *value* to 16.

set -eu

build_dir="${1}"
tool="${build_dir}/tools/jxl_from_tree"
if [[ ! -x "${tool}" ]]; then
  echo "jxl_from_tree not found at ${tool}" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' INT TERM EXIT

# A one-pixel image keeps every accepted case trivial to encode.
write_tree() {
  local value="$1"
  cat >"${tmpdir}/tree.txt" <<EOF
${value}

- Set 0
EOF
}

# Over-limit value must be rejected (this used to be accepted -> OOM).
write_tree "HiddenChannel 100"
if "${tool}" "${tmpdir}/tree.txt" "${tmpdir}/out.jxl" >"${tmpdir}/log" 2>&1; then
  echo "HiddenChannel 100 should have been rejected, but the tool succeeded" >&2
  cat "${tmpdir}/log" >&2
  exit 1
fi
grep -q "Invalid HiddenChannel" "${tmpdir}/log" || {
  echo "expected an 'Invalid HiddenChannel' diagnostic for value 100" >&2
  cat "${tmpdir}/log" >&2
  exit 1
}

# A value <= 16 must be accepted even if it is spelled with leading zeros
# (this used to be rejected because the digit count exceeded 16).
write_tree "HiddenChannel 00000000000000001"
if ! "${tool}" "${tmpdir}/tree.txt" "${tmpdir}/out.jxl" >"${tmpdir}/log" 2>&1; then
  echo "HiddenChannel 00000000000000001 (value 1) should have been accepted" >&2
  cat "${tmpdir}/log" >&2
  exit 1
fi

# The exact limit is still allowed.
write_tree "HiddenChannel 16"
if ! "${tool}" "${tmpdir}/tree.txt" "${tmpdir}/out.jxl" >"${tmpdir}/log" 2>&1; then
  echo "HiddenChannel 16 (the documented maximum) should have been accepted" >&2
  cat "${tmpdir}/log" >&2
  exit 1
fi

echo "hidden_channel_test: OK"
