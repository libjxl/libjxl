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


set -eu

TMPDIR=$(mktemp -d)

cleanup() {
  rm -rf ${TMPDIR}
}

trap cleanup EXIT


CJXL=$(realpath $(dirname "$0"))/../build/tools/cjxl
DJXL=$(realpath $(dirname "$0"))/../build/tools/djxl

${CJXL} "$@" ${TMPDIR}/x.jxl &>/dev/null
S1=$(${DJXL} ${TMPDIR}/x.jxl --print_read_bytes -s 1 2>&1 | grep 'Decoded' | grep -o '[0-9]*')
S2=$(${DJXL} ${TMPDIR}/x.jxl --print_read_bytes -s 2 2>&1 | grep 'Decoded' | grep -o '[0-9]*')
S8=$(${DJXL} ${TMPDIR}/x.jxl --print_read_bytes -s 8 2>&1 | grep 'Decoded' | grep -o '[0-9]*')

echo "8x: $S8 2x: $S2 1x: $S1"
