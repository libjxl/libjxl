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

WEBP_RELEASE="1.0.2"
WEBP_URL="https://codeload.github.com/webmproject/libwebp/tar.gz/v${WEBP_RELEASE}"

set -eu -x

# Temporary files cleanup hooks.
CLEANUP_FILES=()
cleanup() {
  if [[ ${#CLEANUP_FILES[@]} -ne 0 ]]; then
    rm -fr "${CLEANUP_FILES[@]}"
  fi
}
trap "{ set +x; } 2>/dev/null; cleanup" INT TERM EXIT

main() {
  local workdir=$(mktemp -d --suffix=webp)
  CLEANUP_FILES+=("${workdir}")

  local webptar="${workdir}/webp.tar.gz"
  curl --output "${webptar}" "${WEBP_URL}"
  tar -zxvf "${webptar}" -C "${workdir}"
  local srcdir="${workdir}/libwebp-${WEBP_RELEASE}"
  local builddir="${srcdir}/build"
  cmake -B"${builddir}" -H"${srcdir}" -G Ninja \
    -DCMAKE_INSTALL_PREFIX=/usr \
    "$@"
  cmake --build "${builddir}"
  ninja -C "${builddir}" install
}

main "$@"
