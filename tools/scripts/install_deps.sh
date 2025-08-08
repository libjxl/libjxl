#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Script to install build / test deps for CI/CD environment.

# set -eux

ALL_ARGS=("$@")

# Handle options.
ARCH_SUFFIX=''
BUNDLES=()
SETUP_CROSS=false
PKGS=()

for arg in "${ALL_ARGS[@]}"; do
  if [[ "${arg}" == "--cross" ]]; then
    ARCH_SUFFIX=":${ARCH}"
    SETUP_CROSS=true
  else
    BUNDLES+=("${arg}")
  fi
done

##########
# Bundles
##########
BUNDLE_BENCHMARK=(libbenchmark-dev libbenchmark-tools)
BUNDLE_BUILD=(ccache ninja-build)
BUNDLE_CONFORMANCE=(python3-numpy qemu-user-static)
BUNDLE_DOCS=(doxygen graphviz)
BUNDLE_EXTRAS=(libgif-dev libjpeg-dev libpng-dev libwebp-dev libilmbase-dev libopenexr-dev)
BUNDLE_EXTRAS_RUNTIME=(libgif7 libjpeg-turbo8 libpng16-16t64 libwebp7 libopenexr-3-1-30)
BUNDLE_PLUGINS=(libgdk-pixbuf2.0-dev libgtk2.0-dev)
BUNDLE_RUNTIME=(libbrotli1)
# BUNDLE_PREINSTALLED=(clang cmake git pkg-config unzip xvfb)

# Handle bundles
EXTRAS=false
for arg in "${BUNDLES[@]}"; do
  if $EXTRAS ; then
    PKGS+=( ${arg} )
  elif [[ "${arg}" == "--" ]]; then
    EXTRAS=true
  elif [[ "${arg}" == "benchmark" ]]; then
    for dep in "${BUNDLE_BENCHMARK[@]}"; do PKGS+=( ${dep}${ARCH_SUFFIX} ); done
  elif [[ "${arg}" == "build" ]]; then
    for dep in "${BUNDLE_BUILD[@]}"; do PKGS+=( ${dep} ); done
    PKGS+=( libbrotli-dev${ARCH_SUFFIX} )
    if $SETUP_CROSS ; then
      PKGS+=(
        gcc-${BUILD_TARGET} binutils-${BUILD_TARGET}
        libc6-dev-${ARCH}-cross libstdc++-13-dev-${ARCH}-cross
        libstdc++-13-dev:${ARCH}
      )
    fi
  elif [[ "${arg}" == "conformance" ]]; then
    for dep in "${BUNDLE_CONFORMANCE[@]}"; do PKGS+=( ${dep} ); done
  elif [[ "${arg}" == "docs" ]]; then
    for dep in "${BUNDLE_DOCS[@]}"; do PKGS+=( ${dep} ); done
  elif [[ "${arg}" == "extras" ]]; then
    for dep in "${BUNDLE_EXTRAS[@]}"; do PKGS+=( ${dep}${ARCH_SUFFIX} ); done
  elif [[ "${arg}" == "extras_runtime" ]]; then
    for dep in "${BUNDLE_EXTRAS_RUNTIME[@]}"; do PKGS+=( ${dep}${ARCH_SUFFIX} ); done
  elif [[ "${arg}" == "plugins" ]]; then
    for dep in "${BUNDLE_PLUGINS[@]}"; do PKGS+=( ${dep}${ARCH_SUFFIX} ); done
    PKGS+=( xdg-utils )
  elif [[ "${arg}" == "runtime" ]]; then
    for dep in "${BUNDLE_RUNTIME[@]}"; do PKGS+=( ${dep}${ARCH_SUFFIX} ); done
  else
    echo "Unrecognized option/bundle '${arg}'"
    exit 1
  fi
done

echo "Selected packages: ${PKGS[@]}"

rm -f /var/lib/man-db/auto-update

if $SETUP_CROSS ; then
  dpkg --add-architecture "${ARCH}"
fi

DEBIAN_FRONTEND=noninteractive apt update -y
DEBIAN_FRONTEND=noninteractive apt install -y "${PKGS[@]}"
