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

# Main entry point for all the Dockerfile for jpegxl-builder. This centralized
# file helps sharing code and configuration between Dockerfiles.

set -eux

MYDIR=$(dirname $(realpath "$0"))

# List of Ubuntu arch names supported by the builder (such as "i386").
LIST_ARCHS=(
  amd64
  i386
  arm64
  armhf
)

# List of target triplets supported by the builder.
LIST_TARGETS=(
  x86_64-linux-gnu
  i686-linux-gnu
  arm-linux-gnueabihf
  aarch64-linux-gnu
)

# Setup the apt repositories and supported architectures.
setup_apt() {
  apt-get update -y
  apt-get install -y gnupg

  # gcc ppa sources.
  cat >/etc/apt/sources.list.d/gcc.list <<EOF
deb [arch=$(echo ${LIST_ARCHS[@]} | tr ' ' ,)] http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main
EOF
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F

  local porl_list=()
  local main_list=()
  local ubarch
  for ubarch in "${LIST_ARCHS[@]}"; do
    if [[ "${ubarch}" != "amd64" && "${ubarch}" != "i386" ]]; then
      # other archs are not part of the main mirrors, but available in
      # ports.ubuntu.com.
      port_list+=("${ubarch}")
    else
      main_list+=("${ubarch}")
    fi
    # Add the arch to the system.
    if [[ "${ubarch}" != "amd64" ]]; then
      dpkg --add-architecture "${ubarch}"
    fi
  done

  # Update the sources.list with the split of supported architectures.
  local bkplist="/etc/apt/sources.list.bkp"
  [[ -e "${bkplist}" ]] || \
    mv /etc/apt/sources.list "${bkplist}"

  local newlist="/etc/apt/sources.list.tmp"
  rm -f "${newlist}"
  port_list=$(echo "${port_list[@]}" | tr ' ' ,)
  if [[ -n "${port_list}" ]]; then
    local port_url="http://ports.ubuntu.com/ubuntu-ports/"
    grep -v -E '^#' "${bkplist}" |
      sed -E "s;^deb (http[^ ]+) (.*)\$;deb [arch=${port_list}] ${port_url} \\2;" \
      >>"${newlist}"
  fi

  main_list=$(echo "${main_list[@]}" | tr ' ' ,)
  grep -v -E '^#' "${bkplist}" |
    sed -E "s;^deb (http[^ ]+) (.*)\$;deb [arch=${main_list}] \\1 \\2;" \
    >>"${newlist}"
  mv "${newlist}" /etc/apt/sources.list
}

install_pkgs() {
  packages=(
    # Native compilers
    clang-6.0 clang-format-6.0 clang-tidy-6.0

    # TODO: Consider adding clang-7 and clang-8 to every builder:
    #   clang-7 clang-format-7 clang-tidy-7
    #   clang-8 clang-format-8 clang-tidy-8

    # Native tools.
    bsdmainutils
    cmake
    curl
    extra-cmake-modules
    git
    llvm
    nasm
    ninja-build
    parallel
    pkg-config

    # These are used by the ./ci.sh lint in the native builder.
    clang-format-7
    clang-format-8

    # For coverage builds
    gcovr

    # Common libraries.
    libstdc++-8-dev
    libgif-dev  # See libgif-dev comment below.
  )

  # Install packages that are arch-dependent.
  local ubarch
  for ubarch in "${LIST_ARCHS[@]}"; do
    packages+=(
      # Library dependencies. These normally depend on the target architecture
      # we are compiling for and can't usually be installed for multiple
      # architectures at the same time.
      libjpeg-dev:"${ubarch}"
      libpng-dev:"${ubarch}"
      libqt5x11extras5-dev:"${ubarch}"

      libstdc++-8-dev:"${ubarch}"
      qtbase5-dev:"${ubarch}"

      # Cross-compiling tools per arch.
      libc6-dev-"${ubarch}"-cross
      libstdc++-8-dev-"${ubarch}"-cross

      # libgif-dev package can't be installed for multiple architectures at
      # the same time. Instead we unpack it manually later.
      # libgif-dev:"${ubarch}"
    )
  done

  local target
  for target in "${LIST_TARGETS[@]}"; do
    # Per target cross-compiling tools.
    if [[ "${target}" != "x86_64-linux-gnu" ]]; then
      packages+=(
        binutils-"${target}"
        gcc-"${target}"
      )
    fi
  done

  apt install -y "${packages[@]}"
}

# Packages that are manually unpacked for each architecture.
UNPACK_PKGS=(
  libgif-dev
  libclang-common-6.0-dev
)

# Main script entry point.
main() {
  cd "${MYDIR}"

  # Configure the repositories with the sources for multi-arch cross
  # compilation.
  setup_apt
  apt-get update -y
  apt-get dist-upgrade -y

  install_pkgs
  apt clean

  # Manually extract packages for the target arch that can't install it directly
  # at the same time as the native ones.
  local ubarch
  for ubarch in "${LIST_ARCHS[@]}"; do
    if [[ "${ubarch}" != "amd64" ]]; then
      local pkg
      for pkg in "${UNPACK_PKGS[@]}"; do
        apt download "${pkg}":"${ubarch}"
        dpkg -x "${pkg}"_*_"${ubarch}".deb /
      done
    fi
  done
  # TODO: Add clang from the llvm repos. This is problematic since we are
  # installing libclang-common-6.0-dev:"${ubarch}" from the ubuntu ports repos
  # which is not available in the llvm repos so it might have a different
  # version than the ubuntu ones.

  # Install webp manually since the version in Ubuntu is rather old.
  local target
  for target in "${LIST_TARGETS[@]}"; do
    if [[ "${target}" == "x86_64-linux-gnu" ]]; then
      CC=clang-6.0 CXX=clang++-6.0 ./webp_install.sh
    else
      CC=clang-6.0 CXX=clang++-6.0 ./webp_install.sh \
        -DCMAKE_INSTALL_PREFIX=/usr/"${target}" \
        -DCMAKE_C_COMPILER_TARGET="${target}" \
        -DCMAKE_CXX_COMPILER_TARGET="${target}" \
        -DCMAKE_SYSTEM_PROCESSOR="${target%%-*}" \
        -DCMAKE_SYSTEM_NAME=Linux
    fi
  done

  # TODO: Add msan for the target when cross-compiling. This only installs it
  # for amd64.
  ./msan_install.sh

  # Build and install qemu user-linux targets.
  ./qemu_install.sh

  # Install emscripten SDK.
  ./emsdk_install.sh

  # Cleanup.
  rm -rf /var/lib/apt/lists/*
}

main "$@"
