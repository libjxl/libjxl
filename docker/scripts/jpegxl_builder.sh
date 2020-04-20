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

# Main entry point for all the Dockerfile for jpegxl-builder. This centralized
# file helps sharing code and configuration between Dockerfiles.

set -eux

MYDIR=$(dirname $(realpath "$0"))

# libjpeg-turbo.
JPEG_TURBO_RELEASE="2.0.4"
JPEG_TURBO_URL="https://github.com/libjpeg-turbo/libjpeg-turbo/archive/${JPEG_TURBO_RELEASE}.tar.gz"
JPEG_TURBO_SHA256="7777c3c19762940cff42b3ba4d7cd5c52d1671b39a79532050c85efb99079064"

# zlib (dependency of libpng)
ZLIB_RELEASE="1.2.11"
ZLIB_URL="https://www.zlib.net/zlib-${ZLIB_RELEASE}.tar.gz"
ZLIB_SHA256="c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1"
# The name in the .pc and the .dll generated don't match in zlib for Windows
# because they use different .dll names in Windows. We avoid that by defining
# UNIX=1. We also install all the .dll files to ${prefix}/lib instead of the
# default ${prefix}/bin.
ZLIB_FLAGS='-DUNIX=1 -DINSTALL_PKGCONFIG_DIR=/${CMAKE_INSTALL_PREFIX}/lib/pkgconfig -DINSTALL_BIN_DIR=/${CMAKE_INSTALL_PREFIX}/lib'

# libpng
LIBPNG_RELEASE="1.6.37"
LIBPNG_URL="https://download.sourceforge.net/libpng/libpng-${LIBPNG_RELEASE}.tar.gz"
LIBPNG_SHA256="daeb2620d829575513e35fecc83f0d3791a620b9b93d800b763542ece9390fb4"

# giflib
GIFLIB_RELEASE="5.2.1"
GIFLIB_URL="https://netcologne.dl.sourceforge.net/project/giflib/giflib-${GIFLIB_RELEASE}.tar.gz"
GIFLIB_SHA256="31da5562f44c5f15d63340a09a4fd62b48c45620cd302f77a6d9acf0077879bd"

# A patch needed to compile GIFLIB in mingw.
GIFLIB_PATCH_URL="https://github.com/msys2/MINGW-packages/raw/3afde38fcee7b3ba2cafd97d76cca8f06934504f/mingw-w64-giflib/001-mingw-build.patch"
GIFLIB_PATCH_SHA256="2b2262ddea87fc07be82e10aeb39eb699239f883c899aa18a16e4d4e40af8ec8"

# webp
WEBP_RELEASE="1.0.2"
WEBP_URL="https://codeload.github.com/webmproject/libwebp/tar.gz/v${WEBP_RELEASE}"
WEBP_SHA256="347cf85ddc3497832b5fa9eee62164a37b249c83adae0ba583093e039bf4881f"

# Temporary files cleanup hooks.
CLEANUP_FILES=()
cleanup() {
  if [[ ${#CLEANUP_FILES[@]} -ne 0 ]]; then
    rm -fr "${CLEANUP_FILES[@]}"
  fi
}
trap "{ set +x; } 2>/dev/null; cleanup" INT TERM EXIT

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
LIST_MINGW_TARGETS=(
  i686-w64-mingw32
  x86_64-w64-mingw32
)

# Setup the apt repositories and supported architectures.
setup_apt() {
  apt-get update -y
  apt-get install -y curl gnupg

  # gcc ppa sources.
  cat >/etc/apt/sources.list.d/gcc.list <<EOF
deb [arch=$(echo ${LIST_ARCHS[@]} | tr ' ' ,)] http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main
EOF
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F

  # node sources.
  cat >/etc/apt/sources.list.d/nodesource.list <<EOF
  deb https://deb.nodesource.com/node_13.x bionic main
  deb-src https://deb.nodesource.com/node_13.x bionic main
EOF
  curl -s https://deb.nodesource.com/gpgkey/nodesource.gpg.key | apt-key add -

  local port_list=()
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

    # For cross-compiling to Windows with mingw.
    mingw-w64
    wine64

    # Native tools.
    bsdmainutils
    cmake
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

    # For compiling giflib documentation.
    xmlto

    # Common libraries.
    libstdc++-8-dev

    # We don't use tcmalloc on archs other than amd64. This installs
    # libgoogle-perftools4:amd64.
    google-perftools

    # NodeJS for running WASM tests
    nodejs
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

      # For OpenEXR:
      libilmbase12:"${ubarch}"
      libopenexr22:"${ubarch}"

      # TCMalloc dependency
      libunwind-dev:"${ubarch}"

      # Cross-compiling tools per arch.
      libc6-dev-"${ubarch}"-cross
      libstdc++-8-dev-"${ubarch}"-cross
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

  # Install all the manual packages via "apt install" for the main arch. These
  # will be installed for other archs via manual download and unpack.
  DEBIAN_FRONTEND=noninteractive apt install -y \
    "${packages[@]}" "${UNPACK_PKGS[@]}"
}

# Install a library from the source code for multiple targets.
# Usage: install_from_source <tar_url> <sha256> <target> [<target...>]
install_from_source() {
  local package="$1"
  shift

  local url
  eval "url=\${${package}_URL}"
  local sha256
  eval "sha256=\${${package}_SHA256}"
  # Optional package flags
  local pkgflags
  eval "pkgflags=\${${package}_FLAGS:-}"

  local workdir=$(mktemp -d --suffix=_install)
  CLEANUP_FILES+=("${workdir}")

  local tarfile="${workdir}"/$(basename "${url}")
  curl -L --output "${tarfile}" "${url}"
  if ! echo "${sha256} ${tarfile}" | sha256sum -c --status -; then
    echo "SHA256 mismatch for ${url}: expected ${sha256} but found:"
    sha256sum "${tarfile}"
    exit 1
  fi

  local srcdir="${workdir}/source"
  mkdir -p "${srcdir}"
  tar -zxf "${tarfile}" -C "${srcdir}" --strip-components=1

  local target
  for target in "$@"; do
    echo "Installing ${package} for target ${target} from ${url}"

    local builddir="${workdir}/build-${target}"
    mkdir -p "${builddir}"
    local cmake_args=()

    local system_name="Linux"
    if [[ "${target}" == *mingw32 ]]; then
      system_name="Windows"
      # When compiling with clang, CMake doesn't detect that we are using mingw.
      cmake_args+=(
        -DMINGW=1
      )
      local windres=$(which ${target}-windres || true)
      if [[ -n "${windres}" ]]; then
        cmake_args+=(-DCMAKE_RC_COMPILER="${windres}")
      fi
    fi
    cmake_args+=(-DCMAKE_SYSTEM_NAME="${system_name}")

    local prefix="/usr"
    if [[ "${target}" != "x86_64-linux-gnu" ]]; then
      # Cross-compiling.
      prefix="/usr/${target}"
      cmake_args+=(
        -DCMAKE_C_COMPILER_TARGET="${target}"
        -DCMAKE_CXX_COMPILER_TARGET="${target}"
        -DCMAKE_SYSTEM_PROCESSOR="${target%%-*}"
      )
    fi

    if [[ -e "${srcdir}/CMakeLists.txt" ]]; then
      # Most pacakges use cmake for building which is easier to configure for
      # cross-compiling.
      (
        export CC=clang-6.0 CXX=clang++-6.0
        cmake -B"${builddir}" -H"${srcdir}" -G Ninja \
          -DCMAKE_INSTALL_PREFIX="${prefix}" \
          "${cmake_args[@]}" ${pkgflags}
        cmake --build "${builddir}"
        ninja -C "${builddir}" install
      )
    elif [[ "${package}" == "GIFLIB" ]]; then
      # GIFLIB doesn't yet have a cmake build system and the Makefile has
      # several problems so we need to fix them here. We are using a patch from
      # MSYS2 that already fixes the compilation for mingw. There is a pull
      # request in giflib for adding CMakeLists.txt so this might not be
      # needed in the future.
      local srcdir_tgt="${workdir}/source-${target}"
      mkdir -p "${srcdir_tgt}"
      tar -zxf "${tarfile}" -C "${srcdir_tgt}" --strip-components=1

      if [[ "${target}" == *mingw32 ]]; then
        local make_patch="${srcdir_tgt}/libgif.patch"
        curl -L "${GIFLIB_PATCH_URL}" -o "${make_patch}"
        echo "${GIFLIB_PATCH_SHA256} ${make_patch}" | sha256sum -c --status -
        patch "${srcdir_tgt}/Makefile" < "${make_patch}"
      fi
      (
        cd "${srcdir_tgt}"
        local giflib_make_flags=(
          CC=clang-6.0
          PREFIX="${prefix}"
          CFLAGS="-O2 --target=${target} -std=gnu99"
        )
        # giflib make dependencies are not properly set up so parallel building
        # doesn't work for everything.
        make -j$(nproc --all) libgif.a "${giflib_make_flags[@]}"
        make -j$(nproc --all) all "${giflib_make_flags[@]}"
        make install "${giflib_make_flags[@]}"
      )
    else
      echo "Don't know how to install ${package}"
      exit 1
    fi

  done
}

# Packages that are manually unpacked for each architecture.
UNPACK_PKGS=(
  libgif-dev
  libclang-common-6.0-dev

  # For OpenEXR:
  libilmbase-dev
  libopenexr-dev

  # TCMalloc
  libgoogle-perftools-dev
  libtcmalloc-minimal4
  libgoogle-perftools4
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

  # clang-6.0 doesn't find the libgcc version from mingw since it has two
  # version numbers and a suffix (see https://reviews.llvm.org/D45505). This
  # workaround fixes it for clang-6.0:
  local mingwtarget
  for mingwtarget in "${LIST_MINGW_TARGETS[@]}"; do
    local target_path="/usr/lib/gcc/${mingwtarget}"
    local gccver
    for gccver in $(find "${target_path}"/ -maxdepth 1 -mindepth 1 -type d \
                    -exec basename {} \;); do
      # Converts "a.b-mingw32" to "a.b.0-mingw32".
      local symlink_dest="${target_path}/${gccver/-/.0-}"
      [[ -e "${symlink_dest}" ]] || ln -s "${gccver}" "${symlink_dest}"
    done
  done

  # TODO: Add msan for the target when cross-compiling. This only installs it
  # for amd64.
  ./msan_install.sh

  # Build and install qemu user-linux targets.
  ./qemu_install.sh

  # Install emscripten SDK.
  ./emsdk_install.sh

  # Install some dependency libraries manually for the different targets.

  install_from_source JPEG_TURBO "${LIST_MINGW_TARGETS[@]}"
  install_from_source ZLIB "${LIST_MINGW_TARGETS[@]}"
  install_from_source LIBPNG "${LIST_MINGW_TARGETS[@]}"
  install_from_source GIFLIB "${LIST_MINGW_TARGETS[@]}"
  # webp in Ubuntu is relatively old so we install it from source for everybody.
  install_from_source WEBP "${LIST_TARGETS[@]}" "${LIST_MINGW_TARGETS[@]}"


  # TODO(eustas): remove after official v14 release (21 Apr 2020)
  # Nightly NodeJS (v14) install.
  local node_tar_xz="/tmp/node.tar.xz"
  local node_path="/opt/node-nightly"
  local node_version="v14.0.0-nightly2020033067d5c907d2"
  curl -s "https://nodejs.org/download/nightly/${node_version}/node-${node_version}-linux-x64.tar.xz" -o "${node_tar_xz}"
  mkdir -p "${node_path}"
  tar -xf "${node_tar_xz}" -C "${node_path}" --strip-components=1
  rm -f "${node_tar_xz}"

  # Cleanup.
  rm -rf /var/lib/apt/lists/*
}

main "$@"
