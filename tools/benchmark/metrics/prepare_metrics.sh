#!/bin/bash
set -eu

MYDIR=$(dirname $(realpath "$0"))


main() {
  cd "${MYDIR}/../../../third_party"
  rm -rf hdrvdp-2.2.2
  local zipfile="hdrvdp-2.2.2.zip"
  if [[ ! -e "${zipfile}" ]]; then
    wget -O "${zipfile}.tmp" https://sourceforge.net/projects/hdrvdp/files/hdrvdp/2.2.2/hdrvdp-2.2.2.zip
    mv "${zipfile}.tmp" "${zipfile}"
  fi
  unzip hdrvdp-2.2.2.zip


  pushd vmaf/libvmaf
  rm -rf build
  meson build --buildtype release
  ninja -vC build
  popd
}
main "$@"

