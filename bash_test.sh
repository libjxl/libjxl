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

# Tests implemented in bash. These typically will run checks about the source
# code rather than the compiled one.

MYDIR=$(dirname $(realpath "$0"))

set -x
set -u

test_includes() {
  local ret=0
  local f
  for f in $(git ls-files | grep -E '(\.cc|\.cpp|\.h)$'); do
    # Check that the public file (in include/ directory) doesn't use the full
    # path to the public header since users of the library will include the
    # library as: #include "jpegxl/foobar.h".
    if [[ "${f#include/}" != "${f}" ]]; then
      if grep -i -H -n -E '#include\s*[<"]include/jpegxl' "$f" >&2; then
        echo "Don't add \"include/\" to the include path of public headers." >&2
        ret=1
      fi
    fi

    if [[ "${f#third_party/}" == "$f" ]]; then
      # $f is not in third_party/

      # Check that local files don't use the full path to third_party/
      # directory since the installed versions will not have that path.
      # Add an exception for third_party/dirent.h.
      if grep -v -F 'third_party/dirent.h' "$f" | \
          grep -i -H -n -E '#include\s*[<"]third_party/' >&2 &&
          [[ $ret -eq 0 ]]; then
        cat >&2 <<EOF
Don't add third_party/ to the include path of third_party projects. This makes
it harder to use installed system libraries instead of the third_party/ ones.
EOF
        ret=1
      fi
    fi

  done
  return ${ret}
}

main() {
  local ret=0
  cd "${MYDIR}"
  IFS=$'\n'
  for f in $(declare -F); do
    local test_name=$(echo "$f" | cut -f 3 -d ' ')
    # Runs all the local bash functions that start with "test_".
    if [[ "${test_name}" == test_* ]]; then
      echo "Test ${test_name}: Start"
      if ${test_name}; then
        echo "Test ${test_name}: PASS"
      else
        echo "Test ${test_name}: FAIL"
        ret=1
      fi
    fi
  done
  return ${ret}
}

main "$@"
