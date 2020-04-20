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

# Continuous integration helper module. This module is meant to be called from
# the .gitlab-ci.yml file during the continuous integration build, as well as
# from the command line for developers.

# This wrapper is used to enable WASM SIMD when running tests.
# Unfortunately, it is impossible to pass the option directly via the
# CMAKE_CROSSCOMPILING_EMULATOR variable.
# NODE should point to a capable binary; the one currently bundled with EMSDK
# is too old.

# Fallback to default node binary, if override is not set.
NODE="${NODE:-$(which node)}"

"${NODE}" --experimental-wasm-simd "$@"
