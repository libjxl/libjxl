#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

self=$(realpath "$0")
mydir=$(dirname "${self}")

"${mydir}"/../../../build/tools/ssimulacra_main "$1" "$2" > "$3" 2>/dev/null
