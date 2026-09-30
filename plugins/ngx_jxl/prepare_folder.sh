#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Prepares a folder for serving either JPEG or JXL, by pre-compressing all the
# .jpg files in it.

# To configure nginx, prepare your server by adding in your http block
#
# map $http_accept $jxl_suffix { 
#   default "";
#   "~image/jxl" ".jxl";
# }
#
# and then, in every server block:
#
#  location ~ .jpg {
#   try_files $uri$jxl_suffix $uri;
# }



find $1 -name '*.jpg' | parallel cjxl '{}' '{}.jxl' --num_threads=0
