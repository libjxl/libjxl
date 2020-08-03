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

cat dct_gen.md \
    is_zero_base.md num_nonzeros_base.md brn_proto.md app0.md icc.md ducky.md \
    adobe.md stock_counts.md stock_values.md symbol_order.md stock_quant.md \
    quant.md freq_context.md num_nonzero_context.md nonzero_buckets.md \
    context_modes.md > all_tables.md
