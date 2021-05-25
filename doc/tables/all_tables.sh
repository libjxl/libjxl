#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cat dct_gen.md \
    is_zero_base.md num_nonzeros_base.md brn_proto.md app0.md icc.md ducky.md \
    adobe.md stock_counts.md stock_values.md symbol_order.md stock_quant.md \
    quant.md freq_context.md num_nonzero_context.md nonzero_buckets.md \
    context_modes.md > all_tables.md
