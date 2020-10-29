#!/usr/bin/env python3
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

import csv
import sys
import plotly.graph_objects as go

_, results, output_dir, *rest = sys.argv
OUTPUT = rest[0] if rest else 'svg'
# valid values: html, svg, png, webp, jpeg, pdf

with open(results, 'r') as f:
    reader = csv.DictReader(f)
    all_results = list(reader)

nonmetric_columns = set([
    "method", "image", "error", "size", "pixels", "enc_speed", "dec_speed",
    "bpp", "bppp", "qabpp"
])

column_remap = {
    'p': '6-Butteraugli',
    'dist': 'Max-Butteraugli',
    'psnr': "PSNR-YUV 6/8 Y"
}

metrics = set(all_results[0].keys()) - nonmetric_columns


def codec(method):
    sm = method.split(':')
    ssm = set(sm)
    speeds = set([
        'kitten', 'falcon', 'wombat', 'cheetah', 'tortoise', 'squirrel',
        'hare', 'fast'
    ])
    s = speeds.intersection(ssm)
    if sm[0] == 'custom':
        return sm[1]
    if sm[0] == 'jxl' and s:
        return 'jxl-' + list(s)[0]
    return sm[0]


data = {(m, img): {c: []
                   for c in {codec(x['method'])
                             for x in all_results}}
        for m in metrics for img in {x['image']
                                     for x in all_results}}

for r in all_results:
    c = codec(r['method'])
    img = r['image']
    bpp = r['bpp']
    for m in metrics:
        data[(m, img)][c].append((float(bpp), float(r[m])))

for (m, img) in data:
    fname = "%s/%s_%s" % (output_dir, m, img)
    fig = go.Figure()
    for method in sorted(data[(m, img)].keys()):
        vals = data[(m, img)][method]
        zvals = list(zip(*sorted(vals)))
        if not zvals:
            continue
        fig.add_trace(
            go.Scatter(x=zvals[0], y=zvals[1], mode='lines', name=method))
    fig.update_layout(title=img,
                      xaxis_title='bpp',
                      yaxis_title=column_remap.get(m, m))
    if OUTPUT == 'html':
        fig.write_html(fname + '.html')
    else:
        fig.write_image(fname + '.' + OUTPUT, scale=4)
