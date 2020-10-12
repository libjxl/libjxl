#!/usr/bin/env python3
import csv
import sys
import plotly.graph_objects as go

OUTPUT = 'svg' if len(sys.argv) < 3 else sys.argv[2]
# valid values: html, svg, png, webp, jpeg, pdf

with open(sys.argv[1], 'r') as f:
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


data = dict(
    ((m, img),
     dict((c, [])
          for c in set(map(lambda x: codec(x['method']), all_results))))
    for m in metrics for img in set(map(lambda x: x['image'], all_results)))

for r in all_results:
    c = codec(r['method'])
    img = r['image']
    bpp = r['bpp']
    for m in metrics:
        data[(m, img)][c].append((float(bpp), float(r[m])))

for (m, img) in data:
    fname = "plots/%s_%s" % (m, img)
    fig = go.Figure()
    for method in sorted(data[(m, img)].keys()):
        vals = data[(m, img)][method]
        zvals = list(zip(*sorted(vals)))
        fig.add_trace(
            go.Scatter(x=zvals[0], y=zvals[1], mode='lines', name=method))
    fig.update_layout(title=img,
                      xaxis_title='bpp',
                      yaxis_title=column_remap.get(m, m))
    if OUTPUT == 'html':
        fig.write_html(fname + '.html')
    else:
        fig.write_image(fname + '.' + OUTPUT, scale=4)
