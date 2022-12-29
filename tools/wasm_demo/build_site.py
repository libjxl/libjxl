#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import shutil
import subprocess
import sys

from pathlib import Path

EMBED_BIN = ['jxl_decoder.js', 'jxl_decoder.worker.js']
EMBED_SRC = ['client_worker.js']
TEMPLATES = ['service_worker.js']
COPY_BIN = ['jxl_decoder.wasm'] + EMBED_BIN
COPY_SRC = ['one_line_demo.html', 'one_line_demo_with_console.html', 'manual_decode_demo.html', 'netlify.toml', 'netlify'] + EMBED_SRC

COMPRESS = COPY_BIN + COPY_SRC + TEMPLATES
COMPRESSIBLE_EXT = ['.html', '.js', '.wasm']

BROTLIFY = True
ZOPFLIFY = False

def escape_js(js):
  return js.replace('\\', '\\\\').replace('\'', '\\\'')

def compress(path):
  name = path.name
  compressible = any([name.endswith(ext) for ext in COMPRESSIBLE_EXT])
  if not compressible:
    print(f'Not compressing {name}')
    return
  print(f'Compressing {name}')
  orig_size = path.stat().st_size
  if BROTLIFY:
    cmd_brotli = ['brotli', '-Zfk', path.absolute()]
    subprocess.run(cmd_brotli, check=True, stdout=sys.stdout, stderr=sys.stderr)
    br_size = path.parent.joinpath(name + '.br').stat().st_size
    print(f'  Brotli: {orig_size} -> {br_size}')
  if ZOPFLIFY:
    cmd_zopfli = ['zopfli', path.absolute()]
    subprocess.run(cmd_zopfli, check=True, stdout=sys.stdout, stderr=sys.stderr)
    gz_size = path.parent.joinpath(name + '.gz').stat().st_size
    print(f'  Zopfli: {orig_size} -> {gz_size}')

def uglify(text, name):
  cmd = ['uglifyjs', '-m', '-c']
  ugly_result = subprocess.run(
      cmd, capture_output=True, check=True, input=text, text=True)
  ugly_text = ugly_result.stdout.strip()
  print(f'Uglify {name}: {len(text)} -> {len(ugly_text)}')
  return ugly_text

if __name__ == "__main__":
  if len(sys.argv) != 4:
    print(f"Usage: python3 {sys.argv[0]} SRC_DIR BINARY_DIR OUTPUT_DIR")
    exit(-1)
  source_path = Path(sys.argv[1]) # CMake build dir
  binary_path = Path(sys.argv[2]) # Site template dir
  output_path = Path(sys.argv[3]) # Site output

  substitutes = {}

  for name in EMBED_BIN:
    key = '$' + name + '$'
    path = binary_path.joinpath(name)
    value = escape_js(uglify(path.read_text().strip(), name))
    substitutes[key] = value

  for name in EMBED_SRC:
    key = '$' + name + '$'
    path = source_path.joinpath(name)
    value = escape_js(uglify(path.read_text().strip(), name))
    substitutes[key] = value

  for name in TEMPLATES:
    print(f'Processing template {name}')
    path = source_path.joinpath(name)
    text = path.read_text().strip()
    for key, value in substitutes.items():
      text = text.replace(key, value)
    #text = uglify(text, name)
    output_path.joinpath(name).write_text(text)

  for name in COPY_SRC:
    path = source_path.joinpath(name)
    if path.is_dir():
      shutil.copytree(path, output_path.joinpath(
          name).absolute(), dirs_exist_ok=True)
    else:
      shutil.copy(path, output_path.absolute())

  # TODO: uglify
  for name in COPY_BIN:
    shutil.copy(binary_path.joinpath(name), output_path.absolute())

  for name in COMPRESS:
    compress(output_path.joinpath(name))
