#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import sys

def find_key(entries : list[str], key: str) -> int:
  prefix = f"{key.lower()}: "
  for i in range(len(entries)):
    if entries[i].lower().startswith(prefix):
      return i
  return -1

def set_value(entries: list[str], key: str, value: str):
  new_line = f'{key}: {value}'
  # TODO(eustas): deal with repeated items
  idx = find_key(entries, key)
  if idx < 0:
    entries.append(new_line)
  else:
    entries[idx] = new_line

def transform_deb_822(archs):
  sources_path = "/etc/apt/sources.list.d/debian.sources"
  with open(sources_path) as f:
    lines = [line.rstrip() for line in f]
  lines.append('')
  entries = []
  entry = []
  for line in lines:
    if len(line) == 0:
      if len(entry) > 0:
        entries.append(entry)
      entry = []
    else:
      entry.append(line)

  new_entries = []
  for entry in entries:
    types_key = find_key(entry, "Types")
    if types_key < 0:
      continue
    if "types: deb" != entry[types_key].lower():
      continue
    deb_entry = entry[:]
    for arch in archs:
      deb_entry.append(f"Architectures-Add: {arch}")
    new_entries.append(deb_entry)
    deb_src_entry = deb_entry[:]
    set_value(deb_src_entry, "Types", "deb-src")
    new_entries.append(deb_src_entry)

  new_lines = []
  for entry in new_entries:
    if len(new_lines) > 0:
      new_lines.append("")
    new_lines.extend(entry)

  with open(sources_path, "w") as f:
    f.write('\n'.join(new_lines))

def main():
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[1]} ARCHS")
    sys.exit(1)
  archs_str = sys.argv[1]
  archs = archs_str.split(',')
  if True:
    transform_deb_822(archs)
  else:
    sys.exit(1)

if __name__ == '__main__':
  main()
