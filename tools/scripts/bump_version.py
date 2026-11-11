#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


"""bump_version.py: Update version in build / CI files."""
from pathlib import Path
import re
import sys


def bump_version(newver: str) -> None:
  parts = newver.split(".")
  if len(parts) != 3:
    raise ValueError(f"Expected MAJOR.MINOR.PATCH, got {newver}")
  [major, minor, patch] = parts
  print(f"Bumping version to {major}.{minor}.{patch}")

  cmake_path = Path("lib/CMakeLists.txt")
  cmake_text = cmake_path.read_text(encoding="utf-8")
  cmake_text = re.sub(r"(set\(JPEGXL_MAJOR_VERSION) [0-9]+\)", rf"\g<1> {major})", cmake_text)
  cmake_text = re.sub(r"(set\(JPEGXL_MINOR_VERSION) [0-9]+\)", rf"\g<1> {minor})", cmake_text)
  cmake_text = re.sub(r"(set\(JPEGXL_PATCH_VERSION) [0-9]+\)", rf"\g<1> {patch})", cmake_text)
  cmake_path.write_text(cmake_text, encoding="utf-8")

  workflow_path = Path(".github/workflows/conformance.yml")
  workflow_text = workflow_path.read_text(encoding="utf-8")
  workflow_text = re.sub(r'(LIBJXL_VERSION: )"[0-9.]+"', rf'\g<1>"{major}.{minor}.{patch}"', workflow_text)
  workflow_text = re.sub(r'(LIBJXL_ABI_VERSION: )"[0-9.]+"', rf'\g<1>"{major}.{minor}"', workflow_text)
  workflow_path.write_text(workflow_text, encoding="utf-8")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    raise RuntimeError(f"Usage: {sys.argv[0]} MAJOR.MINOR.PATCH")
  bump_version(sys.argv[1])
