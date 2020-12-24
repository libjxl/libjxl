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


"""build_cleaner.py: Update build files.

This tool keeps certain parts of the build files up to date.
"""

import argparse
import locale
import os
import re
import subprocess
import sys
import tempfile


def RepoFiles(src_dir):
  """Return the list of files from the source git repository"""
  git_bin = os.environ.get('GIT_BIN', 'git')
  files = subprocess.check_output([git_bin, '-C', src_dir, 'ls-files'])
  ret = files.decode(locale.getpreferredencoding()).splitlines()
  ret.sort()
  return ret


def SplitLibFiles(repo_files, prefix, suffixes=('.h', '.cc', '.ui'),
                  testonly=('_test.cc',)):
  """Split the files that start with the prefix into sources and test files."""
  prefix_files = [
      fn for fn in repo_files
      if fn.startswith(prefix) and any(fn.endswith(suf) for suf in suffixes)]
  main_srcs = [fn for fn in prefix_files
               if not any(patt in fn for patt in testonly)]
  test_srcs = [fn for fn in prefix_files
               if any(patt in fn for patt in testonly)]
  return main_srcs, test_srcs


def CleanFile(args, filename, pattern_data_list):
  """Replace a pattern match with new data in the passed file.

  Given a regular expression pattern with a single () match, it runs the regex
  over the passed filename and replaces the match () with the new data. If
  args.update is set, it will update the file with the new contents, otherwise
  it will return True when no changes were needed.

  Multiple pairs of (regular expression, new data) can be passed to the
  pattern_data_list parameter and will be applied in order.

  The regular expression must match at least once in the file.
  """
  filepath = os.path.join(args.src_dir, filename)
  with open(filepath, 'r') as f:
    src_text = f.read()

  if not pattern_data_list:
    return True

  new_text = src_text

  for pattern, data in pattern_data_list:
    offset = 0
    chunks = []
    for match in re.finditer(pattern, new_text):
      chunks.append(new_text[offset:match.start(1)])
      offset = match.end(1)
      chunks.append(data)
    if not chunks:
      raise Exception('Pattern not found for %s: %r' % (filename, pattern))
    chunks.append(new_text[offset:])
    new_text = ''.join(chunks)

  if new_text == src_text:
    return True

  if args.update:
    print('Updating %s' % filename)
    with open(filepath, 'w') as f:
      f.write(new_text)
    return True
  else:
    with tempfile.NamedTemporaryFile(
        mode='w', prefix=os.path.basename(filename)) as new_file:
      new_file.write(new_text)
      new_file.flush()
      subprocess.call(
          ['diff', '-u', filepath, '--label', 'a/' + filename, new_file.name,
           '--label', 'b/' + filename])
    return False


def BuildCleaner(args):
  repo_files = RepoFiles(args.src_dir)
  ok = True

  gni_patterns = []

  # jxl version
  with open(os.path.join(args.src_dir, 'lib/CMakeLists.txt'), 'r') as f:
    cmake_text = f.read()

  for varname in ('JPEGXL_MAJOR_VERSION', 'JPEGXL_MINOR_VERSION',
                  'JPEGXL_PATCH_VERSION'):
    # Defined in CMakeLists.txt as "set(varname 1234)"
    match = re.search(r'set\(' + varname + r' ([0-9]+)\)', cmake_text)
    version_value = match.group(1)
    gni_patterns.append((r'"' + varname + r'=([0-9]+)"', version_value))

  # libjxl
  jxl_src, jxl_tests = SplitLibFiles(
      repo_files, 'lib/jxl/',
      testonly=('testdata.h', 'test_utils.h', '_test.h', '_test.cc'))

  ok = CleanFile(
      args, 'lib/jxl.cmake',
      [(r'set\(JPEGXL_INTERNAL_SOURCES\n([^\)]+)\)',
        ''.join('  %s\n' % fn[len('lib/'):] for fn in jxl_src))]) and ok

  gni_patterns.append((
      r'libjxl_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):] for fn in jxl_src)))
  gni_patterns.append((
      r'libjxl_tests_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):] for fn in jxl_tests
              if fn.endswith('_test.cc'))))

  # libjxl_threads
  threads_src, _ = SplitLibFiles(
      repo_files, 'lib/threads/')
  ok = CleanFile(
      args, 'lib/jxl_threads.cmake',
      [(r'set\(JPEGXL_THREADS_SOURCES\n([^\)]+)\)',
        ''.join('  %s\n' % fn[len('lib/'):] for fn in threads_src))]) and ok

  gni_patterns.append((
      r'libjxl_threads_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):] for fn in threads_src)))

  # Update the list of tests.
  tests = [fn[len('lib/'):] for fn in repo_files
           if fn.endswith('_test.cc') and fn.startswith('lib/')]
  ok = CleanFile(
      args, 'lib/jxl_tests.cmake',
      [(r'set\(TEST_FILES\n([^\)]+)  ### Files before this line',
        ''.join('  %s\n' % fn for fn in tests))]) and ok

  # Update lib.gni
  ok = CleanFile(args, 'lib/lib.gni', gni_patterns) and ok

  return ok


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--src-dir',
                      default=os.path.realpath(os.path.join(
                          os.path.dirname(__file__), '..')),
                      help='path to the build directory')
  parser.add_argument('--update', default=False, action='store_true',
                      help='update the build files instead of only checking')
  args = parser.parse_args()
  if not BuildCleaner(args):
    print('Build files need update.')
    sys.exit(2)


if __name__ == '__main__':
  main()
