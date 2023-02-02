#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


"""build_cleaner.py: Update build files.

This tool keeps certain parts of the build files up to date.
"""

import argparse
import collections
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

def GetPrefixLibFiles(repo_files, prefix, suffixes=('.h', '.cc', '.ui')):
  """Gets the library files that start with the prefix and end with source
  code suffix."""
  prefix_files = [
      fn for fn in repo_files
      if fn.startswith(prefix) and any(fn.endswith(suf) for suf in suffixes)]
  return prefix_files

# Type holding the different types of sources in libjxl:
#   * decoder and common sources for minimal decoder,
#   * decoder and common sources,
#   * encoder-only sources,
#   * tests-only sources,
#   * google benchmark sources,
#   * threads library sources,
#   * extras library sources,
#   * libjxl (encoder+decoder) public include/ headers and
#   * threads public include/ headers.
JxlSources = collections.namedtuple(
    'JxlSources', ['dec_minimal', 'dec', 'enc', 'test',
                   'gbench', 'threads', 'extras', 'jxl_public_hdrs',
                   'threads_public_hdrs'])

def SplitLibFiles(repo_files):
  """Splits the library files into the different groups.

  """
  testonly = (
      'testdata.h', 'test_utils.h', 'test_image.h', '_test.h', '_test.cc',
      # _testonly.* files are library code used in tests only.
      '_testonly.h', '_testonly.cc'
  )
  main_srcs = GetPrefixLibFiles(repo_files, 'lib/jxl/')
  extras_srcs = GetPrefixLibFiles(repo_files, 'lib/extras/')
  test_srcs = [fn for fn in main_srcs
               if any(patt in fn for patt in testonly)]
  lib_srcs = [fn for fn in main_srcs
              if not any(patt in fn for patt in testonly)]

  # Google benchmark sources.
  gbench_srcs = sorted(fn for fn in lib_srcs + extras_srcs
                       if fn.endswith('_gbench.cc'))
  lib_srcs = [fn for fn in lib_srcs if fn not in gbench_srcs]
  # Exclude optional codecs from extras.
  exclude_extras = [
    '/dec/gif',
    '/dec/apng', '/enc/apng',
    '/dec/exr', '/enc/exr',
    '/dec/jpg', '/dec/jpegli',
    '/enc/jpg', '/enc/jpegli',
  ]
  extras_srcs = [fn for fn in extras_srcs if fn not in gbench_srcs and
                 not any(patt in fn for patt in testonly) and
                 not any(patt in fn for patt in exclude_extras)]

  enc_srcs = [fn for fn in lib_srcs
              if os.path.basename(fn).startswith('enc_') or
                 os.path.basename(fn).startswith('butteraugli')]
  enc_srcs.extend([
      "lib/jxl/encode.cc",
      "lib/jxl/encode_internal.h",
      # TODO(deymo): Add luminance.cc and luminance.h here too. Currently used
      # by aux_out.h.
  ])
  # Temporarily remove enc_bit_writer from the encoder sources: a lot of
  # decoder source code still needs to be split up into encoder and decoder.
  # Including the enc_bit_writer in the decoder allows to build a working
  # libjxl_dec library.
  # TODO(lode): remove the dependencies of the decoder on enc_bit_writer and
  # remove enc_bit_writer from the dec_srcs again.
  enc_srcs.remove("lib/jxl/enc_bit_writer.cc")
  enc_srcs.remove("lib/jxl/enc_bit_writer.h")
  enc_srcs.sort()

  enc_srcs_set = set(enc_srcs)
  lib_srcs = [fn for fn in lib_srcs if fn not in enc_srcs_set]

  # The remaining of the files are in the dec_library.
  dec_srcs = lib_srcs

  dec_opt_srcs = [
    "lib/jxl/box_content_decoder.cc",
    "lib/jxl/box_content_decoder.h",
    "lib/jxl/decode_to_jpeg.cc",
    "lib/jxl/decode_to_jpeg.h",
  ]
  dec_opt_srcs.extend([fn for fn in dec_srcs
                       if fn.startswith('lib/jxl/jpeg')])
  dec_opt_srcs_set = set(dec_opt_srcs)
  dec_minimal_srcs = [fn for fn in dec_srcs if fn not in dec_opt_srcs_set]

  thread_srcs = GetPrefixLibFiles(repo_files, 'lib/threads/')
  thread_srcs = [fn for fn in thread_srcs
                 if not any(patt in fn for patt in testonly)]
  public_hdrs = GetPrefixLibFiles(repo_files, 'lib/include/jxl/')

  threads_public_hdrs = [fn for fn in public_hdrs if '_parallel_runner' in fn]
  jxl_public_hdrs = list(sorted(set(public_hdrs) - set(threads_public_hdrs)))
  return JxlSources(dec_minimal_srcs, dec_srcs, enc_srcs, test_srcs,
                    gbench_srcs, thread_srcs, extras_srcs, jxl_public_hdrs,
                    threads_public_hdrs)


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

  # jxl version
  with open(os.path.join(args.src_dir, 'lib/CMakeLists.txt'), 'r') as f:
    cmake_text = f.read()

  gni_patterns = []
  for varname in ('JPEGXL_MAJOR_VERSION', 'JPEGXL_MINOR_VERSION',
                  'JPEGXL_PATCH_VERSION'):
    # Defined in CMakeLists.txt as "set(varname 1234)"
    match = re.search(r'set\(' + varname + r' ([0-9]+)\)', cmake_text)
    version_value = match.group(1)
    gni_patterns.append((r'"' + varname + r'=([0-9]+)"', version_value))

  jxl_src = SplitLibFiles(repo_files)

  # libjxl
  jxl_cmake_patterns = []
  jxl_cmake_patterns.append(
      (r'set\(JPEGXL_INTERNAL_SOURCES_DEC\n([^\)]+)\)',
       ''.join('  %s\n' % fn[len('lib/'):] for fn in jxl_src.dec_minimal)))
  jxl_cmake_patterns.append(
      (r'set\(JPEGXL_INTERNAL_SOURCES_ENC\n([^\)]+)\)',
       ''.join('  %s\n' % fn[len('lib/'):] for fn in jxl_src.enc)))
  ok = CleanFile(
      args, 'lib/jxl.cmake',
      jxl_cmake_patterns) and ok

  ok = CleanFile(
      args, 'lib/jxl_benchmark.cmake',
      [(r'set\(JPEGXL_INTERNAL_SOURCES_GBENCH\n([^\)]+)\)',
        ''.join('  %s\n' % fn[len('lib/'):] for fn in jxl_src.gbench))]) and ok

  gni_patterns.append((
      r'libjxl_dec_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):] for fn in jxl_src.dec)))
  gni_patterns.append((
      r'libjxl_enc_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):] for fn in jxl_src.enc)))
  gni_patterns.append((
      r'libjxl_gbench_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):] for fn in jxl_src.gbench)))


  tests = [fn[len('lib/'):] for fn in jxl_src.test if fn.endswith('_test.cc')]
  testlib = [fn[len('lib/'):] for fn in jxl_src.test
             if not fn.endswith('_test.cc')]
  gni_patterns.append((
      r'libjxl_tests_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn for fn in tests)))
  gni_patterns.append((
      r'libjxl_testlib_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn for fn in testlib)))

  # libjxl_threads
  ok = CleanFile(
      args, 'lib/jxl_threads.cmake',
      [(r'set\(JPEGXL_THREADS_SOURCES\n([^\)]+)\)',
        ''.join('  %s\n' % fn[len('lib/'):] for fn in jxl_src.threads))]) and ok

  gni_patterns.append((
      r'libjxl_threads_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):] for fn in jxl_src.threads)))

  # libjxl_extras
  ok = CleanFile(
      args, 'lib/jxl_extras.cmake',
      [(r'set\(JPEGXL_EXTRAS_SOURCES\n([^\)]+)\)',
        ''.join('  %s\n' % fn[len('lib/'):] for fn in jxl_src.extras))]) and ok

  gni_patterns.append((
      r'libjxl_extras_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):] for fn in jxl_src.extras)))

  # libjxl_profiler
  profiler_srcs = [fn[len('lib/'):] for fn in repo_files
                   if fn.startswith('lib/profiler')]
  ok = CleanFile(
      args, 'lib/jxl_profiler.cmake',
      [(r'set\(JPEGXL_PROFILER_SOURCES\n([^\)]+)\)',
        ''.join('  %s\n' % fn for fn in profiler_srcs))]) and ok

  gni_patterns.append((
      r'libjxl_profiler_sources = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn for fn in profiler_srcs)))

  # Public headers.
  gni_patterns.append((
      r'libjxl_public_headers = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):]
              for fn in jxl_src.jxl_public_hdrs)))
  gni_patterns.append((
      r'libjxl_threads_public_headers = \[\n([^\]]+)\]',
      ''.join('    "%s",\n' % fn[len('lib/'):]
              for fn in jxl_src.threads_public_hdrs)))


  # Update the list of tests. CMake version include test files in other libs,
  # not just in libjxl.
  tests = [fn[len('lib/'):] for fn in repo_files
           if fn.endswith('_test.cc') and fn.startswith('lib/')
           and not fn.startswith('lib/jpegli')]
  ok = CleanFile(
      args, 'lib/jxl_tests.cmake',
      [(r'set\(TEST_FILES\n([^\)]+)  ### Files before this line',
        ''.join('  %s\n' % fn for fn in tests))]) and ok
  ok = CleanFile(
      args, 'lib/jxl_tests.cmake',
      [(r'set\(TESTLIB_FILES\n([^\)]+)\)',
        ''.join('  %s\n' % fn for fn in testlib))]) and ok

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
