#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""JPEG XL conformance test runner.

Tool to perform a conformance test for a decoder.
"""

import argparse
import json
import numpy
import os
import subprocess
import sys
import tempfile


class ConformanceTestError(Exception):
  """General conformance test error."""

def CompareNPY(ref_npy, ref_icc, dec_npy, dec_icc, threshold=0.):
  """Compare a decoded numpy against the reference one."""
  ref = numpy.load(ref_npy)
  dec = numpy.load(dec_npy)
  if ref.shape != dec.shape:
    raise ConformanceTestError('Expected shape %s but found %s' %
                               (ref.shape, dec.shape))
  # TODO(deymo): Implement this comparison.


def CompareBinaries(ref_bin, dec_bin):
  """Compare a decoded binary file against the reference for exact contents."""
  with open(ref_bin, 'rb') as reff:
    ref_data = reff.read()

  with open(dec_bin, 'rb') as decf:
    dec_data = decf.read()

  if ref_data != dec_data:
    raise ConformanceTestError('Binary files mismatch: %s %s' %
                               (ref_bin, dec_bin))


def ConformanceTestRunner(args):
  # We can pass either the .txt file or the directory which defaults to the
  # full corpus. This is useful to run a subset of the corpus in other .txt
  # files.
  if os.path.isdir(args.corpus):
    corpus_dir = args.corpus
    corpus_txt = os.path.join(args.corpus, 'corpus.txt')
  else:
    corpus_dir = os.path.dirname(args.corpus)
    corpus_txt = args.corpus

  with open(corpus_txt, 'r') as f:
    for test_id in f:
      test_id = test_id.rstrip('\n')
      print('Testing %s' % test_id)
      test_dir = os.path.join(corpus_dir, test_id)

      with open(os.path.join(test_dir, 'test.json'), 'r') as f:
        descriptor = json.load(f)

      exact_tests = []

      with tempfile.TemporaryDirectory(prefix=test_id) as work_dir:
        cmd = [args.decoder, os.path.join(test_dir, descriptor['jxl'])]
        # Select the parameters to run.
        pixel_prefix = os.path.join(work_dir, 'dec')
        cmd.extend(['-p', pixel_prefix])
        # TODO(deymo): Add preview.
        if 'preview main' in descriptor:
          cmd.append('-w')
        if 'jpeg' in descriptor:
          jpeg_filename = os.path.join(work_dir, 'decoded.jpg')
          cmd.extend(['-j', jpeg_filename])
          exact_tests.append(('jpeg', jpeg_filename))
        if 'original_icc' in descriptor:
          decoded_original_icc = os.path.join(work_dir, 'decoded_org.icc')
          cmd.extend(['-i', decoded_original_icc])
          exact_tests.append(('original_icc', decoded_original_icc))

        if subprocess.call(cmd) != 0:
          raise ConformanceTestError(
              'Running the decoder (%s) returned error' % ' '.join(cmd))

        # Run validation of exact files.
        for key, decoded_filename in exact_tests:
          reference_filename = os.path.join(test_dir, descriptor[key])
          CompareBinaries(reference_filename, decoded_filename)

        # Pixel data.
        decoded_icc = pixel_prefix + '.icc'
        reference_icc = os.path.join(test_dir, descriptor['reference_icc'])

        if 'reference' in descriptor:
          reference_npy = os.path.join(test_dir, descriptor['reference'])
          decoded_npy = os.path.join(work_dir, 'dec.npy')
          if not os.path.exists(decoded_npy):
            raise ConformanceTestError('File not decoded: dec.npy')
          CompareNPY(reference_npy, reference_icc, decoded_npy, decoded_icc)

        # TODO(deymo): Add preview

  return True


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--decoder', metavar='DECODER', required=True,
                      help='path to the decoder binary under test.')
  parser.add_argument('--corpus', metavar='CORPUS', required=True,
                      help=('path to the corpus directory or corpus descriptor'
                            ' text file.'))
  args = parser.parse_args()
  if not ConformanceTestRunner(args):
    sys.exit(1)

if __name__ == '__main__':
  main()
