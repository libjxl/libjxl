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

import lcms2


class ConformanceTestError(Exception):
    """General conformance test error."""


def CompareNPY(ref, ref_icc, dec, dec_icc, frame_idx, rmse, peak_error):
    """Compare a decoded numpy against the reference one."""
    if ref.shape != dec.shape:
        raise ConformanceTestError(
            f'Expected shape {ref.shape} but found {dec.shape}')
    ref_frame = ref[frame_idx]
    dec_frame = dec[frame_idx]
    num_channels = ref_frame.shape[2]

    if ref_icc != dec_icc:
        # Transform colors before comparison.
        if num_channels < 3:
            raise ConformanceTestError(f"Only RGB images are supported")
        ref_clr = ref_frame[:, :, 0:3]
        dec_clr = dec_frame[:, :, 0:3]
        dec_frame[:, :, 0:3] = lcms2.convert_pixels(dec_icc, ref_icc, dec_clr)

    error = numpy.abs(ref_frame - dec_frame)
    for ch in range(num_channels):
        error_ch = error[:, :, ch]
        actual_rmse = numpy.sqrt(numpy.mean(error_ch * error_ch))
        if actual_rmse > rmse:
            raise ConformanceTestError(
                f"RMSE too large: {actual_rmse} > {rmse}")

    actual_peak_error = error.max()
    if actual_peak_error > peak_error:
        raise ConformanceTestError(
            f"Peak error too large: {actual_peak_error} > {peak_error}")


def CompareBinaries(ref_bin, dec_bin):
    """Compare a decoded binary file against the reference for exact contents."""
    with open(ref_bin, 'rb') as reff:
        ref_data = reff.read()

    with open(dec_bin, 'rb') as decf:
        dec_data = decf.read()

    if ref_data != dec_data:
        raise ConformanceTestError(
            f'Binary files mismatch: {ref_bin} {dec_bin}')


TEST_KEYS = set(
    ['reconstructed_jpeg', 'original_icc', 'rms_error', 'peak_error'])


def CheckMeta(dec, ref):
    if isinstance(ref, dict):
        if not isinstance(dec, dict):
            raise ConformanceTestError("Malformed metadata file")
        for k, v in ref.items():
            if k in TEST_KEYS:
                continue
            if k not in dec:
                raise ConformanceTestError(
                    f"Malformed metadata file: key {k} not found")
            vv = dec[k]
            CheckMeta(vv, v)
    elif isinstance(ref, list):
        if not isinstance(dec, list) or len(dec) != len(ref):
            raise ConformanceTestError("Malformed metadata file")
        for vv, v in zip(dec, ref):
            CheckMeta(vv, v)
    elif isinstance(ref, float):
        if not isinstance(dec, float):
            raise ConformanceTestError("Malformed metadata file")
        if abs(dec - ref) > 0.0001:
            raise ConformanceTestError(
                f"Metadata: Expected {ref}, found {dec}")
    elif dec != ref:
        raise ConformanceTestError(f"Metadata: Expected {ref}, found {dec}")


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
                if 'sha256sums' in descriptor:
                    del descriptor['sha256sums']

            exact_tests = []

            with tempfile.TemporaryDirectory(prefix=test_id) as work_dir:
                cmd = [args.decoder, os.path.join(test_dir, 'input.jxl')]
                # Select the parameters to run.
                pixel_prefix = os.path.join(work_dir, 'decoded')
                cmd.extend(['-p', pixel_prefix])
                if 'reconstructed_jpeg' in descriptor:
                    jpeg_filename = os.path.join(work_dir, 'reconstructed.jpg')
                    cmd.extend(['-j', jpeg_filename])
                    exact_tests.append(('reconstructed.jpg', jpeg_filename))
                if 'original_icc' in descriptor:
                    decoded_original_icc = os.path.join(
                        work_dir, 'decoded_org.icc')
                    cmd.extend(['-i', decoded_original_icc])
                    exact_tests.append(('original.icc', decoded_original_icc))
                meta_filename = os.path.join(work_dir, 'meta.json')
                cmd.extend(['-m', meta_filename])

                if subprocess.call(cmd) != 0:
                    raise ConformanceTestError(
                        'Running the decoder (%s) returned error' %
                        ' '.join(cmd))

                # Run validation of exact files.
                for reference_basename, decoded_filename in exact_tests:
                    reference_filename = os.path.join(test_dir,
                                                      reference_basename)
                    CompareBinaries(reference_filename, decoded_filename)

                # Validate metadata.
                with open(meta_filename, 'r') as f:
                    meta = json.load(f)

                CheckMeta(meta, descriptor)

                # Pixel data.
                decoded_icc = pixel_prefix + '.icc'
                with open(decoded_icc, 'rb') as f:
                    decoded_icc = f.read()
                reference_icc = os.path.join(test_dir, "reference.icc")
                with open(reference_icc, 'rb') as f:
                    reference_icc = f.read()

                reference_npy = os.path.join(test_dir, 'reference_image.npy')
                decoded_npy = os.path.join(work_dir, 'decoded_image.npy')

                if not os.path.exists(decoded_npy):
                    raise ConformanceTestError(
                        'File not decoded: decoded_image.npy')

                reference_npy = numpy.load(reference_npy)
                decoded_npy = numpy.load(decoded_npy)

                for i, fd in enumerate(descriptor['frames']):
                    CompareNPY(reference_npy, reference_icc, decoded_npy,
                               decoded_icc, i, fd['rms_error'],
                               fd['peak_error'])

                if 'preview' in descriptor:
                    reference_npy = os.path.join(test_dir,
                                                 'reference_preview.npy')
                    decoded_npy = os.path.join(work_dir, 'decoded_preview.npy')

                    if not os.path.exists(decoded_npy):
                        raise ConformanceTestError(
                            'File not decoded: decoded_preview.npy')

                    reference_npy = numpy.load(reference_npy)
                    decoded_npy = numpy.load(decoded_npy)
                    CompareNPY(reference_npy, reference_icc, decoded_npy,
                               decoded_icc, 0,
                               descriptor['preview']['rms_error'],
                               descriptor['preview']['peak_error'])

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder',
                        metavar='DECODER',
                        required=True,
                        help='path to the decoder binary under test.')
    parser.add_argument(
        '--corpus',
        metavar='CORPUS',
        required=True,
        help=('path to the corpus directory or corpus descriptor'
              ' text file.'))
    args = parser.parse_args()
    if not ConformanceTestRunner(args):
        sys.exit(1)


if __name__ == '__main__':
    main()
