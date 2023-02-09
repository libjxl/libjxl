package(default_visibility = ['//:__subpackages__'])

load('lib.bzl',
    'libjxl_codec_apng_sources',
    #'libjxl_codec_exr_sources',
    'libjxl_codec_gif_sources',
    #'libjxl_codec_jpegli_sources',
    'libjxl_codec_jpg_sources',
    'libjxl_codec_jxl_sources',
    'libjxl_codec_npy_sources',
    'libjxl_codec_pgx_sources',
    'libjxl_codec_pnm_sources',
    'libjxl_dec_box_sources',
    'libjxl_dec_jpeg_sources',
    'libjxl_dec_sources',
    'libjxl_enc_sources',
    'libjxl_extras_for_tools_sources',
    'libjxl_extras_sources',
    #'libjxl_gbench_sources',
    'libjxl_major_version',
    'libjxl_minor_version',
    'libjxl_patch_version',
    #'libjxl_profiler_sources',
    'libjxl_public_headers',
    'libjxl_testlib_files',
    'libjxl_tests',
    'libjxl_threads_public_headers',
    'libjxl_threads_sources',
)

load('@bazel_skylib//rules:expand_template.bzl', 'expand_template')


EXPORT_TEMPLATE = '''
#ifndef @_EXPORT_H
#define @_EXPORT_H

#define @_EXPORT
#define @_NO_EXPORT

#ifndef @_DEPRECATED
#  define @_DEPRECATED __attribute__ ((__deprecated__))
#endif

#endif
'''

JXL_EXPORT_H = 'include/jxl/jxl_export.h'
genrule(
    name = 'create_jxl_export',
    outs = [JXL_EXPORT_H],
    cmd = 'echo \'' + EXPORT_TEMPLATE.replace('@', 'JXL') + '\' > $@',
)

JXL_THREADS_EXPORT_H = 'include/jxl/jxl_threads_export.h'
genrule(
    name = 'create_jxl_threads_export',
    outs = [JXL_THREADS_EXPORT_H],
    cmd = 'echo \'' + EXPORT_TEMPLATE.replace('@', 'JXL_THREADS') + '\' > $@',
)

JXL_VERSION_H = 'include/jxl/version.h'

# TODO(eustas): extract version from CMake file
expand_template(
    name = 'expand_jxl_version',
    template = 'jxl/version.h.in',
    out = JXL_VERSION_H,
    substitutions = {
        '@JPEGXL_MAJOR_VERSION@': str(libjxl_major_version),
        '@JPEGXL_MINOR_VERSION@': str(libjxl_minor_version),
        '@JPEGXL_PATCH_VERSION@': str(libjxl_patch_version),
    },
)

cc_library(
    name = 'jpegxl_version',
    hdrs = [JXL_VERSION_H],
    strip_include_prefix = 'include',
)

cc_library(
    name = 'jpegxl_private',
    srcs = libjxl_dec_sources + libjxl_dec_box_sources + libjxl_dec_jpeg_sources + libjxl_enc_sources,
    hdrs = libjxl_public_headers + [JXL_EXPORT_H],
    strip_include_prefix = 'include',
    defines = ['JPEGXL_ENABLE_SKCMS=1'],
    deps = [
        ':jpegxl_version',
        '@brotli//:brotlidec',
        '@brotli//:brotlienc',
        '@highway//:hwy',
        '@skcms//:skcms',
    ],
)

cc_library(
    name = 'jpegxl_threads',
    srcs = libjxl_threads_sources,
    hdrs = libjxl_threads_public_headers + [JXL_THREADS_EXPORT_H],
    strip_include_prefix = 'include',
    # TODO(eustas): used only for [memory_manager.h, profiler.h, parallel_runner.h]
    deps = [':jpegxl_private'],
)

CODEC_FILES = libjxl_codec_apng_sources + libjxl_codec_gif_sources + libjxl_codec_jpg_sources + libjxl_codec_jxl_sources + libjxl_codec_npy_sources + libjxl_codec_pgx_sources + libjxl_codec_pnm_sources
CODEC_SRCS = [path for path in CODEC_FILES if path.endswith('.cc')]
CODEC_HDRS = [path for path in CODEC_FILES if path.endswith('.h')]

# TODO(eustas): build codecs separately?
cc_library(
    name = 'jpegxl_extras',
    srcs = libjxl_extras_sources + libjxl_extras_for_tools_sources + CODEC_SRCS,
    hdrs = CODEC_HDRS,
    defines = [
        'JPEGXL_ENABLE_APNG=1',
        'JPEGXL_ENABLE_EXR=0', # TODO(eustas): add
        'JPEGXL_ENABLE_GIF=1',
        'JPEGXL_ENABLE_JPEG=1',
    ],
    deps = [
        ':jpegxl_private',
        ':jpegxl_threads',
        ':jpegxl_version',
        '@gif//:gif',
        '@libjpeg_turbo//:jpeg',
        '@png//:png',
    ],
)

cc_library(
    name = 'test_utils',
    testonly = 1,
    srcs = [src for src in libjxl_testlib_files if not src.endswith('.h')],
    hdrs = [src for src in libjxl_testlib_files if src.endswith('.h')],
    deps = [
        ':jpegxl_extras',
        ':jpegxl_private',
        '@bazel_tools//tools/cpp/runfiles',
    ],
)

TESTS = [src.partition('.')[0] for src in libjxl_tests]

# TODO(eustas): move to lib.bzl?
TEST_SHARDS = {
    'jxl/ans_test' : 2,
    'jxl/linalg_test' : 2,
    'jxl/modular_test' : 4,
    'jxl/roundtrip_test' : 4,
    'jxl/xorshift128plus_test' : 2,
    'jxl/ac_strategy_test' : 10, # TODO(eustas): separate heavy shard
    'jxl/dct_test' : 32,
    'jxl/decode_test' : 10, # TODO(eustas): separate heavy shard
    'jxl/fast_dct_test' : 8, # TODO(eustas): separate ultra-heavy shard
    'jxl/fast_math_test' : 10, # TODO(eustas): separate heavy shard
    'jxl/jxl_test' : 10, # TODO(eustas): separate heavy shard
    'jxl/render_pipeline/render_pipeline_test' : 10, # ? ->
}

# TODO(eustas): move to lib.bzl?
TEST_TIMEOUTS = {
    'jxl/fast_dct_test' : 'long',
    'jxl/dct_test' : 'long',
}

[
    cc_test(
        name = test,
        srcs = [test + '.cc'],
        data = ['//:testdata'],
        timeout = TEST_TIMEOUTS.get(test, 'moderate'),
        shard_count = TEST_SHARDS.get(test, 1),
        deps = [
            ':jpegxl_extras',
            ':jpegxl_private',
            ':jpegxl_threads',  # Some tests use the public threads API.
            ':test_utils',
            '@googletest//:gtest_main',
            '@highway//:hwy_test_util',
            '@highway//:nanobenchmark',
            '//tools:box',
        ],
    ) for test in TESTS
]
