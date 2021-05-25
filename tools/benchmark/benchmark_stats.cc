// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/benchmark/benchmark_stats.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cmath>

#include "lib/jxl/base/status.h"
#include "tools/benchmark/benchmark_args.h"

namespace jxl {
namespace {

// Computes longest codec name from Args()->codec, for table alignment.
uint32_t ComputeLargestCodecName() {
  std::vector<std::string> methods = SplitString(Args()->codec, ',');
  size_t max = strlen("Aggregate:");  // Include final row's name
  for (const auto& method : methods) {
    max = std::max(max, method.size());
  }
  return max;
}

// The benchmark result is a table of heterogeneous data, the column type
// specifies its data type. The type affects how it is printed as well as how
// aggregate values are computed.
enum ColumnType {
  // Formatted string
  TYPE_STRING,
  // Positive size, prints 0 as "---"
  TYPE_SIZE,
  // Floating point value (double precision) which is interpreted as
  // "not applicable" if <= 0, must be strictly positive to be valid but can be
  // set to 0 or negative to be printed as "---", for example for a speed that
  // is not measured.
  TYPE_POSITIVE_FLOAT,
  // Counts of some event
  TYPE_COUNT,
};

struct ColumnDescriptor {
  // Column name, printed across two lines.
  std::string label[2];
  // Total width to render the values of this column. If t his is a floating
  // point value, make sure this is large enough to contain a space and the
  // point, plus precision digits after the point, plus the max amount of
  // integer digits you expect in front of the point.
  uint32_t width;
  // Amount of digits after the point, or 0 if not a floating point value.
  uint32_t precision;
  ColumnType type;
  bool more;  // Whether to print only if more_columns is enabled
};

// To add or change a column to the benchmark ASCII table output, add/change
// an entry here with table header line 1, table header line 2, width of the
// column, precision after the point in case of floating point, and the
// data type. Then add/change the corresponding formula or formatting in
// the function ComputeColumns.
const std::vector<ColumnDescriptor>& GetColumnDescriptors() {
  // clang-format off
  static const std::vector<ColumnDescriptor> result = {
      {{"Compr", "Method"}, ComputeLargestCodecName() + 1, 0, TYPE_STRING, false},
      {{"Input", "Pixels"},         13,  0, TYPE_SIZE, false},
      {{"Compr", "Size"},            9,  0, TYPE_SIZE, false},
      {{"Compr", "BPP"},            17, 11, TYPE_POSITIVE_FLOAT, false},
      {{"", "#"},                    4,  0, TYPE_STRING, false},
      {{"Compr", "MP/s"},            8,  3, TYPE_POSITIVE_FLOAT, false},
      {{"Decomp", "MP/s"},           8,  3, TYPE_POSITIVE_FLOAT, false},
      {{"Butteraugli", "Distance"}, 13,  8, TYPE_POSITIVE_FLOAT, false},
      {{"", "Error p norm"},        16, 11, TYPE_POSITIVE_FLOAT, false},
      {{"", "PSNR"},                 7,  2, TYPE_POSITIVE_FLOAT, true},
      {{"", "QABPP"},                8,  3, TYPE_POSITIVE_FLOAT, true},
      {{"", "DCT2"},                 9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "DCT4"},                 9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "DCT4X8"},               9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "AFV"},                  9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "DCT8"},                 9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "DCT8X16"},              9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "DCT8X32"},              9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "DCT16"},                9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "DCT16X32"},             9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "DCT32"},                9,  7, TYPE_POSITIVE_FLOAT, true},
      {{"", "BPP*pnorm"},           20, 16, TYPE_POSITIVE_FLOAT, false},
      {{"", "Errors"},               9,  7, TYPE_COUNT, false},
  };
  // clang-format on

  return result;
}

// Computes throughput [megapixels/s] as reported in the report table
static double ComputeSpeed(size_t pixels, double time_s) {
  if (time_s == 0.0) return 0;
  return pixels * 1E-6 / time_s;
}

static std::string FormatFloat(const ColumnDescriptor& label, double value) {
  std::string result =
      StringPrintf("%*.*f", label.width - 1, label.precision, value);

  // Reduce precision if the value is too wide for the column. However, keep
  // at least one digit to the right of the point, and especially the integer
  // digits.
  if (result.size() >= label.width) {
    size_t point = result.rfind('.');
    if (point != std::string::npos) {
      int end = std::max<int>(point + 2, label.width - 1);
      result = result.substr(0, end);
    }
  }
  return result;
}

}  // namespace

std::string StringPrintf(const char* format, ...) {
  char buf[2000];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);
  return std::string(buf);
}

void BenchmarkStats::Assimilate(const BenchmarkStats& victim) {
  total_input_files += victim.total_input_files;
  total_input_pixels += victim.total_input_pixels;
  total_compressed_size += victim.total_compressed_size;
  total_adj_compressed_size += victim.total_adj_compressed_size;
  total_time_encode += victim.total_time_encode;
  total_time_decode += victim.total_time_decode;
  max_distance = std::max(max_distance, victim.max_distance);
  distance_p_norm += victim.distance_p_norm;
  distance_2 += victim.distance_2;
  distances.insert(distances.end(), victim.distances.begin(),
                   victim.distances.end());
  total_errors += victim.total_errors;
  jxl_stats.Assimilate(victim.jxl_stats);
}

void BenchmarkStats::PrintMoreStats() const {
  if (Args()->print_more_stats) {
    jxl_stats.Print();
  }
  if (Args()->print_distance_percentiles) {
    std::vector<float> sorted = distances;
    std::sort(sorted.begin(), sorted.end());
    int p50idx = 0.5 * distances.size();
    int p90idx = 0.9 * distances.size();
    printf("50th/90th percentile distance: %.8f  %.8f\n", sorted[p50idx],
           sorted[p90idx]);
  }
}

std::vector<ColumnValue> BenchmarkStats::ComputeColumns(
    const std::string& codec_desc, size_t corpus_size,
    size_t num_threads) const {
  JXL_CHECK(total_input_files == corpus_size);
  const double comp_bpp = total_compressed_size * 8.0 / total_input_pixels;
  const double adj_comp_bpp =
      total_adj_compressed_size * 8.0 / total_input_pixels;
  // Note: this is not affected by alpha nor bit depth.
  const double compression_speed =
      ComputeSpeed(total_input_pixels, total_time_encode);
  const double decompression_speed =
      ComputeSpeed(total_input_pixels, total_time_decode);
  // Already weighted, no need to divide by #channels.
  const double rmse = std::sqrt(distance_2 / total_input_pixels);
  const double psnr =
      total_compressed_size == 0
          ? 0.0
          : (distance_2 == 0) ? 99.99 : (20 * std::log10(1 / rmse));
  const double p_norm = distance_p_norm / total_input_pixels;
  const double bpp_p_norm = p_norm * comp_bpp;

  std::vector<ColumnValue> values(GetColumnDescriptors().size());

  values[0].s = codec_desc;
  values[1].i = total_input_pixels;
  values[2].i = total_compressed_size;
  values[3].f = comp_bpp;
  values[4].s = StringPrintf("%zu", num_threads);
  values[5].f = compression_speed;
  values[6].f = decompression_speed;
  values[7].f = static_cast<double>(max_distance);
  values[8].f = p_norm;
  values[9].f = psnr;
  values[10].f = adj_comp_bpp;
  // The DCT2, DCT4, AFV and DCT4X8 are applied to an 8x8 block by having 4x4
  // DCT2X2s, 2x2 DCT4x4s/AFVs, or 2x1 DCT4X8s, filling the whole 8x8 blocks.
  // Thus we need to multiply the block count by 8.0 * 8.0 pixels for these
  // transforms.
  values[11].f =
      jxl_stats.aux_out.num_dct2_blocks * 8.0 * 8.0 / total_input_pixels;
  values[12].f =
      jxl_stats.aux_out.num_dct4_blocks * 8.0 * 8.0 / total_input_pixels;
  values[13].f =
      jxl_stats.aux_out.num_dct4x8_blocks * 8.0 * 8.0 / total_input_pixels;
  values[14].f =
      jxl_stats.aux_out.num_afv_blocks * 8.0 * 8.0 / total_input_pixels;
  values[15].f =
      jxl_stats.aux_out.num_dct8_blocks * 8.0 * 8.0 / total_input_pixels;
  values[16].f =
      jxl_stats.aux_out.num_dct8x16_blocks * 8.0 * 16.0 / total_input_pixels;
  values[17].f =
      jxl_stats.aux_out.num_dct8x32_blocks * 8.0 * 32.0 / total_input_pixels;
  values[18].f =
      jxl_stats.aux_out.num_dct16_blocks * 16.0 * 16.0 / total_input_pixels;
  values[19].f =
      jxl_stats.aux_out.num_dct16x32_blocks * 16.0 * 32.0 / total_input_pixels;
  values[20].f =
      jxl_stats.aux_out.num_dct32_blocks * 32.0 * 32.0 / total_input_pixels;
  values[21].f = bpp_p_norm;
  values[22].i = total_errors;
  return values;
}

static std::string PrintFormattedEntries(
    const std::vector<ColumnValue>& values) {
  const auto& descriptors = GetColumnDescriptors();

  std::string out;
  for (size_t i = 0; i < descriptors.size(); i++) {
    if (!Args()->more_columns && descriptors[i].more) continue;
    std::string value;
    if (descriptors[i].type == TYPE_STRING) {
      value = values[i].s;
    } else if (descriptors[i].type == TYPE_SIZE) {
      value = values[i].i ? StringPrintf("%zd", values[i].i) : "---";
    } else if (descriptors[i].type == TYPE_POSITIVE_FLOAT) {
      value = FormatFloat(descriptors[i], values[i].f);
      value = FormatFloat(descriptors[i], values[i].f);
    } else if (descriptors[i].type == TYPE_COUNT) {
      value = StringPrintf("%zd", values[i].i);
    }

    int numspaces = descriptors[i].width - value.size();
    if (numspaces < 1) {
      numspaces = 1;
    }
    // All except the first one are right-aligned, the first one is the name,
    // others are numbers with digits matching from the right.
    if (i == 0) out += value.c_str();
    out += std::string(numspaces, ' ');
    if (i != 0) out += value.c_str();
  }
  return out + "\n";
}

std::string BenchmarkStats::PrintLine(const std::string& codec_desc,
                                      size_t corpus_size,
                                      size_t num_threads) const {
  std::vector<ColumnValue> values =
      ComputeColumns(codec_desc, corpus_size, num_threads);
  return PrintFormattedEntries(values);
}

std::string PrintHeader() {
  std::string out;
  const auto& descriptors = GetColumnDescriptors();
  for (int row = 0; row < 2; row++) {
    for (size_t i = 0; i < descriptors.size(); i++) {
      if (!Args()->more_columns && descriptors[i].more) continue;
      const std::string& label = descriptors[i].label[row];
      int numspaces = descriptors[i].width - label.size();
      // All except the first one are right-aligned.
      if (i == 0) out += label.c_str();
      out += std::string(numspaces, ' ');
      if (i != 0) out += label.c_str();
    }
    out += '\n';
  }
  for (const auto& descriptor : descriptors) {
    if (!Args()->more_columns && descriptor.more) continue;
    out += std::string(descriptor.width, '-');
  }
  return out + "\n";
}

std::string PrintAggregate(
    const std::vector<std::vector<ColumnValue>>& aggregate) {
  const auto& descriptors = GetColumnDescriptors();

  for (size_t i = 0; i < aggregate.size(); i++) {
    // Check when statistics has wrong amount of column entries
    JXL_CHECK(aggregate[i].size() == descriptors.size());
  }

  std::vector<ColumnValue> result(descriptors.size());

  // Statistics for the aggregate row are combined together with different
  // formulas than Assimilate uses for combining the statistics of files.
  for (size_t i = 0; i < descriptors.size(); i++) {
    if (descriptors[i].type == TYPE_STRING) {
      // "---" for the Iters column since this does not have meaning for
      // the aggregate stats.
      result[i].s = i == 0 ? "Aggregate:" : "---";
      continue;
    }
    if (descriptors[i].type == TYPE_COUNT) {
      size_t sum = 0;
      for (size_t j = 0; j < aggregate.size(); j++) {
        sum += aggregate[j][i].i;
      }
      result[i].i = sum;
      continue;
    }

    ColumnType type = descriptors[i].type;

    double logsum = 0;
    size_t numvalid = 0;
    for (size_t j = 0; j < aggregate.size(); j++) {
      double value =
          (type == TYPE_SIZE) ? aggregate[j][i].i : aggregate[j][i].f;
      if (value > 0) {
        numvalid++;
        logsum += std::log2(value);
      }
    }
    double geomean = numvalid ? std::exp2(logsum / numvalid) : 0.0;

    if (type == TYPE_SIZE || type == TYPE_COUNT) {
      result[i].i = static_cast<size_t>(geomean + 0.5);
    } else if (type == TYPE_POSITIVE_FLOAT) {
      result[i].f = geomean;
    } else {
      JXL_ABORT("unknown entry type");
    }
  }

  return PrintFormattedEntries(result);
}

}  // namespace jxl
