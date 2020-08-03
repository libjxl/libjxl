// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "jxl/enc_ans.h"

#include <stdint.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "c/common/constants.h"
#include "c/enc/context_map_encode.h"
#include "c/enc/write_bits.h"
#include "jxl/ans_common.h"
#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/bits.h"
#include "jxl/base/fast_log.h"
#include "jxl/dec_ans.h"
#include "jxl/enc_cluster.h"
#include "jxl/enc_context_map.h"
#include "jxl/fields.h"

namespace jxl {

namespace {

static const int kMaxNumSymbolsForSmallCode = 4;

void ANSBuildInfoTable(const ANSHistBin* counts, const AliasTable::Entry* table,
                       size_t alphabet_size, size_t log_alpha_size,
                       ANSEncSymbolInfo* info) {
  size_t log_entry_size = ANS_LOG_TAB_SIZE - log_alpha_size;
  size_t entry_size_minus_1 = (1 << log_entry_size) - 1;
  // create valid alias table for empty streams.
  for (size_t s = 0; s < std::max<size_t>(1, alphabet_size); ++s) {
    const ANSHistBin freq = s == alphabet_size ? ANS_TAB_SIZE : counts[s];
    info[s].freq_ = static_cast<uint16_t>(freq);
#ifdef USE_MULT_BY_RECIPROCAL
    if (freq != 0) {
      info[s].ifreq_ =
          ((1ull << RECIPROCAL_PRECISION) + info[s].freq_ - 1) / info[s].freq_;
    } else {
      info[s].ifreq_ = 1;  // shouldn't matter (symbol shouldn't occur), but...
    }
#endif
    info[s].reverse_map_.resize(freq);
  }
  for (int i = 0; i < ANS_TAB_SIZE; i++) {
    AliasTable::Symbol s =
        AliasTable::Lookup(table, i, log_entry_size, entry_size_minus_1);
    info[s.value].reverse_map_[s.offset] = i;
  }
}

float EstimateDataBits(const ANSHistBin* histogram, const ANSHistBin* counts,
                       size_t len) {
  float sum = 0.0f;
  int total_histogram = 0;
  int total_counts = 0;
  for (size_t i = 0; i < len; ++i) {
    total_histogram += histogram[i];
    total_counts += counts[i];
    if (histogram[i] > 0) {
      JXL_ASSERT(counts[i] > 0);
      // += histogram[i] * -log(counts[i]/total_counts)
      sum += histogram[i] *
             std::max(0.0f, ANS_LOG_TAB_SIZE - FastLog2f(counts[i]));
    }
  }
  if (total_histogram > 0) {
    JXL_ASSERT(total_counts == ANS_TAB_SIZE);
  }
  return sum;
}

float EstimateDataBitsFlat(const ANSHistBin* histogram, size_t len) {
  const float flat_bits = FastLog2f(len);
  int total_histogram = 0;
  for (size_t i = 0; i < len; ++i) {
    total_histogram += histogram[i];
  }
  return total_histogram * flat_bits;
}

// Static Huffman code for encoding logcounts. The last symbol is used as RLE
// sequence.
static const uint8_t kLogCountBitLengths[ANS_LOG_TAB_SIZE + 2] = {
    5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 6, 7, 7,
};
static const uint8_t kLogCountSymbols[ANS_LOG_TAB_SIZE + 2] = {
    17, 11, 15, 3, 9, 7, 4, 2, 5, 6, 0, 33, 1, 65,
};

// Returns the difference between largest count that can be represented and is
// smaller than "count" and smallest representable count larger than "count".
static int SmallestIncrement(uint32_t count, uint32_t shift) {
  int bits = count == 0 ? -1 : FloorLog2Nonzero(count);
  int drop_bits = bits - GetPopulationCountPrecision(bits, shift);
  return drop_bits < 0 ? 1 : (1 << drop_bits);
}

template <bool minimize_error_of_sum>
bool RebalanceHistogram(const float* targets, int max_symbol, int table_size,
                        uint32_t shift, int* omit_pos, ANSHistBin* counts) {
  int sum = 0;
  float sum_nonrounded = 0.0;
  int remainder_pos = 0;  // if all of them are handled in first loop
  int remainder_log = -1;
  for (int n = 0; n < max_symbol; ++n) {
    if (targets[n] > 0 && targets[n] < 1.0f) {
      counts[n] = 1;
      sum_nonrounded += targets[n];
      sum += counts[n];
    }
  }
  const float discount_ratio =
      (table_size - sum) / (table_size - sum_nonrounded);
  JXL_ASSERT(discount_ratio > 0);
  JXL_ASSERT(discount_ratio <= 1.0f);
  // Invariant for minimize_error_of_sum == true:
  // abs(sum - sum_nonrounded)
  //   <= SmallestIncrement(max(targets[])) + max_symbol
  for (int n = 0; n < max_symbol; ++n) {
    if (targets[n] >= 1.0f) {
      sum_nonrounded += targets[n];
      counts[n] =
          static_cast<ANSHistBin>(targets[n] * discount_ratio);  // truncate
      if (counts[n] == 0) counts[n] = 1;
      if (counts[n] == table_size) counts[n] = table_size - 1;
      // Round the count to the closest nonzero multiple of SmallestIncrement
      // (when minimize_error_of_sum is false) or one of two closest so as to
      // keep the sum as close as possible to sum_nonrounded.
      int inc = SmallestIncrement(counts[n], shift);
      counts[n] -= counts[n] & (inc - 1);
      // TODO(robryk): Should we rescale targets[n]?
      const float target =
          minimize_error_of_sum ? (sum_nonrounded - sum) : targets[n];
      if (counts[n] == 0 ||
          (target > counts[n] + inc / 2 && counts[n] + inc < table_size)) {
        counts[n] += inc;
      }
      sum += counts[n];
      const int count_log = FloorLog2Nonzero(static_cast<uint32_t>(counts[n]));
      if (count_log > remainder_log) {
        remainder_pos = n;
        remainder_log = count_log;
      }
    }
  }
  JXL_ASSERT(remainder_pos != -1);
  // NOTE: This is the only place where counts could go negative. We could
  // detect that, return false and make ANSHistBin uint32_t.
  counts[remainder_pos] -= sum - table_size;
  *omit_pos = remainder_pos;
  return counts[remainder_pos] > 0;
}

Status NormalizeCounts(ANSHistBin* counts, int* omit_pos, const int length,
                       const int precision_bits, uint32_t shift,
                       int* num_symbols, int* symbols) {
  const int32_t table_size = 1 << precision_bits;  // target sum / table size
  uint64_t total = 0;
  int max_symbol = 0;
  int symbol_count = 0;
  for (int n = 0; n < length; ++n) {
    total += counts[n];
    if (counts[n] > 0) {
      if (symbol_count < kMaxNumSymbolsForSmallCode) {
        symbols[symbol_count] = n;
      }
      ++symbol_count;
      max_symbol = n + 1;
    }
  }
  *num_symbols = symbol_count;
  if (symbol_count == 0) {
    return true;
  }
  if (symbol_count == 1) {
    counts[symbols[0]] = table_size;
    return true;
  }
  if (symbol_count > table_size)
    return JXL_FAILURE("Too many entries in an ANS histogram");

  const float norm = 1.f * table_size / total;
  std::vector<float> targets(max_symbol);
  for (size_t n = 0; n < targets.size(); ++n) {
    targets[n] = norm * counts[n];
  }
  if (!RebalanceHistogram<false>(&targets[0], max_symbol, table_size, shift,
                                 omit_pos, counts)) {
    // Use an alternative rebalancing mechanism if the one above failed
    // to create a histogram that is positive wherever the original one was.
    if (!RebalanceHistogram<true>(&targets[0], max_symbol, table_size, shift,
                                  omit_pos, counts)) {
      return JXL_FAILURE("Logic error: couldn't rebalance a histogram");
    }
  }
  return true;
}

struct SizeWriter {
  size_t size = 0;
  void Write(size_t num, size_t bits) { size += num; }
};

template <typename Writer>
void StoreVarLenUint8(size_t n, Writer* writer) {
  if (n == 0) {
    writer->Write(1, 0);
  } else {
    writer->Write(1, 1);
    size_t nbits = FloorLog2Nonzero(n);
    writer->Write(3, nbits);
    writer->Write(nbits, n - (1ULL << nbits));
  }
}

template <typename Writer>
void EncodeCounts(const ANSHistBin* counts, const int alphabet_size,
                  const int omit_pos, const int num_symbols, uint32_t shift,
                  const int* symbols, Writer* writer) {
  if (num_symbols <= 2) {
    // Small tree marker to encode 1-2 symbols.
    writer->Write(1, 1);
    if (num_symbols == 0) {
      writer->Write(1, 0);
      StoreVarLenUint8(0, writer);
    } else {
      writer->Write(1, num_symbols - 1);
      for (int i = 0; i < num_symbols; ++i) {
        StoreVarLenUint8(symbols[i], writer);
      }
    }
    if (num_symbols == 2) {
      writer->Write(ANS_LOG_TAB_SIZE, counts[symbols[0]]);
    }
  } else {
    // Mark non-small tree.
    writer->Write(1, 0);
    // Mark non-flat histogram.
    writer->Write(1, 0);

    // Precompute sequences for RLE encoding. Contains the number of identical
    // values starting at a given index. Only contains the value at the first
    // element of the series.
    std::vector<uint32_t> same(alphabet_size, 0);
    int last = 0;
    for (int i = 1; i < alphabet_size; i++) {
      // Store the sequence length once different symbol reached, or we're at
      // the end, or the length is longer than we can encode, or we are at
      // the omit_pos. We don't support including the omit_pos in an RLE
      // sequence because this value may use a different amount of log2 bits
      // than standard, it is too complex to handle in the decoder.
      if (counts[i] != counts[last] || i + 1 == alphabet_size ||
          (i - last) >= 255 || i == omit_pos || i == omit_pos + 1) {
        same[last] = (i - last);
        last = i + 1;
      }
    }

    int length = 0;
    std::vector<int> logcounts(alphabet_size);
    int omit_log = 0;
    for (int i = 0; i < alphabet_size; ++i) {
      JXL_ASSERT(counts[i] <= ANS_TAB_SIZE);
      JXL_ASSERT(counts[i] >= 0);
      if (i == omit_pos) {
        length = i + 1;
      } else if (counts[i] > 0) {
        logcounts[i] = FloorLog2Nonzero(static_cast<uint32_t>(counts[i])) + 1;
        length = i + 1;
        if (i < omit_pos) {
          omit_log = std::max(omit_log, logcounts[i] + 1);
        } else {
          omit_log = std::max(omit_log, logcounts[i]);
        }
      }
    }
    logcounts[omit_pos] = omit_log;

    // Elias gamma-like code for shift. Only difference is that if the number
    // of bits to be encoded is equal to FloorLog2(ANS_LOG_TAB_SIZE+1), we skip
    // the terminating 0 in unary coding.
    int upper_bound_log = FloorLog2Nonzero(ANS_LOG_TAB_SIZE + 1);
    int log = FloorLog2Nonzero(shift + 1);
    writer->Write(log, (1 << log) - 1);
    if (log != upper_bound_log) writer->Write(1, 0);
    writer->Write(log, ((1 << log) - 1) & (shift + 1));

    // Since num_symbols >= 3, we know that length >= 3, therefore we encode
    // length - 3.
    StoreVarLenUint8(length - 3, writer);

    // The logcount values are encoded with a static Huffman code.
    static const size_t kMinReps = 4;
    size_t rep = ANS_LOG_TAB_SIZE + 1;
    for (int i = 0; i < length; ++i) {
      if (i > 0 && same[i - 1] > kMinReps) {
        // Encode the RLE symbol and skip the repeated ones.
        writer->Write(kLogCountBitLengths[rep], kLogCountSymbols[rep]);
        StoreVarLenUint8(same[i - 1] - kMinReps - 1, writer);
        i += same[i - 1] - 2;
        continue;
      }
      writer->Write(kLogCountBitLengths[logcounts[i]],
                    kLogCountSymbols[logcounts[i]]);
    }
    for (int i = 0; i < length; ++i) {
      if (i > 0 && same[i - 1] > kMinReps) {
        // Skip symbols encoded by RLE.
        i += same[i - 1] - 2;
        continue;
      }
      if (logcounts[i] > 1 && i != omit_pos) {
        int bitcount = GetPopulationCountPrecision(logcounts[i] - 1, shift);
        int drop_bits = logcounts[i] - 1 - bitcount;
        JXL_CHECK((counts[i] & ((1 << drop_bits) - 1)) == 0);
        writer->Write(bitcount, (counts[i] >> drop_bits) - (1 << bitcount));
      }
    }
  }
}

void EncodeFlatHistogram(const int alphabet_size, BitWriter* writer) {
  // Mark non-small tree.
  writer->Write(1, 0);
  // Mark uniform histogram.
  writer->Write(1, 1);
  JXL_ASSERT(alphabet_size > 0);
  // Encode alphabet size.
  StoreVarLenUint8(alphabet_size - 1, writer);
}

float ComputeHistoAndDataCost(const ANSHistBin* histogram, size_t alphabet_size,
                              uint32_t method) {
  if (method == 0) {  // Flat code
    return ANS_LOG_TAB_SIZE + 2 +
           EstimateDataBitsFlat(histogram, alphabet_size);
  }
  // Non-flat: shift = method-1.
  uint32_t shift = method - 1;
  std::vector<ANSHistBin> counts(histogram, histogram + alphabet_size);
  int omit_pos = 0;
  int num_symbols;
  int symbols[kMaxNumSymbolsForSmallCode] = {};
  JXL_CHECK(NormalizeCounts(counts.data(), &omit_pos, alphabet_size,
                            ANS_LOG_TAB_SIZE, shift, &num_symbols, symbols));
  SizeWriter writer;
  EncodeCounts(counts.data(), alphabet_size, omit_pos, num_symbols, shift,
               symbols, &writer);
  return writer.size +
         EstimateDataBits(histogram, counts.data(), alphabet_size);
}

uint32_t ComputeBestMethod(const ANSHistBin* histogram, size_t alphabet_size,
                           float* cost, bool approximate = false) {
  size_t method = 0;
  float fcost = ComputeHistoAndDataCost(histogram, alphabet_size, 0);
  for (uint32_t shift = 0; shift <= ANS_LOG_TAB_SIZE;
       approximate ? shift += 2 : shift++) {
    float c = ComputeHistoAndDataCost(histogram, alphabet_size, shift + 1);
    if (c < fcost) {
      method = shift + 1;
      fcost = c;
    } else if (approximate) {
      // do not be as precise if estimating cost.
      break;
    }
  }
  *cost = fcost;
  return method;
}

}  // namespace

// Returns an estimate of the cost of encoding this histogram and the
// corresponding data.
size_t BuildAndStoreANSEncodingData(const ANSHistBin* histogram,
                                    size_t alphabet_size, size_t log_alpha_size,
                                    bool use_prefix_code,
                                    ANSEncSymbolInfo* info, BitWriter* writer) {
  if (use_prefix_code) {
    if (alphabet_size <= 1) return 0;
    uint32_t histo[ANS_MAX_ALPHA_SIZE];
    size_t total = 0;
    for (size_t i = 0; i < alphabet_size; i++) {
      histo[i] = histogram[i];
      JXL_CHECK(histogram[i] >= 0);
      total += histo[i];
    }
    uint8_t depths[ANS_MAX_ALPHA_SIZE] = {};
    uint16_t bits[ANS_MAX_ALPHA_SIZE];
    uint8_t storage[1024] = {};
    brunsli::Storage st(storage, 1024);
    brunsli::BuildAndStoreHuffmanTree(histo, alphabet_size, depths, bits, &st);
    for (size_t i = 0; i < alphabet_size; i++) {
      info[i].bits = depths[i] == 0 ? 0 : bits[i];
      info[i].depth = depths[i];
    }
    if (writer) {
      size_t pos = 0;
      while ((pos + 1) * 8 < st.pos) {
        writer->Write(8, storage[pos++]);
      }
      writer->Write(st.pos - pos * 8, storage[pos]);
    }
    // Estimate data cost.
    size_t cost = st.pos;
    for (size_t i = 0; i < alphabet_size; i++) {
      cost += histogram[i] * depths[i];
    }
    return cost;
  }
  JXL_ASSERT(alphabet_size <= ANS_TAB_SIZE);
  // Ensure we ignore trailing zeros in the histogram.
  if (alphabet_size != 0) {
    size_t largest_symbol = 0;
    for (size_t i = 0; i < alphabet_size; i++) {
      if (histogram[i] != 0) largest_symbol = i;
    }
    alphabet_size = largest_symbol + 1;
  }
  float cost;
  uint32_t method = ComputeBestMethod(histogram, alphabet_size, &cost);
  JXL_ASSERT(cost >= 0);
  int num_symbols;
  int symbols[kMaxNumSymbolsForSmallCode] = {};
  std::vector<ANSHistBin> counts(histogram, histogram + alphabet_size);
  if (!counts.empty()) {
    size_t sum = 0;
    for (size_t i = 0; i < counts.size(); i++) {
      sum += counts[i];
    }
    if (sum == 0) {
      counts[0] = ANS_TAB_SIZE;
    }
  }
  if (method == 0) {
    counts = CreateFlatHistogram(alphabet_size, ANS_TAB_SIZE);
    AliasTable::Entry a[ANS_MAX_ALPHA_SIZE];
    InitAliasTable(counts, ANS_TAB_SIZE, log_alpha_size, a);
    ANSBuildInfoTable(counts.data(), a, alphabet_size, log_alpha_size, info);
    if (writer != nullptr) {
      EncodeFlatHistogram(alphabet_size, writer);
    }
    return cost;
  }
  int omit_pos = 0;
  uint32_t shift = method - 1;
  JXL_CHECK(NormalizeCounts(counts.data(), &omit_pos, alphabet_size,
                            ANS_LOG_TAB_SIZE, shift, &num_symbols, symbols));
  AliasTable::Entry a[ANS_MAX_ALPHA_SIZE];
  InitAliasTable(counts, ANS_TAB_SIZE, log_alpha_size, a);
  ANSBuildInfoTable(counts.data(), a, alphabet_size, log_alpha_size, info);
  if (writer != nullptr) {
    EncodeCounts(counts.data(), alphabet_size, omit_pos, num_symbols, shift,
                 symbols, writer);
  }
  return cost;
}

float ANSPopulationCost(const ANSHistBin* data, size_t alphabet_size) {
  float c;
  ComputeBestMethod(data, alphabet_size, &c, /*approximate=*/true);
  return c;
}

template <typename Writer>
void EncodeUintConfig(const HybridUintConfig uint_config, Writer* writer,
                      size_t log_alpha_size) {
  writer->Write(CeilLog2Nonzero(log_alpha_size + 1),
                uint_config.split_exponent);
  if (uint_config.split_exponent == log_alpha_size) {
    return;  // msb/lsb don't matter.
  }
  size_t nbits = CeilLog2Nonzero(uint_config.split_exponent + 1);
  writer->Write(nbits, uint_config.msb_in_token);
  nbits = CeilLog2Nonzero(uint_config.split_exponent -
                          uint_config.msb_in_token + 1);
  writer->Write(nbits, uint_config.lsb_in_token);
}
template <typename Writer>
void EncodeUintConfigs(const std::vector<HybridUintConfig>& uint_config,
                       Writer* writer, size_t log_alpha_size) {
  // TODO(veluca): RLE?
  for (size_t i = 0; i < uint_config.size(); i++) {
    EncodeUintConfig(uint_config[i], writer, log_alpha_size);
  }
}
template void EncodeUintConfigs(const std::vector<HybridUintConfig>&,
                                BitWriter*, size_t);

namespace {

void ChooseUintConfigs(const HistogramParams& params,
                       const std::vector<std::vector<Token>>& tokens,
                       const std::vector<uint8_t>& context_map,
                       std::vector<Histogram>* clustered_histograms,
                       EntropyEncodingData* codes, size_t* log_alpha_size) {
  codes->uint_config.resize(clustered_histograms->size());
  if (params.uint_method == HistogramParams::HybridUintMethod::kNone) return;

  // Brute-force method that tries a few options.
  HybridUintConfig configs[] = {
      HybridUintConfig(4, 2, 0),  // default
      HybridUintConfig(4, 1, 0),  // less precise
      HybridUintConfig(4, 2, 1),  // add sign
      HybridUintConfig(4, 2, 2),  // add sign+parity
      HybridUintConfig(4, 1, 2),  // add parity but less msb
      // Same as above, but more direct coding.
      HybridUintConfig(5, 2, 0), HybridUintConfig(5, 1, 0),
      HybridUintConfig(5, 2, 1), HybridUintConfig(5, 2, 2),
      HybridUintConfig(5, 1, 2),
      // Same as above, but less direct coding.
      HybridUintConfig(3, 2, 0), HybridUintConfig(3, 1, 0),
      HybridUintConfig(3, 2, 1), HybridUintConfig(3, 1, 2),
      // For near-lossless.
      HybridUintConfig(4, 1, 3), HybridUintConfig(5, 1, 4),
      HybridUintConfig(5, 2, 3), HybridUintConfig(6, 1, 5),
      HybridUintConfig(6, 2, 4), HybridUintConfig(6, 0, 0),
      // Other
      HybridUintConfig(0, 0, 0),  // varlenuint
      HybridUintConfig(2, 0, 1),  // works well for ctx map
      HybridUintConfig(7, 0, 0),  // direct coding
  };

  std::vector<float> costs(clustered_histograms->size(),
                           std::numeric_limits<float>::max());
  std::vector<float> extra_bits(clustered_histograms->size());
  std::vector<uint8_t> is_valid(clustered_histograms->size());
  for (HybridUintConfig cfg : configs) {
    std::fill(is_valid.begin(), is_valid.end(), true);
    std::fill(extra_bits.begin(), extra_bits.end(), 0);

    for (size_t i = 0; i < clustered_histograms->size(); i++) {
      (*clustered_histograms)[i].Clear();
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
      for (size_t j = 0; j < tokens[i].size(); ++j) {
        const Token token = tokens[i][j];
        uint32_t tok, nbits, bits;
        size_t histo = context_map[token.context];
        cfg.Encode(token.value, &tok, &nbits, &bits);
        if (tok >= ANS_MAX_ALPHA_SIZE) {
          is_valid[histo] = false;
          continue;
        }
        extra_bits[histo] += nbits;
        (*clustered_histograms)[histo].Add(tok);
      }
    }

    for (size_t i = 0; i < clustered_histograms->size(); i++) {
      if (!is_valid[i]) continue;
      float cost = (*clustered_histograms)[i].PopulationCost() + extra_bits[i];
      if (cost < costs[i]) {
        codes->uint_config[i] = cfg;
        costs[i] = cost;
      }
    }
  }

  // Rebuild histograms.
  for (size_t i = 0; i < clustered_histograms->size(); i++) {
    (*clustered_histograms)[i].Clear();
  }
  *log_alpha_size = 4;
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t j = 0; j < tokens[i].size(); ++j) {
      const Token token = tokens[i][j];
      uint32_t tok, nbits, bits;
      size_t histo = context_map[token.context];
      codes->uint_config[histo].Encode(token.value, &tok, &nbits, &bits);
      (*clustered_histograms)[histo].Add(tok);
      while (tok >= (1 << *log_alpha_size)) (*log_alpha_size)++;
    }
  }
  JXL_ASSERT(*log_alpha_size <= 8);
}

class HistogramBuilder {
 public:
  explicit HistogramBuilder(const size_t num_contexts)
      : histograms_(num_contexts) {}

  void VisitSymbol(int symbol, size_t histo_idx) {
    JXL_ASSERT(histo_idx < histograms_.size());
    histograms_[histo_idx].Add(symbol);
  }

  // NOTE: `layer` is only for clustered_entropy; caller does ReclaimAndCharge.
  size_t BuildAndStoreEntropyCodes(
      const HistogramParams params,
      const std::vector<std::vector<Token>>& tokens, EntropyEncodingData* codes,
      std::vector<uint8_t>* context_map, bool use_prefix_code,
      const BitWriter::Allotment& allotment, BitWriter* writer, size_t layer,
      AuxOut* aux_out) const {
    size_t cost = 0;
    codes->encoding_info.clear();
    std::vector<Histogram> clustered_histograms(histograms_);
    context_map->resize(histograms_.size());
    if (histograms_.size() > 1) {
      std::vector<uint32_t> histogram_symbols;
      ClusterHistograms(params, histograms_, histograms_.size(), kClustersLimit,
                        &clustered_histograms, &histogram_symbols);
      for (size_t c = 0; c < histograms_.size(); ++c) {
        (*context_map)[c] = static_cast<uint8_t>(histogram_symbols[c]);
      }
      if (writer != nullptr) {
        EncodeContextMap(*context_map, clustered_histograms.size(), allotment,
                         writer);
      }
    }
    if (aux_out != nullptr) {
      for (size_t i = 0; i < clustered_histograms.size(); ++i) {
        aux_out->layers[layer].clustered_entropy +=
            clustered_histograms[i].ShannonEntropy();
      }
    }
    codes->use_prefix_code = use_prefix_code;
    size_t log_alpha_size = 7;  // Sane default.
    ChooseUintConfigs(params, tokens, *context_map, &clustered_histograms,
                      codes, &log_alpha_size);
    if (log_alpha_size < 5) log_alpha_size = 5;
    SizeWriter size_writer;  // Used if writer == nullptr to estimate costs.
    cost += 1;
    if (writer) writer->Write(1, use_prefix_code);

    if (use_prefix_code) {
      log_alpha_size = brunsli::kMaxHuffmanBits;
    } else {
      cost += 2;
    }
    if (writer == nullptr) {
      EncodeUintConfigs(codes->uint_config, &size_writer, log_alpha_size);
    } else {
      if (!use_prefix_code) writer->Write(2, log_alpha_size - 5);
      EncodeUintConfigs(codes->uint_config, writer, log_alpha_size);
    }
    if (use_prefix_code) {
      for (size_t c = 0; c < clustered_histograms.size(); ++c) {
        size_t num_symbol = 1;
        for (size_t i = 0; i < ANS_MAX_ALPHA_SIZE; i++) {
          if (clustered_histograms[c].data_[i]) num_symbol = i + 1;
        }
        if (writer) {
          StoreVarLenUint8(num_symbol - 1, writer);
        } else {
          StoreVarLenUint8(num_symbol - 1, &size_writer);
        }
      }
    }
    cost += size_writer.size;
    for (size_t c = 0; c < clustered_histograms.size(); ++c) {
      size_t num_symbol = 1;
      for (size_t i = 0; i < ANS_MAX_ALPHA_SIZE; i++) {
        if (clustered_histograms[c].data_[i]) num_symbol = i + 1;
      }
      codes->encoding_info.emplace_back();
      codes->encoding_info.back().resize(std::max<size_t>(1, num_symbol));

      cost += BuildAndStoreANSEncodingData(
          clustered_histograms[c].data_, num_symbol, log_alpha_size,
          use_prefix_code, codes->encoding_info.back().data(), writer);
    }
    return cost;
  }

 private:
  std::vector<Histogram> histograms_;
};
}  // namespace

size_t BuildAndEncodeHistograms(const HistogramParams& params,
                                size_t num_contexts,
                                const std::vector<std::vector<Token>>& tokens,
                                EntropyEncodingData* codes,
                                std::vector<uint8_t>* context_map,
                                BitWriter* writer, size_t layer,
                                AuxOut* aux_out) {
  LZ77Params lz77;
  size_t total_bits = 0;
  // TODO(veluca): figure out if lz77 should be used, and transform the token
  // streams.
  if (writer) {
    JXL_CHECK(Bundle::Write(lz77, writer, layer, aux_out));
  } else {
    size_t ebits, bits;
    JXL_CHECK(Bundle::CanEncode(lz77, &ebits, &bits));
    total_bits += bits;
  }
  if (lz77.enabled) {
    if (writer) {
      size_t b = writer->BitsWritten();
      EncodeUintConfig(lz77.length_uint_config, writer, /*log_alpha_size=*/7);
      total_bits += writer->BitsWritten() - b;
    } else {
      SizeWriter size_writer;
      EncodeUintConfig(lz77.length_uint_config, &size_writer,
                       /*log_alpha_size=*/7);
      total_bits += size_writer.size;
    }
  }
  size_t total_tokens = 0;
  // Build histograms.
  HistogramBuilder builder(num_contexts);
  HybridUintConfig uint_config;  //  Default config for clustering.
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t j = 0; j < tokens[i].size(); ++j) {
      const Token token = tokens[i][j];
      total_tokens++;
      uint32_t tok, nbits, bits;
      uint_config.Encode(token.value, &tok, &nbits, &bits);
      builder.VisitSymbol(tok, token.context);
    }
  }

  // TODO(veluca): better heuristics.
  bool use_prefix_code =
      total_tokens < 100 ||
      params.clustering == HistogramParams::ClusteringType::kFastest;

  // Encode histograms.
  const size_t max_contexts = std::min(num_contexts, kClustersLimit);
  BitWriter::Allotment allotment(writer, 8192 * (max_contexts + 4));
  total_bits += builder.BuildAndStoreEntropyCodes(
      params, tokens, codes, context_map, use_prefix_code, allotment, writer,
      layer, aux_out);
  allotment.FinishedHistogram(writer);
  ReclaimAndCharge(writer, &allotment, layer, aux_out);

  if (aux_out != nullptr) {
    aux_out->layers[layer].num_clustered_histograms +=
        codes->encoding_info.size();
  }
  return total_bits;
}

size_t WriteTokens(const std::vector<Token>& tokens,
                   const EntropyEncodingData& codes,
                   const std::vector<uint8_t>& context_map,
                   const BitWriter::Allotment& allotment, BitWriter* writer) {
  size_t num_extra_bits = 0;
  if (codes.use_prefix_code) {
    for (size_t i = 0; i < tokens.size(); i++) {
      uint32_t tok, nbits, bits;
      codes.uint_config[context_map[tokens[i].context]].Encode(
          tokens[i].value, &tok, &nbits, &bits);
      const uint8_t histo_idx = context_map[tokens[i].context];
      writer->Write(codes.encoding_info[histo_idx][tok].depth,
                    codes.encoding_info[histo_idx][tok].bits);
      writer->Write(nbits, bits);
      num_extra_bits += nbits;
    }
    return num_extra_bits;
  }
  std::vector<std::pair<uint32_t, uint32_t>> out;
  out.reserve(tokens.size());
  const int end = tokens.size();
  ANSCoder ans;
  for (int i = end - 1; i >= 0; --i) {
    const Token token = tokens[i];
    const uint8_t histo_idx = context_map[token.context];
    uint32_t tok, unused_nbits, unused_bits;
    codes.uint_config[context_map[tokens[i].context]].Encode(
        tokens[i].value, &tok, &unused_nbits, &unused_bits);
    const ANSEncSymbolInfo& info = codes.encoding_info[histo_idx][tok];
    uint8_t nbits = 0;
    uint32_t bits = ans.PutSymbol(info, &nbits);
    if (nbits == 16) {
      out.push_back(std::make_pair(i, bits));
    }
  }
  const uint32_t state = ans.GetState();
  writer->Write(32, state);
  int tokenidx = 0;
  for (int i = out.size(); i >= 0; --i) {
    int nextidx = i > 0 ? out[i - 1].first : end;
    for (; tokenidx < nextidx; ++tokenidx) {
      const Token token = tokens[tokenidx];
      uint32_t tok, nbits, bits;
      codes.uint_config[context_map[token.context]].Encode(token.value, &tok,
                                                           &nbits, &bits);
      writer->Write(nbits, bits);
      num_extra_bits += nbits;
    }
    if (i > 0) {
      writer->Write(16, out[i - 1].second);
    }
  }
  return num_extra_bits;
}

void WriteTokens(const std::vector<Token>& tokens,
                 const EntropyEncodingData& codes,
                 const std::vector<uint8_t>& context_map, BitWriter* writer,
                 size_t layer, AuxOut* aux_out) {
  BitWriter::Allotment allotment(writer, 32 * tokens.size() + 32 * 1024 * 4);
  size_t num_extra_bits =
      WriteTokens(tokens, codes, context_map, allotment, writer);
  ReclaimAndCharge(writer, &allotment, layer, aux_out);
  if (aux_out != nullptr) {
    aux_out->layers[layer].extra_bits += num_extra_bits;
  }
}

float TokenCost(const std::vector<Token>& tokens) {
  // TODO(veluca): implement this without using a writer.
  size_t num_contexts = 0;
  for (Token t : tokens) {
    if (num_contexts <= t.context) {
      num_contexts = t.context + 1;
    }
  }
  BitWriter writer;
  EntropyEncodingData codes;
  std::vector<uint8_t> context_map;
  BuildAndEncodeHistograms(HistogramParams(), num_contexts, {tokens}, &codes,
                           &context_map, &writer, 0, nullptr);
  WriteTokens(tokens, codes, context_map, &writer, 0, nullptr);
  return writer.BitsWritten();
}

}  // namespace jxl
