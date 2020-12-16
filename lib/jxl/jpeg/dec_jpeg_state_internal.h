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

#ifndef LIB_JXL_JPEG_DEC_JPEG_STATE_INTERNAL_H_
#define LIB_JXL_JPEG_DEC_JPEG_STATE_INTERNAL_H_

#include <brotli/decode.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/jpeg/brunsli_status.h"
#include "lib/jxl/jpeg/dec_jpeg_data_writer.h"
#include "lib/jxl/jpeg/dec_jpeg_serialization_state.h"
#include "lib/jxl/jpeg/dec_jpeg_state.h"
#include "lib/jxl/jpeg/enc_jpeg_huffman_decode.h"
#include "lib/jxl/jpeg/jpeg_lehmer_code.h"

namespace jxl {
namespace jpeg {

struct HuffmanDecodingData;

// Aid for section / subsection parsing.
struct SectionState {
  // Current value tag and type.
  size_t tag = 0;
  // True, if section is entered.
  bool is_active = false;
  // True, if "message" is actually "section", not a primitive value.
  bool is_section = false;

  // Encountered tags tracker.
  uint32_t tags_met = 0;

  // Remaining section length. Actual only when outside of workflow.
  size_t remaining = 0;

  // Position in current input, for which |remaining| was actual.
  size_t milestone = 0;

  // Projected section end, given enough input is provided.
  // |projected_end| == |milestone| + |provided|
  size_t projected_end = 0;
};

// Fields used for "Header" section parsing.
struct HeaderState {
  enum Stage {
    // Check section tag.
    READ_TAG,
    // Read section length.
    ENTER_SECTION,
    // Read value marker.
    ITEM_READ_TAG,
    // Read subsection length.
    ITEM_ENTER_SECTION,
    // Skip subsection payload.
    ITEM_SKIP_CONTENTS,
    // Read value.
    ITEM_READ_VALUE,
    // Verify values and apply to decoder state
    FINALE,
    // Finish section decoding.
    DONE
  };

  size_t stage = READ_TAG;

  // Subsection properties.
  SectionState section;
  // Length of subsection remaining to skip.
  size_t remaining_skip_length = 0;

  // Collected data (values).
  std::array<size_t, 16> varint_values;
};

// Fields used for section header parsing.
struct SectionHeaderState {
  enum Stage {
    // Check section tag.
    READ_TAG,
    // Read (dummy) value.
    READ_VALUE,
    // Read section length.
    ENTER_SECTION,
    // Finish section header decoding.
    DONE
  };

  size_t stage = READ_TAG;
};

enum class MetadataDecompressionStage {
  // Initial state in which it is decided which one of 3 processing variants to
  // use.
  INITIAL,
  // Read the length of uncompressed payload.
  READ_LENGTH,
  // Continuing as stream-decompressing/-parsing of Brotli-compressed metadata.
  DECOMPRESSING,
  // Parsing is finished, no further processing expected.
  DONE,
};

struct MetadataState {
  enum Stage {
    // Parse sequence type.
    READ_MARKER,
    // Dump the remaining of metadata to tail sequence.
    READ_TAIL,
    // Parse second byte of 2-byte sequence.
    READ_CODE,
    // Parse multi-byte sequence length.
    READ_LENGTH_HI,
    READ_LENGTH_LO,
    // Parse multi-byte sequence.
    READ_MULTIBYTE,
  };

  size_t short_marker_count = 0;
  uint8_t marker;
  uint8_t length_hi;
  size_t remaining_multibyte_length;
  std::vector<uint8_t>* multibyte_sink;
  size_t stage = READ_MARKER;

  BrotliDecoderStateStruct* brotli = nullptr;
  size_t metadata_size;
  size_t decompressed_size = 0;
  BrunsliStatus result = BRUNSLI_DECOMPRESSION_ERROR;
  MetadataDecompressionStage decompression_stage =
      MetadataDecompressionStage::INITIAL;

  ~MetadataState();

  bool CanFinish() { return (stage == READ_MARKER) || (stage == READ_TAIL); }
};

/**
 * Fits both DecodeVarint and DecodeLimitedVarint workflows.
 *
 * TODO(eustas): we could turn those methods back to stateless,
 * when "mark / rewind" utilities are added to BrunsliBitReader, and outer
 * parsing workflow supports input buffering.
 */
struct VarintState {
  enum Stage {
    INIT,
    READ_CONTINUATION,
    READ_DATA
  };

  Stage stage = INIT;
  size_t value;
  size_t i;
};

struct JpegInternalsState {
  enum Stage {
    INIT = 0,
    READ_MARKERS,
    READ_DRI,

    DECODE_HUFFMAN_MASK = 0x10,
    READ_HUFFMAN_LAST,
    READ_HUFFMAN_SIMPLE,
    READ_HUFFMAN_MAX_LEN,
    READ_HUFFMAN_COUNT,
    READ_HUFFMAN_PERMUTATION,
    HUFFMAN_UPDATE,

    PREPARE_READ_SCANS = 0x20,

    DECODE_SCAN_MASK = 0x40,
    READ_SCAN_COMMON,
    READ_SCAN_COMPONENT,
    READ_SCAN_RESET_POINT_CONTINUATION,
    READ_SCAN_RESET_POINT_DATA,
    READ_SCAN_ZERO_RUN_CONTINUATION,
    READ_SCAN_ZERO_RUN_DATA,

    READ_NUM_QUANT = 0x80,
    READ_QUANT,
    READ_COMP_ID_SCHEME,
    READ_COMP_ID,
    READ_NUM_PADDING_BITS,
    READ_PADDING_BITS,

    ITERATE_MARKERS,
    READ_INTERMARKER_LENGTH,
    READ_INTERMARKER_DATA,

    DONE
  };

  Stage stage = INIT;

  bool have_dri = false;
  size_t num_scans = 0;
  size_t dht_count = 0;

  jxl::BitReader br;
  size_t is_known_last_huffman_code;
  size_t terminal_huffman_code_count = 0;
  bool is_dc_table;
  size_t total_count;
  size_t space;
  size_t max_len;
  size_t max_count;
  size_t i;
  PermutationCoder p;
  VarintState varint;

  size_t j;
  int last_block_idx;
  int last_num;

  size_t num_padding_bits;
  size_t intermarker_length;
};

struct QuantDataState {
  enum Stage {
    INIT,

    READ_NUM_QUANT,

    READ_STOCK,
    READ_Q_FACTOR,
    READ_DIFF_IS_ZERO,
    READ_DIFF_SIGN,
    READ_DIFF,
    APPLY_DIFF,
    UPDATE,

    READ_QUANT_IDX,

    FINISH
  };

  Stage stage = INIT;

  jxl::BitReader br;
  size_t i;
  size_t j;
  uint8_t data_precision;
  VarintState vs;
  int delta;
  int sign;
  std::vector<uint8_t> predictor;
};

struct Buffer {
  size_t data_len = 0;
  size_t borrowed_len;
  std::vector<uint8_t> data;

  const uint8_t* external_data;
  size_t external_pos;
  size_t external_len;
};

struct InternalState {
  /* Parsing */

  SectionState section;

  // Sections.
  HeaderState header;
  SectionHeaderState section_header;
  MetadataState metadata;
  JpegInternalsState internals;
  QuantDataState quant;

  // "JPEGDecodingState" storage.
  std::vector<std::vector<uint8_t>> block_state_;

  BrunsliStatus result = BRUNSLI_OK;

  Stage last_stage = Stage::ERROR;

  Buffer buffer;

  /* Serialization */

  SerializationState serialization;
};

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_DEC_JPEG_STATE_INTERNAL_H_
