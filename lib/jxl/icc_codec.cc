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

#include "lib/jxl/icc_codec.h"

#include <stdint.h>

#include <map>
#include <string>
#include <vector>

#include "lib/jxl/aux_out.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/common.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/fields.h"

namespace jxl {
namespace {

bool EncodeVarInt(uint64_t value, size_t output_size, size_t* output_pos,
                  uint8_t* output) {
  // While more than 7 bits of data are left,
  // store 7 bits and set the next byte flag
  while (value > 127) {
    if (*output_pos > output_size) return false;
    // |128: Set the next byte flag
    output[(*output_pos)++] = ((uint8_t)(value & 127)) | 128;
    // Remove the seven bits we just wrote
    value >>= 7;
  }
  if (*output_pos > output_size) return false;
  output[(*output_pos)++] = ((uint8_t)value) & 127;
  return true;
}

void EncodeVarInt(uint64_t value, PaddedBytes* data) {
  size_t pos = data->size();
  data->resize(data->size() + 9);
  JXL_CHECK(EncodeVarInt(value, data->size(), &pos, data->data()));
  data->resize(pos);
}

size_t EncodeVarInt(uint64_t value, uint8_t* output) {
  size_t pos = 0;
  EncodeVarInt(value, 9, &pos, output);
  return pos;
}

uint64_t DecodeVarInt(const uint8_t* input, size_t inputSize, size_t* pos) {
  size_t i;
  uint64_t ret = 0;
  for (i = 0; *pos + i < inputSize && i < 10; ++i) {
    ret |= uint64_t(input[*pos + i] & 127) << uint64_t(7 * i);
    // If the next-byte flag is not set, stop
    if ((input[*pos + i] & 128) == 0) break;
  }
  // TODO: Return a decoding error if i == 10.
  *pos += i + 1;
  return ret;
}

uint32_t DecodeUint32(const uint8_t* data, size_t size, size_t pos) {
  return pos + 4 > size ? 0 : LoadBE32(data + pos);
}

void EncodeUint32(size_t pos, uint32_t value, PaddedBytes* data) {
  if (pos + 4 > data->size()) return;
  StoreBE32(value, data->data() + pos);
}

void AppendUint32(uint32_t value, PaddedBytes* data) {
  data->resize(data->size() + 4);
  EncodeUint32(data->size() - 4, value, data);
}

typedef std::array<uint8_t, 4> Tag;

Tag DecodeKeyword(const uint8_t* data, size_t size, size_t pos) {
  if (pos + 4 > size) return {' ', ' ', ' ', ' '};
  return {data[pos], data[pos + 1], data[pos + 2], data[pos + 3]};
}

void EncodeKeyword(const Tag& keyword, uint8_t* data, size_t size,
                   size_t pos) {
  if (keyword.size() != 4 || pos + 3 >= size) return;
  for (size_t i = 0; i < 4; ++i) data[pos + i] = keyword[i];
}

void AppendKeyword(const Tag& keyword, PaddedBytes* data) {
  JXL_ASSERT(keyword.size() == 4);
  data->append(keyword);
}

// Unshuffles or de-interleaves bytes, for example with width 2, turns
// "AaBbCcDc" into "ABCDabcd", this for example de-interleaves UTF-16 bytes into
// first all the high order bytes, then all the low order bytes.
// Transposes a matrix of width columns and ceil(size / width) rows. There are
// size elements, size may be < width * height, if so the
// last elements of the bottom row are missing, the missing spots are
// transposed along with the filled spots, and the result has the missing
// elements at the bottom of the rightmost column. The input is the input matrix
// in scanline order, the output is the result matrix in scanline order, with
// missing elements skipped over (this may occur at multiple positions).
void Unshuffle(uint8_t* data, size_t size, size_t width) {
  size_t height = (size + width - 1) / width;  // amount of rows of input
  PaddedBytes result(size);
  // i = input index, j output index
  size_t s = 0, j = 0;
  for (size_t i = 0; i < size; i++) {
    result[j] = data[i];
    j += height;
    if (j >= size) j = ++s;
  }

  for (size_t i = 0; i < size; i++) {
    data[i] = result[i];
  }
}

// Shuffles or interleaves bytes, for example with width 2, turns "ABCDabcd"
// into "AaBbCcDc". Transposes a matrix of ceil(size / width) columns and
// width rows. There are size elements, size may be < width * height, if so the
// last elements of the rightmost column are missing, the missing spots are
// transposed along with the filled spots, and the result has the missing
// elements at the end of the bottom row. The input is the input matrix in
// scanline order but with missing elements skipped (which may occur in multiple
// locations), the output is the result matrix in scanline order (with
// no need to skip missing elements as they are past the end of the data).
void Shuffle(uint8_t* data, size_t size, size_t width) {
  size_t height = (size + width - 1) / width;  // amount of rows of output
  PaddedBytes result(size);
  // i = output index, j input index
  size_t s = 0, j = 0;
  for (size_t i = 0; i < size; i++) {
    result[i] = data[j];
    j += height;
    if (j >= size) j = ++s;
  }

  for (size_t i = 0; i < size; i++) {
    data[i] = result[i];
  }
}

static constexpr size_t kICCHeaderSize = 128;

static const Tag kAcspTag = {'a', 'c', 's', 'p'};
static const Tag kBkptTag = {'b', 'k', 'p', 't'};
static const Tag kBtrcTag = {'b', 'T', 'R', 'C'};
static const Tag kBxyzTag = {'b', 'X', 'Y', 'Z'};
static const Tag kChadTag = {'c', 'h', 'a', 'd'};
static const Tag kChrmTag = {'c', 'h', 'r', 'm'};
static const Tag kCprtTag = {'c', 'p', 'r', 't'};
static const Tag kCurvTag = {'c', 'u', 'r', 'v'};
static const Tag kDescTag = {'d', 'e', 's', 'c'};
static const Tag kDmddTag = {'d', 'm', 'd', 'd'};
static const Tag kDmndTag = {'d', 'm', 'n', 'd'};
static const Tag kGbd_Tag = {'g', 'b', 'd', ' '};
static const Tag kGtrcTag = {'g', 'T', 'R', 'C'};
static const Tag kGxyzTag = {'g', 'X', 'Y', 'Z'};
static const Tag kKtrcTag = {'k', 'T', 'R', 'C'};
static const Tag kKxyzTag = {'k', 'X', 'Y', 'Z'};
static const Tag kLumiTag = {'l', 'u', 'm', 'i'};
static const Tag kMab_Tag = {'m', 'A', 'B', ' '};
static const Tag kMba_Tag = {'m', 'B', 'A', ' '};
static const Tag kMlucTag = {'m', 'l', 'u', 'c'};
static const Tag kMntrTag = {'m', 'n', 't', 'r'};
static const Tag kParaTag = {'p', 'a', 'r', 'a'};
static const Tag kRgb_Tag = {'R', 'G', 'B', ' '};
static const Tag kRtrcTag = {'r', 'T', 'R', 'C'};
static const Tag kRxyzTag = {'r', 'X', 'Y', 'Z'};
static const Tag kSf32Tag = {'s', 'f', '3', '2'};
static const Tag kTextTag = {'t', 'e', 'x', 't'};
static const Tag kVcgtTag = {'v', 'c', 'g', 't'};
static const Tag kWtptTag = {'w', 't', 'p', 't'};
static const Tag kXyz_Tag = {'X', 'Y', 'Z', ' '};

// Tag names focused on RGB and GRAY monitor profiles
constexpr const size_t kNumTagStrings = 17;
static const Tag* kTagStrings[kNumTagStrings] = {
    &kCprtTag, &kWtptTag, &kBkptTag, &kRxyzTag, &kGxyzTag, &kBxyzTag,
    &kKxyzTag, &kRtrcTag, &kGtrcTag, &kBtrcTag, &kKtrcTag, &kChadTag,
    &kDescTag, &kChrmTag, &kDmndTag, &kDmddTag, &kLumiTag};

static constexpr size_t kCommandTagUnknown = 1;
static constexpr size_t kCommandTagTRC = 2;
static constexpr size_t kCommandTagXYZ = 3;
static constexpr size_t kCommandTagStringFirst = 4;

// Tag types focused on RGB and GRAY monitor profiles
constexpr const size_t kNumTypeStrings = 8;
static const Tag* kTypeStrings[kNumTypeStrings] = {
    &kXyz_Tag, &kDescTag, &kTextTag, &kMlucTag,
    &kParaTag, &kCurvTag, &kSf32Tag, &kGbd_Tag};

static constexpr size_t kCommandInsert = 1;
static constexpr size_t kCommandShuffle2 = 2;
static constexpr size_t kCommandShuffle4 = 3;
static constexpr size_t kCommandPredict = 4;
static constexpr size_t kCommandXYZ = 10;
static constexpr size_t kCommandTypeStartFirst = 16;

static constexpr size_t kFlagBitOffset = 64;
static constexpr size_t kFlagBitSize = 128;

PaddedBytes InitHeaderPred() {
  PaddedBytes result(kICCHeaderSize);
  for (size_t i = 0; i < kICCHeaderSize; i++) {
    result[i] = 0;
  }
  result[8] = 4;
  EncodeKeyword(kMntrTag, result.data(), result.size(), 12);
  EncodeKeyword(kRgb_Tag, result.data(), result.size(), 16);
  EncodeKeyword(kXyz_Tag, result.data(), result.size(), 20);
  EncodeKeyword(kAcspTag, result.data(), result.size(), 36);
  result[68] = 0;
  result[69] = 0;
  result[70] = 246;
  result[71] = 214;
  result[72] = 0;
  result[73] = 1;
  result[74] = 0;
  result[75] = 0;
  result[76] = 0;
  result[77] = 0;
  result[78] = 211;
  result[79] = 45;
  return result;
}

void PredictHeader(const uint8_t* icc, size_t size, uint8_t* header,
                   size_t pos) {
  if (pos == 8 && size >= 8) {
    header[80] = icc[4];
    header[81] = icc[5];
    header[82] = icc[6];
    header[83] = icc[7];
  }
  if (pos == 41 && size >= 41) {
    if (icc[40] == 'A') {
      header[41] = 'P';
      header[42] = 'P';
      header[43] = 'L';
    }
    if (icc[40] == 'M') {
      header[41] = 'S';
      header[42] = 'F';
      header[43] = 'T';
    }
  }
  if (pos == 42 && size >= 42) {
    if (icc[40] == 'S' && icc[41] == 'G') {
      header[42] = 'I';
      header[43] = ' ';
    }
    if (icc[40] == 'S' && icc[41] == 'U') {
      header[42] = 'N';
      header[43] = 'W';
    }
  }
}

template <typename T>
T PredictValue(T p1, T p2, T p3, int order) {
  if (order == 0) return p1;
  if (order == 1) return 2 * p1 - p2;
  if (order == 2) return 3 * p1 - 3 * p2 + p3;
  return 0;
}

// Predicts a value with linear prediction of given order (0-2), for integers
// with width bytes and given stride in bytes between values.
// The start position is at start + i, and the relevant modulus of i describes
// which byte of the multi-byte integer is being handled.
// The value start + i must be at least stride * 4.
uint8_t PredictValue(const uint8_t* data, size_t start, size_t i, size_t stride,
                     size_t width, int order) {
  size_t pos = start + i;
  if (width == 1) {
    uint8_t p1 = data[pos - stride];
    uint8_t p2 = data[pos - stride * 2];
    uint8_t p3 = data[pos - stride * 3];
    return PredictValue(p1, p2, p3, order);
  } else if (width == 2) {
    size_t p = start + (i & ~1);
    uint16_t p1 = (data[p - stride * 1] << 8) + data[p - stride * 1 + 1];
    uint16_t p2 = (data[p - stride * 2] << 8) + data[p - stride * 2 + 1];
    uint16_t p3 = (data[p - stride * 3] << 8) + data[p - stride * 3 + 1];
    uint16_t pred = PredictValue(p1, p2, p3, order);
    return (i & 1) ? (pred & 255) : ((pred >> 8) & 255);
  } else {
    size_t p = start + (i & ~3);
    uint32_t p1 = DecodeUint32(data, pos, p - stride);
    uint32_t p2 = DecodeUint32(data, pos, p - stride * 2);
    uint32_t p3 = DecodeUint32(data, pos, p - stride * 3);
    uint32_t pred = PredictValue(p1, p2, p3, order);
    unsigned shiftbytes = 3 - (i & 3);
    return (pred >> (shiftbytes * 8)) & 255;
  }
}

// Checks if a + b > size, taking possible integer overflow into account.
Status CheckOutOfBounds(size_t a, size_t b, size_t size) {
  size_t pos = a + b;
  if (pos > size) return JXL_FAILURE("Out of bounds");
  if (pos < a) return JXL_FAILURE("Out of bounds");  // overflow happened
  return true;
}

Status CheckIs32Bit(uint64_t v) {
  static constexpr const uint64_t kUpper32 = ~static_cast<uint64_t>(0xFFFFFFFF);
  if ((v & kUpper32) != 0) return  JXL_FAILURE("32-bit value expected");
  return true;
}

// This is performed by the encoder, the encoder must be able to encode any
// random byte stream (not just byte streams that are a valid ICC profile), so
// an error returned by this function is an implementation error.
Status PredictAndShuffle(size_t stride, size_t width, int order, size_t num,
                         const uint8_t* data, size_t size, size_t* pos,
                         PaddedBytes* result) {
  JXL_RETURN_IF_ERROR(CheckOutOfBounds(*pos, num, size));
  if (*pos < stride * 4) return JXL_FAILURE("Too large stride");
  size_t start = result->size();
  for (size_t i = 0; i < num; i++) {
    uint8_t predicted = PredictValue(data, *pos, i, stride, width, order);
    result->push_back(data[*pos + i] - predicted);
  }
  *pos += num;
  if (width > 1) Unshuffle(result->data() + start, num, width);
  return true;
}

}  // namespace

// Decodes the result of PredictICC back to a valid ICC profile.
Status UnpredictICC(const uint8_t* enc, size_t size, PaddedBytes* result) {
  if (!result->empty()) return JXL_FAILURE("result must be empty initially");
  size_t pos = 0;
  // TODO(lode): technically speaking we need to check that the entire varint
  // decoding never goes out of bounds, not just the first byte. This requires
  // a DecodeVarInt function that returns an error code. It is safe to use
  // DecodeVarInt with out of bounds values, it silently returns, but the
  // specification requires an error. Idem for all DecodeVarInt below.
  if (pos >= size) return JXL_FAILURE("Out of bounds");
  uint64_t osize = DecodeVarInt(enc, size, &pos);  // Output size
  JXL_RETURN_IF_ERROR(CheckIs32Bit(osize));
  if (pos >= size) return JXL_FAILURE("Out of bounds");
  uint64_t csize = DecodeVarInt(enc, size, &pos);  // Commands size
  // Every command is translated to at least on byte.
  JXL_RETURN_IF_ERROR(CheckIs32Bit(csize));
  size_t cpos = pos;  // pos in commands stream
  JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, csize, size));
  size_t commands_end = cpos + csize;
  pos = commands_end;  // pos in data stream

  // Header
  PaddedBytes header = InitHeaderPred();
  EncodeUint32(0, osize, &header);
  for (size_t i = 0; i <= kICCHeaderSize; i++) {
    if (result->size() == osize) {
      if (cpos != commands_end) return JXL_FAILURE("Not all commands used");
      if (pos != size) return JXL_FAILURE("Not all data used");
      return true;  // Valid end
    }
    if (i == kICCHeaderSize) break;  // Done
    PredictHeader(result->data(), result->size(), header.data(), i);
    if (pos >= size) return JXL_FAILURE("Out of bounds");
    result->push_back(enc[pos++] + header[i]);
  }
  if (cpos >= commands_end) return JXL_FAILURE("Out of bounds");

  // Tag list
  uint64_t numtags = DecodeVarInt(enc, size, &cpos);
  JXL_RETURN_IF_ERROR(CheckIs32Bit(numtags));

  if (numtags != 0) {
    numtags--;
    AppendUint32(numtags, result);
    uint64_t prevtagstart = kICCHeaderSize + numtags * 12;
    JXL_RETURN_IF_ERROR(CheckIs32Bit(prevtagstart));
    uint64_t prevtagsize = 0;
    for (;;) {
      if (result->size() > osize) return JXL_FAILURE("Invalid result size");
      if (cpos > commands_end) return JXL_FAILURE("Out of bounds");
      if (cpos == commands_end) break;  // Valid end
      uint8_t command = enc[cpos++];
      uint8_t tagcode = command & 63;
      Tag tag;
      if (tagcode == 0) {
        break;
      } else if (tagcode == kCommandTagUnknown) {
        JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, 4, size));
        tag = DecodeKeyword(enc, size, pos);
        pos += 4;
      } else if (tagcode == kCommandTagTRC) {
        tag = kRtrcTag;
      } else if (tagcode == kCommandTagXYZ) {
        tag = kRxyzTag;
      } else {
        if (tagcode - kCommandTagStringFirst >= kNumTagStrings) {
          return JXL_FAILURE("Unknown tagcode");
        }
        tag = *kTagStrings[tagcode - kCommandTagStringFirst];
      }
      AppendKeyword(tag, result);

      // Safe to add, since the addends are 32-bit.
      uint64_t tagstart = prevtagstart + prevtagsize;
      uint64_t tagsize = prevtagsize;
      if (tag == kRxyzTag || tag == kGxyzTag || tag == kBxyzTag ||
          tag == kKxyzTag || tag == kWtptTag || tag == kBkptTag ||
          tag == kLumiTag) {
        tagsize = 20;
      }

      if (command & kFlagBitOffset) {
        if (cpos >= commands_end) return JXL_FAILURE("Out of bounds");
        tagstart = DecodeVarInt(enc, size, &cpos);
      }
      JXL_RETURN_IF_ERROR(CheckIs32Bit(tagstart));
      AppendUint32(tagstart, result);
      if (command & kFlagBitSize) {
        if (cpos >= commands_end) return JXL_FAILURE("Out of bounds");
        tagsize = DecodeVarInt(enc, size, &cpos);
      }
      JXL_RETURN_IF_ERROR(CheckIs32Bit(tagsize));
      AppendUint32(tagsize, result);
      prevtagstart = tagstart;
      prevtagsize = tagsize;

      if (tagcode == kCommandTagTRC) {
        AppendKeyword(kGtrcTag, result);
        AppendUint32(tagstart, result);
        AppendUint32(tagsize, result);
        AppendKeyword(kBtrcTag, result);
        AppendUint32(tagstart, result);
        AppendUint32(tagsize, result);
      }

      if (tagcode == kCommandTagXYZ) {
        JXL_RETURN_IF_ERROR(CheckIs32Bit(tagstart + tagsize * 2));
        AppendKeyword(kGxyzTag, result);
        AppendUint32(tagstart + tagsize, result);
        AppendUint32(tagsize, result);
        AppendKeyword(kBxyzTag, result);
        AppendUint32(tagstart + tagsize * 2, result);
        AppendUint32(tagsize, result);
      }
    }
  }

  // Main Content
  for (;;) {
    if (result->size() > osize) return JXL_FAILURE("Invalid result size");
    if (cpos > commands_end) return JXL_FAILURE("Out of bounds");
    if (cpos == commands_end) break;  // Valid end
    uint8_t command = enc[cpos++];
    if (command == kCommandInsert) {
      if (cpos >= commands_end) return JXL_FAILURE("Out of bounds");
      uint64_t num = DecodeVarInt(enc, size, &cpos);
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, num, size));
      for (size_t i = 0; i < num; i++) {
        result->push_back(enc[pos++]);
      }
    } else if (command == kCommandShuffle2 || command == kCommandShuffle4) {
      if (cpos >= commands_end) return JXL_FAILURE("Out of bounds");
      uint64_t num = DecodeVarInt(enc, size, &cpos);
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, num, size));
      PaddedBytes shuffled(num);
      for (size_t i = 0; i < num; i++) {
        shuffled[i] = enc[pos + i];
      }
      if (command == kCommandShuffle2) {
        Shuffle(shuffled.data(), num, 2);
      } else if (command == kCommandShuffle4) {
        Shuffle(shuffled.data(), num, 4);
      }
      for (size_t i = 0; i < num; i++) {
        result->push_back(shuffled[i]);
        pos++;
      }
    } else if (command == kCommandPredict) {
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(cpos, 2, commands_end));
      uint8_t flags = enc[cpos++];

      size_t width = (flags & 3) + 1;
      if (width == 3) return JXL_FAILURE("Invalid width");

      int order = (flags & 12) >> 2;
      if (order == 3) return JXL_FAILURE("Invalid order");

      uint64_t stride = width;
      if (flags & 16) {
        if (cpos >= commands_end) return JXL_FAILURE("Out of bounds");
        stride = DecodeVarInt(enc, size, &cpos);
        if (stride < width) {
          return JXL_FAILURE("Invalid stride");
        }
      }
      // If stride * 4 >= result->size(), return failure. The check
      // "size == 0 || ((size - 1) >> 2) < stride" corresponds to
      // "stride * 4 >= size", but does not suffer from integer overflow.
      // TODO(lode): this check is more strict than necessary, returning
      // failure if stride * (order + 1) > result->size() is sufficient, but
      // changing this requires a comment to the spec and to PredictValue.
      if (result->empty() || ((result->size() - 1u) >> 2u) < stride) {
        return JXL_FAILURE("Invalid stride");
      }

      if (cpos >= commands_end) return JXL_FAILURE("Out of bounds");
      uint64_t num = DecodeVarInt(enc, size, &cpos);  // in bytes
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, num, size));

      PaddedBytes shuffled(num);
      for (size_t i = 0; i < num; i++) {
        shuffled[i] = enc[pos + i];
      }
      if (width > 1) Shuffle(shuffled.data(), num, width);

      size_t start = result->size();
      for (size_t i = 0; i < num; i++) {
        uint8_t predicted =
            PredictValue(result->data(), start, i, stride, width, order);
        result->push_back(predicted + shuffled[i]);
      }
      pos += num;
    } else if (command == kCommandXYZ) {
      AppendKeyword(kXyz_Tag, result);
      for (int i = 0; i < 4; i++) result->push_back(0);
      JXL_RETURN_IF_ERROR(CheckOutOfBounds(pos, 12, size));
      for (size_t i = 0; i < 12; i++) {
        result->push_back(enc[pos++]);
      }
    } else if (command >= kCommandTypeStartFirst &&
               command < kCommandTypeStartFirst + kNumTypeStrings) {
      AppendKeyword(*kTypeStrings[command - kCommandTypeStartFirst], result);
      for (size_t i = 0; i < 4; i++) {
        result->push_back(0);
      }
    } else {
      return JXL_FAILURE("Unknown command");
    }
  }

  if (pos != size) return JXL_FAILURE("Not all data used");
  if (result->size() != osize) return JXL_FAILURE("Invalid result size");

  return true;
}

// Outputs a transformed form of the given icc profile. The result itself is
// not particularly smaller than the input data in bytes, but it will be in a
// form that is easier to compress (more zeroes, ...) and will compress better
// with brotli.
Status PredictICC(const uint8_t* icc, size_t size, PaddedBytes* result) {
  PaddedBytes commands;
  PaddedBytes data;

  EncodeVarInt(size, result);

  // Header
  PaddedBytes header = InitHeaderPred();
  EncodeUint32(0, size, &header);
  for (size_t i = 0; i < kICCHeaderSize && i < size; i++) {
    PredictHeader(icc, size, header.data(), i);
    data.push_back(icc[i] - header[i]);
  }
  if (size <= kICCHeaderSize) {
    EncodeVarInt(0, result);  // 0 commands
    for (size_t i = 0; i < data.size(); i++) {
      result->push_back(data[i]);
    }
    return true;
  }

  std::vector<Tag> tags;
  std::vector<size_t> tagstarts;
  std::vector<size_t> tagsizes;
  std::map<size_t, size_t> tagmap;

  // Tag list
  size_t pos = kICCHeaderSize;
  if (pos + 4 <= size) {
    uint64_t numtags = DecodeUint32(icc, size, pos);
    pos += 4;
    EncodeVarInt(numtags + 1, &commands);
    uint64_t prevtagstart = kICCHeaderSize + numtags * 12;
    uint32_t prevtagsize = 0;
    for (size_t i = 0; i < numtags; i++) {
      if (pos + 12 > size) break;

      Tag tag = DecodeKeyword(icc, size, pos + 0);
      uint32_t tagstart = DecodeUint32(icc, size, pos + 4);
      uint32_t tagsize = DecodeUint32(icc, size, pos + 8);
      pos += 12;

      tags.push_back(tag);
      tagstarts.push_back(tagstart);
      tagsizes.push_back(tagsize);
      tagmap[tagstart] = tags.size() - 1;

      uint8_t tagcode = kCommandTagUnknown;
      for (size_t j = 0; j < kNumTagStrings; j++) {
        if (tag == *kTagStrings[j]) {
          tagcode = j + kCommandTagStringFirst;
          break;
        }
      }

      if (tag == kRtrcTag && pos + 24 < size) {
        bool ok = true;
        ok &= DecodeKeyword(icc, size, pos + 0) == kGtrcTag;
        ok &= DecodeKeyword(icc, size, pos + 12) == kBtrcTag;
        if (ok) {
          for (size_t i = 0; i < 8; i++) {
            if (icc[pos - 8 + i] != icc[pos + 4 + i]) ok = false;
            if (icc[pos - 8 + i] != icc[pos + 16 + i]) ok = false;
          }
        }
        if (ok) {
          tagcode = kCommandTagTRC;
          pos += 24;
          i += 2;
        }
      }

      if (tag == kRxyzTag && pos + 24 < size) {
        bool ok = true;
        ok &= DecodeKeyword(icc, size, pos + 0) == kGxyzTag;
        ok &= DecodeKeyword(icc, size, pos + 12) == kBxyzTag;
        uint32_t offsetr = tagstart;
        uint32_t offsetg = DecodeUint32(icc, size, pos + 4);
        uint32_t offsetb = DecodeUint32(icc, size, pos + 16);
        uint32_t sizer = tagsize;
        uint32_t sizeg = DecodeUint32(icc, size, pos + 8);
        uint32_t sizeb = DecodeUint32(icc, size, pos + 20);
        ok &= sizer == 20;
        ok &= sizeg == 20;
        ok &= sizeb == 20;
        ok &= (offsetg == offsetr + 20);
        ok &= (offsetb == offsetr + 40);
        if (ok) {
          tagcode = kCommandTagXYZ;
          pos += 24;
          i += 2;
        }
      }

      uint8_t command = tagcode;
      uint64_t predicted_tagstart = prevtagstart + prevtagsize;
      if (predicted_tagstart != tagstart) command |= kFlagBitOffset;
      size_t predicted_tagsize = prevtagsize;
      if (tag == kRxyzTag || tag == kGxyzTag || tag == kBxyzTag ||
          tag == kKxyzTag || tag == kWtptTag || tag == kBkptTag ||
          tag == kLumiTag) {
        predicted_tagsize = 20;
      }
      if (predicted_tagsize != tagsize) command |= kFlagBitSize;
      commands.push_back(command);
      if (tagcode == 1) {
        AppendKeyword(tag, &data);
      }
      if (command & kFlagBitOffset) EncodeVarInt(tagstart, &commands);
      if (command & kFlagBitSize) EncodeVarInt(tagsize, &commands);

      prevtagstart = tagstart;
      prevtagsize = tagsize;
    }
  }
  // Indicate end of tag list or varint indicating there's none
  commands.push_back(0);

  // Main content
  // The main content in a valid ICC profile contains tagged elements, with the
  // tag types (4 letter names) given by the tag list above, and the tag list
  // pointing to the start and indicating the size of each tagged element. It is
  // allowed for tagged elements to overlap, e.g. the curve for R, G and B could
  // all point to the same one.
  Tag tag;
  size_t tagstart = 0, tagsize = 0, clutstart = 0;

  size_t last0 = pos;
  // This loop appends commands to the output, processing some sub-section of a
  // current tagged element each time. We need to keep track of the tagtype of
  // the current element, and update it when we encounter the boundary of a
  // next one.
  // It is not required that the input data is a valid ICC profile, if the
  // encoder does not recognize the data it will still be able to output bytes
  // but will not predict as well.
  while (pos <= size) {
    size_t last1 = pos;
    PaddedBytes commands_add;
    PaddedBytes data_add;

    // This means the loop brought the position beyond the tag end.
    if (pos > tagstart + tagsize) {
      tag = {0, 0, 0, 0};  // nonsensical value
    }

    if (commands_add.empty() && data_add.empty() && tagmap.count(pos) &&
        pos + 4 <= size) {
      size_t index = tagmap[pos];
      tag = DecodeKeyword(icc, size, pos);
      tagstart = tagstarts[index];
      tagsize = tagsizes[index];

      if (tag == kMlucTag && pos + tagsize <= size && tagsize > 8 &&
          icc[pos + 4] == 0 && icc[pos + 5] == 0 && icc[pos + 6] == 0 &&
          icc[pos + 7] == 0) {
        size_t num = tagsize - 8;
        commands_add.push_back(kCommandTypeStartFirst + 3);
        pos += 8;
        commands_add.push_back(kCommandShuffle2);
        EncodeVarInt(num, &commands_add);
        size_t start = data_add.size();
        for (size_t i = 0; i < num; i++) {
          data_add.push_back(icc[pos]);
          pos++;
        }
        Unshuffle(data_add.data() + start, num, 2);
      }

      if (tag == kCurvTag && pos + tagsize <= size && tagsize > 8 &&
          icc[pos + 4] == 0 && icc[pos + 5] == 0 && icc[pos + 6] == 0 &&
          icc[pos + 7] == 0) {
        size_t num = tagsize - 8;
        if (num > 16 && num < (1 << 28) && pos + num <= size) {
          commands_add.push_back(kCommandTypeStartFirst + 5);
          pos += 8;
          commands_add.push_back(kCommandPredict);
          int order = 1, width = 2, stride = width;
          commands_add.push_back((order << 2) | (width - 1));
          EncodeVarInt(num, &commands_add);
          JXL_RETURN_IF_ERROR(PredictAndShuffle(stride, width, order, num, icc,
                                                size, &pos, &data_add));
        }
      }
    }

    if (tag == kMab_Tag || tag == kMba_Tag) {
      Tag subTag = DecodeKeyword(icc, size, pos);
      if (pos + 12 < size &&
          (subTag == kCurvTag || subTag == kVcgtTag) &&
          DecodeUint32(icc, size, pos + 4) == 0) {
        uint32_t num = DecodeUint32(icc, size, pos + 8) * 2;
        if (num > 16 && num < (1 << 28) && pos + 12 + num <= size) {
          pos += 12;
          last1 = pos;
          commands_add.push_back(kCommandPredict);
          int order = 1, width = 2, stride = width;
          commands_add.push_back((order << 2) | (width - 1));
          EncodeVarInt(num, &commands_add);
          JXL_RETURN_IF_ERROR(PredictAndShuffle(stride, width, order, num, icc,
                                                size, &pos, &data_add));
        }
      }

      if (pos == tagstart + 24 && pos + 4 < size) {
        // Note that this value can be remembered for next iterations of the
        // loop, so the "pos == clutstart" if below can trigger during a later
        // iteration.
        clutstart = tagstart + DecodeUint32(icc, size, pos);
      }

      if (pos == clutstart && clutstart + 16 < size) {
        size_t numi = icc[tagstart + 8];
        size_t numo = icc[tagstart + 9];
        size_t width = icc[clutstart + 16];
        size_t stride = width * numo;
        size_t num = width * numo;
        for (size_t i = 0; i < numi && clutstart + i < size; i++) {
          num *= icc[clutstart + i];
        }
        if ((width == 1 || width == 2) && num > 64 && num < (1 << 28) &&
            pos + num <= size && pos >= stride * 4) {
          commands_add.push_back(kCommandPredict);
          int order = 1;
          uint8_t flags =
              (order << 2) | (width - 1) | (stride == width ? 0 : 16);
          commands_add.push_back(flags);
          if (flags & 16) EncodeVarInt(stride, &commands_add);
          EncodeVarInt(num, &commands_add);
          JXL_RETURN_IF_ERROR(PredictAndShuffle(stride, width, order, num, icc,
                                                size, &pos, &data_add));
        }
      }
    }

    if (commands_add.empty() && data_add.empty() && tag == kGbd_Tag &&
        pos == tagstart + 8 && pos + tagsize - 8 <= size && pos >= 16 &&
        tagsize > 8) {
      size_t width = 4, order = 0, stride = width;
      size_t num = tagsize - 8;
      uint8_t flags = (order << 2) | (width - 1) | (stride == width ? 0 : 16);
      commands_add.push_back(kCommandPredict);
      commands_add.push_back(flags);
      if (flags & 16) EncodeVarInt(stride, &commands_add);
      EncodeVarInt(num, &commands_add);
      JXL_RETURN_IF_ERROR(PredictAndShuffle(stride, width, order, num, icc,
                                            size, &pos, &data_add));
    }

    if (commands_add.empty() && data_add.empty() && pos + 20 <= size) {
      Tag subTag = DecodeKeyword(icc, size, pos);
      if (subTag == kXyz_Tag && DecodeUint32(icc, size, pos + 4) == 0) {
        commands_add.push_back(kCommandXYZ);
        pos += 8;
        for (size_t j = 0; j < 12; j++) data_add.push_back(icc[pos++]);
      }
    }

    if (commands_add.empty() && data_add.empty() && pos + 8 <= size) {
      if (DecodeUint32(icc, size, pos + 4) == 0) {
        Tag subTag = DecodeKeyword(icc, size, pos);
        for (size_t i = 0; i < kNumTypeStrings; i++) {
          if (subTag == *kTypeStrings[i]) {
            commands_add.push_back(kCommandTypeStartFirst + i);
            pos += 8;
            break;
          }
        }
      }
    }

    if (!(commands_add.empty() && data_add.empty()) || pos == size) {
      if (last0 < last1) {
        commands.push_back(kCommandInsert);
        EncodeVarInt(last1 - last0, &commands);
        while (last0 < last1) {
          data.push_back(icc[last0++]);
        }
      }
      for (size_t i = 0; i < commands_add.size(); i++) {
        commands.push_back(commands_add[i]);
      }
      for (size_t i = 0; i < data_add.size(); i++) {
        data.push_back(data_add[i]);
      }
      last0 = pos;
    }
    if (commands_add.empty() && data_add.empty()) {
      pos++;
    }
  }

  EncodeVarInt(commands.size(), result);
  for (size_t i = 0; i < commands.size(); i++) {
    result->push_back(commands[i]);
  }
  for (size_t i = 0; i < data.size(); i++) {
    result->push_back(data[i]);
  }

  return true;
}

static constexpr size_t kNumICCContexts = 41;

static uint8_t ByteKind1(uint8_t b) {
  if ('a' <= b && b <= 'z') return 0;
  if ('A' <= b && b <= 'Z') return 0;
  if ('0' <= b && b <= '9') return 1;
  if (b == '.' || b == ',') return 1;
  if (b == 0) return 2;
  if (b == 1) return 3;
  if (b < 16) return 4;
  if (b == 255) return 6;
  if (b > 240) return 5;
  return 7;
}

static uint8_t ByteKind2(uint8_t b) {
  if ('a' <= b && b <= 'z') return 0;
  if ('A' <= b && b <= 'Z') return 0;
  if ('0' <= b && b <= '9') return 1;
  if (b == '.' || b == ',') return 1;
  if (b < 16) return 2;
  if (b > 240) return 3;
  return 4;
}

size_t Context(size_t i, size_t b1, size_t b2) {
  if (i <= 128) return 0;
  return 1 + ByteKind1(b1) + ByteKind2(b2) * 8;
}

Status WriteICC(const PaddedBytes& icc, BitWriter* JXL_RESTRICT writer,
                size_t layer, AuxOut* JXL_RESTRICT aux_out) {
  if (icc.empty()) return JXL_FAILURE("ICC must be non-empty");
  PaddedBytes enc;
  JXL_RETURN_IF_ERROR(PredictICC(icc.data(), icc.size(), &enc));
  std::vector<std::vector<Token>> tokens(1);
  BitWriter::Allotment allotment(writer, 128);
  JXL_RETURN_IF_ERROR(U64Coder::Write(enc.size(), writer));
  ReclaimAndCharge(writer, &allotment, layer, aux_out);

  for (size_t i = 0; i < enc.size(); i++) {
    tokens[0].emplace_back(
        Context(i, i > 0 ? enc[i - 1] : 0, i > 1 ? enc[i - 2] : 0), enc[i]);
  }
  HistogramParams params;
  params.lz77_method = HistogramParams::LZ77Method::kOptimal;
  EntropyEncodingData code;
  std::vector<uint8_t> context_map;
  params.force_huffman = true;
  BuildAndEncodeHistograms(params, kNumICCContexts, tokens, &code, &context_map,
                           writer, layer, aux_out);
  WriteTokens(tokens[0], code, context_map, writer, layer, aux_out);
  return true;
}

Status ReadICC(BitReader* JXL_RESTRICT reader, PaddedBytes* JXL_RESTRICT icc) {
  icc->clear();
  if (!reader->AllReadsWithinBounds()) {
    return JXL_STATUS(StatusCode::kNotEnoughBytes,
                      "Not enough bytes for reading ICC profile");
  }
  uint64_t enc_size = U64Coder::Read(reader);
  if (enc_size > 268435456) {
    // Avoid too large memory allocation for invalid file.
    // TODO(lode): a more accurate limit would be the filesize of the JXL file,
    // if we can have it available here.
    return JXL_FAILURE("Too large encoded profile");
  }
  PaddedBytes decompressed(enc_size);
  std::vector<uint8_t> context_map;
  ANSCode code;
  JXL_RETURN_IF_ERROR(
      DecodeHistograms(reader, kNumICCContexts, &code, &context_map));
  ANSSymbolReader ans_reader(&code, reader);
  for (size_t i = 0; i < decompressed.size(); i++) {
    decompressed[i] =
        ans_reader.ReadHybridUint(Context(i, i > 0 ? decompressed[i - 1] : 0,
                                          i > 1 ? decompressed[i - 2] : 0),
                                  reader, context_map);
    if (!reader->AllReadsWithinBounds()) {
      return JXL_STATUS(StatusCode::kNotEnoughBytes,
                        "Not enough bytes for reading ICC profile");
    }
  }
  if (!ans_reader.CheckANSFinalState()) {
    return JXL_FAILURE("Corrupted ICC profile");
  }

  JXL_RETURN_IF_ERROR(
      UnpredictICC(decompressed.data(), decompressed.size(), icc));
  return true;
}

}  // namespace jxl
