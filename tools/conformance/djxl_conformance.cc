
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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <vector>

#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"

namespace {

struct DecodeOptions {
  // Path to the input .jxl file.
  const char* input = nullptr;

  // Prefix of the output path where to generate the pixel data or nullptr if
  // no pixel data should be save to disk.
  const char* pixel_prefix = nullptr;

  // Whether to also generate a preview output when generating pixel data.
  bool preview = false;

  // Path to the original ICC profile to be generated, if requested.
  const char* icc_path = nullptr;

  // Path to JPEG reconstruction file to be generated, if requested.
  const char* jpeg_path = nullptr;

  // Path to the metadata text file to be generated, if requested.
  const char* metadata_path = nullptr;
};

bool LoadFile(const char* filename, std::vector<uint8_t>* data) {
  std::ifstream ifs(filename, std::ios::binary);
  std::vector<uint8_t> contents((std::istreambuf_iterator<char>(ifs)),
                                std::istreambuf_iterator<char>());
  ifs.close();
  *data = std::move(contents);
  return ifs.good();
}

bool SaveFile(const char* filename, std::vector<uint8_t> data) {
  std::ofstream ofs(filename, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
  ofs.close();
  return ofs.good();
}

struct ImageArray {
  uint32_t xsize, ysize;
  uint32_t num_channels;

  // An array of "frames", where each frame is a 3D array of samples with
  // dimensions (ysize, xsize, channel).
  std::vector<std::vector<uint8_t>> frames;
};

// Saves an ImageArray as a numpy 4D ndarray in binary format.
bool SaveNPYArray(const char* filename, const ImageArray& arr) {
  size_t image_size = sizeof(float) * arr.xsize * arr.ysize * arr.num_channels;
  for (const auto& frame : arr.frames) {
    if (frame.size() != image_size) {
      fprintf(stderr, "Invalid frame size\n");
      return false;
    }
  }

  FILE* file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Could not open %s for writing", filename);
    return false;
  }
#define WRITE_TO_FILE(ptr, len)                                         \
  do {                                                                  \
    if (fwrite((ptr), (len), 1, file) != 1) {                           \
      fprintf(stderr, "Error writing " #ptr " to file %s\n", filename); \
      fclose(file);                                                     \
      return false;                                                     \
    }                                                                   \
  } while (0)

  const uint8_t header[] = "\x93NUMPY\x01\x00";
  WRITE_TO_FILE(header, 8);

  {
    std::stringstream ss;
    ss << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
       << arr.frames.size() << ", " << arr.ysize << ", " << arr.xsize << ", "
       << arr.num_channels << "), }\n";
    // 16-bit little endian header length.
    uint8_t header_len[2] = {static_cast<uint8_t>(ss.str().size() % 256),
                             static_cast<uint8_t>(ss.str().size() / 256)};
    WRITE_TO_FILE(header_len, 2);
    WRITE_TO_FILE(ss.str().data(), ss.str().size());
  }

  for (const auto& frame : arr.frames) {
    WRITE_TO_FILE(frame.data(), frame.size());
  }

  return fclose(file) == 0;
#undef WRITE_TO_FILE
}

// JSON value writing

class JSONField {
 public:
  virtual ~JSONField() = default;
  virtual void Write(std::ostream& o, uint32_t indent) const = 0;

 protected:
  JSONField() = default;
};

class JSONValue : public JSONField {
 public:
  template <typename T>
  explicit JSONValue(const T& value) : value_(std::to_string(value)) {}

  explicit JSONValue(const std::string& value) : value_("\"" + value + "\"") {}

  explicit JSONValue(bool value) : value_(value ? "true" : "false") {}

  void Write(std::ostream& o, uint32_t indent) const override { o << value_; }

 private:
  std::string value_;
};

class JSONDict : public JSONField {
 public:
  JSONDict() = default;

  template <typename T>
  T* AddEmpty(const std::string& key) {
    static_assert(std::is_convertible<T*, JSONField*>::value,
                  "T must be a JSONField");
    T* ret = new T();
    values_.emplace_back(
        key, std::unique_ptr<JSONField>(static_cast<JSONField*>(ret)));
    return ret;
  }

  template <typename T>
  void Add(const std::string& key, const T& value) {
    values_.emplace_back(key, std::unique_ptr<JSONField>(new JSONValue(value)));
  }

  void Write(std::ostream& o, uint32_t indent) const override {
    std::string indent_str(indent, ' ');
    o << "{";
    bool is_first = true;
    for (const auto& key_value : values_) {
      if (!is_first) {
        o << ",";
      }
      is_first = false;
      o << std::endl << indent_str << "  \"" << key_value.first << "\": ";
      key_value.second->Write(o, indent + 2);
    }
    if (!values_.empty()) {
      o << std::endl << indent_str;
    }
    o << "}";
  }

 private:
  // Dictionary with order.
  std::vector<std::pair<std::string, std::unique_ptr<JSONField>>> values_;
};

class JSONArray : public JSONField {
 public:
  JSONArray() = default;

  template <typename T>
  T* AddEmpty() {
    static_assert(std::is_convertible<T*, JSONField*>::value,
                  "T must be a JSONField");
    T* ret = new T();
    values_.emplace_back(ret);
    return ret;
  }

  template <typename T>
  void Add(const T& value) {
    values_.emplace_back(new JSONValue(value));
  }

  void Write(std::ostream& o, uint32_t indent) const override {
    std::string indent_str(indent, ' ');
    o << "[";
    bool is_first = true;
    for (const auto& value : values_) {
      if (!is_first) {
        o << ",";
      }
      is_first = false;
      o << std::endl << indent_str << "  ";
      value->Write(o, indent + 2);
    }
    if (!values_.empty()) {
      o << std::endl << indent_str;
    }
    o << "]";
  }

 private:
  std::vector<std::unique_ptr<JSONField>> values_;
};

#define EXPECT_TRUE(X)                     \
  do {                                     \
    if (!(X)) {                            \
      fprintf(stderr, "Failed: %s\n", #X); \
      return false;                        \
    }                                      \
  } while (false)

// Helper macro for decoder error checking.
#define EXPECT_SUCCESS(X) EXPECT_TRUE((X) == JXL_DEC_SUCCESS)

bool DecodeJXL(const DecodeOptions& opts) {
  auto dec = JxlDecoderMake(nullptr);

  uint32_t events = JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING;
  if (opts.pixel_prefix) events |= JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE;
  if (opts.jpeg_path) events |= JXL_DEC_JPEG_RECONSTRUCTION;
  if (opts.preview) events |= JXL_DEC_PREVIEW_IMAGE;
  // We need to output the frame header info in the metadata.
  if (opts.metadata_path) events |= JXL_DEC_FRAME;

  EXPECT_SUCCESS(JxlDecoderSubscribeEvents(dec.get(), events));

  // TODO(deymo): Consider using a multi-threading decoder for conformance
  // testing as well.

  // Load and set input all at oncee.
  std::vector<uint8_t> jxl_input;
  EXPECT_TRUE(LoadFile(opts.input, &jxl_input));
  EXPECT_SUCCESS(
      JxlDecoderSetInput(dec.get(), jxl_input.data(), jxl_input.size()));

  JxlBasicInfo info{};

  // JPEG output buffer when reconstructing a JPEG file.
  std::vector<uint8_t> jpeg_data;
  std::vector<uint8_t> jpeg_data_chunk(16 * 1024);

  // Pixel data when decoding a frame or a preview frame.
  std::vector<uint8_t> pixels;
  std::vector<uint8_t> preview_pixels;

  std::vector<JxlExtraChannelInfo> extra_channels;
  std::vector<std::string> extra_channel_names;

  std::vector<JxlFrameHeader> frame_headers;
  std::vector<std::string> frame_names;

  JxlPixelFormat format;
  uint32_t num_channels = 0;

  ImageArray image, preview;

  while (true) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Error decoding.\n");
      return false;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      fprintf(stderr, "Error decoding: expected more input.\n");
      return false;
    } else if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
      // Decoding to JPEG.
      EXPECT_TRUE(opts.jpeg_path);
      EXPECT_SUCCESS(JxlDecoderSetJPEGBuffer(dec.get(), jpeg_data_chunk.data(),
                                             jpeg_data_chunk.size()));
    } else if (status == JXL_DEC_JPEG_NEED_MORE_OUTPUT) {
      // Decoded a chunk to JPEG.
      EXPECT_TRUE(opts.jpeg_path);
      size_t used_jpeg_output =
          jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec.get());
      jpeg_data.insert(jpeg_data.end(), jpeg_data_chunk.data(),
                       jpeg_data_chunk.data() + used_jpeg_output);
      EXPECT_SUCCESS(JxlDecoderSetJPEGBuffer(dec.get(), jpeg_data_chunk.data(),
                                             jpeg_data_chunk.size()));
    } else if (status == JXL_DEC_BASIC_INFO) {
      // Basic info.
      EXPECT_SUCCESS(JxlDecoderGetBasicInfo(dec.get(), &info));
      extra_channels.resize(info.num_extra_channels);
      for (uint32_t i = 0; i < info.num_extra_channels; ++i) {
        EXPECT_SUCCESS(
            JxlDecoderGetExtraChannelInfo(dec.get(), i, &extra_channels[i]));
        std::vector<char> name(extra_channels[i].name_length + 1);
        EXPECT_SUCCESS(JxlDecoderGetExtraChannelName(dec.get(), i, name.data(),
                                                     name.size()));
        extra_channel_names.emplace_back(name.begin(), name.end() - 1);
      }

      // Select the output pixel format based on the basic info.
      num_channels = (info.alpha_bits > 0 ? 1 : 0) + info.num_color_channels;
      format =
          JxlPixelFormat{num_channels, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, 0};
      image.num_channels = num_channels;
      image.xsize = info.xsize;
      image.ysize = info.ysize;

      if (opts.preview) {
        // Requesting a preview if the .jxl file doesn't have one is an error.
        EXPECT_TRUE(info.have_preview);
        preview.num_channels = num_channels;
        preview.xsize = info.preview.xsize;
        preview.ysize = info.preview.ysize;
      }
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      // ICC profiles.
      if (opts.icc_path) {
        // Store the original ICC if requested.
        size_t icc_size;
        EXPECT_SUCCESS(JxlDecoderGetICCProfileSize(
            dec.get(), nullptr, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &icc_size));
        std::vector<uint8_t> icc_original(icc_size);
        EXPECT_SUCCESS(JxlDecoderGetColorAsICCProfile(
            dec.get(), nullptr, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
            icc_original.data(), icc_original.size()));
        EXPECT_TRUE(SaveFile(opts.icc_path, icc_original));
      }

      if (opts.pixel_prefix) {
        // Get the ICC color profile of the pixel data and store it.
        size_t icc_size;
        EXPECT_SUCCESS(JxlDecoderGetICCProfileSize(
            dec.get(), &format, JXL_COLOR_PROFILE_TARGET_DATA, &icc_size));
        std::vector<uint8_t> icc_data(icc_size);
        EXPECT_SUCCESS(JxlDecoderGetColorAsICCProfile(
            dec.get(), &format, JXL_COLOR_PROFILE_TARGET_DATA, icc_data.data(),
            icc_data.size()));
        std::string icc_data_filename = std::string(opts.pixel_prefix) + ".icc";
        EXPECT_TRUE(SaveFile(icc_data_filename.c_str(), icc_data));
      }
    } else if (status == JXL_DEC_FRAME) {
      // Capture the frame header information.
      JxlFrameHeader frame_header;
      EXPECT_SUCCESS(JxlDecoderGetFrameHeader(dec.get(), &frame_header));
      std::vector<char> frame_name(frame_header.name_length + 1);
      EXPECT_SUCCESS(JxlDecoderGetFrameName(dec.get(), frame_name.data(),
                                            frame_name.size()));
      EXPECT_TRUE(frame_name[frame_name.size() - 1] == '\0');
      frame_headers.emplace_back(frame_header);
      frame_names.emplace_back(frame_name.begin(), frame_name.end() - 1);
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      // Set pixel output buffer.
      size_t buffer_size;
      EXPECT_SUCCESS(
          JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size));
      pixels.resize(buffer_size);
      memset(pixels.data(), 0, pixels.size());
      EXPECT_SUCCESS(JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                                 pixels.data(), pixels.size()));
    } else if (status == JXL_DEC_NEED_PREVIEW_OUT_BUFFER) {
      // Set preview pixel output buffer.
      size_t buffer_size;
      EXPECT_SUCCESS(
          JxlDecoderPreviewOutBufferSize(dec.get(), &format, &buffer_size));
      preview_pixels.resize(buffer_size);
      memset(preview_pixels.data(), 0, preview_pixels.size());
      EXPECT_SUCCESS(JxlDecoderSetPreviewOutBuffer(
          dec.get(), &format, preview_pixels.data(), preview_pixels.size()));
    } else if (status == JXL_DEC_FULL_IMAGE) {
      // Pixel output buffer is set.
      if (opts.pixel_prefix) {
        image.frames.emplace_back();
        swap(image.frames.back(), pixels);
      }

      // TODO(deymo): Get the extra channel pixel data an store it.
    } else if (status == JXL_DEC_PREVIEW_IMAGE) {
      // Preview pixel output buffer is set.
      if (opts.pixel_prefix && opts.preview) {
        preview.frames.emplace_back();
        swap(preview.frames.back(), preview_pixels);
      }
    } else if (status == JXL_DEC_SUCCESS) {
      break;
    } else {
      fprintf(stderr, "Error: unexpected status: %d\n",
              static_cast<int>(status));
      return false;
    }
  }

  if (opts.pixel_prefix) {
    std::string name = std::string(opts.pixel_prefix) + ".npy";
    EXPECT_TRUE(SaveNPYArray(name.c_str(), image));
  }

  if (opts.preview) {
    std::string name = std::string(opts.pixel_prefix) + "_preview.npy";
    EXPECT_TRUE(SaveNPYArray(name.c_str(), preview));
  }

  if (opts.jpeg_path) {
    size_t used_jpeg_output =
        jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec.get());
    jpeg_data.insert(jpeg_data.end(), jpeg_data_chunk.data(),
                     jpeg_data_chunk.data() + used_jpeg_output);
    EXPECT_TRUE(SaveFile(opts.jpeg_path, jpeg_data));
  }

  if (opts.metadata_path) {
    JSONDict meta;
#define METADATA(FIELD) meta.Add(#FIELD, info.FIELD)
    METADATA(xsize);
    METADATA(ysize);
    METADATA(uses_original_profile);
    METADATA(bits_per_sample);
    METADATA(exponent_bits_per_sample);
    METADATA(have_preview);
    if (info.have_preview) {
      auto* meta_preview = meta.AddEmpty<JSONDict>("preview");
      meta_preview->Add("xsize", info.preview.xsize);
      meta_preview->Add("ysize", info.preview.ysize);
    }
    METADATA(have_animation);
    if (info.have_animation) {
      auto* meta_animation = meta.AddEmpty<JSONDict>("animation");
      meta_animation->Add("tps_numerator", info.animation.tps_numerator);
      meta_animation->Add("tps_denominator", info.animation.tps_denominator);
      meta_animation->Add("num_loops", info.animation.num_loops);
      meta_animation->Add("have_timecodes", info.animation.have_timecodes);
    }
    METADATA(orientation);
    METADATA(num_extra_channels);
    METADATA(alpha_bits);
    if (info.alpha_bits > 0) {
      METADATA(alpha_exponent_bits);
      METADATA(alpha_premultiplied);
    }

    // Extra channels.
    auto* meta_channels = meta.AddEmpty<JSONArray>("extra_channels");
    for (uint32_t i = 0; i < info.num_extra_channels; i++) {
      auto* channel_i = meta_channels->AddEmpty<JSONDict>();

#define METADATA_CHANNEL(FIELD) \
  channel_i->Add(#FIELD, JSONValue(extra_channels[i].FIELD))

      // TODO(deymo): Make the type a string.
      METADATA_CHANNEL(type);
      METADATA_CHANNEL(bits_per_sample);
      METADATA_CHANNEL(exponent_bits_per_sample);
      METADATA_CHANNEL(dim_shift);
      channel_i->Add("name", JSONValue(extra_channel_names[i]));
      METADATA_CHANNEL(alpha_associated);
      METADATA_CHANNEL(cfa_channel);
      // TODO(deymo): Spot color.
    }

    // Frames.
    meta.Add("num_frames", JSONValue(frame_headers.size()));
    auto* meta_frames = meta.AddEmpty<JSONArray>("frames");
    for (size_t i = 0; i < frame_headers.size(); i++) {
      auto* frame_i = meta_frames->AddEmpty<JSONDict>();
      frame_i->Add("duration", JSONValue(frame_headers[i].duration));

      if (info.animation.have_timecodes) {
        frame_i->Add("timecode", JSONValue(frame_headers[i].timecode));
      }
      frame_i->Add("name", JSONValue(frame_names[i]));
    }

    std::ofstream ofs(opts.metadata_path);
    meta.Write(ofs, 0);
    ofs << std::endl;
    ofs.close();
    EXPECT_TRUE(ofs.good());
  }
  return true;
}

int Usage(const char* program) {
  fprintf(
      stderr,
      "Usage: %s INPUT_JXL [-i ORG_ICC] [-p PREFIX [-w]] [-m METADATA]\n"
      "\n"
      "  INPUT_JXL: Path to the input .jxl file.\n"
      "  -i ORG_ICC: Path to the output \"original\" ICC profile.\n"
      "  -p PREFIX: Prefix path to generate the pixel numpy data image (with\n"
      "       suffix \".npy\") and ICC profile (with suffix \".icc\"). The \n"
      "       image data will be a 4D numpy array with dimensions (number of \n"
      "       frames, height, width, number of channels).\n"
      "  -w: Generate a preview image as well with suffix \"_preview.npy\".\n"
      "       The preview numpy image will have 1 frame. Requires -p.\n"
      "  -j JPEG: Path to the output reconstructed JPEG file.\n"
      "  -m METADATA: Path to the output JSON text metadata file.\n",
      program);
  return 1;
}

}  // namespace

// Helper macro to check that an extra argument was passed to ARG.
#define EXPECT_ARG(ARG)                                    \
  if (optind >= argc) {                                    \
    fprintf(stderr, "%s needs an argument value.\n", ARG); \
    return Usage(argv[0]);                                 \
  }

int main(int argc, char* argv[]) {
  DecodeOptions opts;

  for (int optind = 1; optind < argc;) {
    if (!strcmp(argv[optind], "-i")) {
      // -i ORG_ICC
      optind++;
      EXPECT_ARG("-i");
      opts.icc_path = argv[optind++];
    } else if (!strcmp(argv[optind], "-p")) {
      optind++;
      EXPECT_ARG("-p");
      opts.pixel_prefix = argv[optind++];
    } else if (!strcmp(argv[optind], "-w")) {
      optind++;
      opts.preview = true;
    } else if (!strcmp(argv[optind], "-j")) {
      optind++;
      EXPECT_ARG("-j");
      opts.jpeg_path = argv[optind++];
    } else if (!strcmp(argv[optind], "-m")) {
      optind++;
      EXPECT_ARG("-m");
      opts.metadata_path = argv[optind++];
    } else if (opts.input == nullptr) {
      opts.input = argv[optind++];
    } else {
      fprintf(stderr, "Unknown parameter: \"%s\".\n", argv[optind]);
      return Usage(argv[0]);
    }
  }
  if (opts.preview && !opts.pixel_prefix) {
    fprintf(stderr, "-w parameter requires -p\n");
    return Usage(argv[0]);
  }
  if (!opts.input) {
    fprintf(stderr, "JXL decoder for conformance testing.\n");
    return Usage(argv[0]);
  }

  return DecodeJXL(opts) ? 0 : 1;
}
