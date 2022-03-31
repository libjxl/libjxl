
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
  // amount of color channels: 1 for grayscale, 3 for RGB
  uint32_t num_color_channels;
  // amount of extra channels, including alpha channels, spot colors, ...
  uint32_t num_extra_channels;

  // Both frames and ec_frames are filled in by the JXL decoder, and will be
  // converted into a numpy array of the form (frame, ysize, xsize, channel)

  // Array of the color channels of the frames. This is an array of frames,
  // where each frame is an array of pixels. The pixels in a frame are laid
  // out per scanline, then per channel, and finally individual pixels as
  // little endian 32-bit floating point.
  std::vector<std::vector<uint8_t>> frames;

  // Array of the extra channels of the frames. This is an array of frames,
  // where each frame is an array of extra channels. The pixels in an extra
  // channel are laid out per scanline, then individual pixels as
  // little endian 32-bit floating point.
  std::vector<std::vector<std::vector<uint8_t>>> ec_frames;
};

// Saves an ImageArray as a numpy 4D ndarray in binary format.
bool SaveNPYArray(const char* filename, const ImageArray& arr) {
  size_t image_size =
      sizeof(float) * arr.xsize * arr.ysize * arr.num_color_channels;
  size_t ec_size = sizeof(float) * arr.xsize * arr.ysize;
  for (const auto& frame : arr.frames) {
    if (frame.size() != image_size) {
      fprintf(stderr, "Invalid frame size\n");
      return false;
    }
  }
  for (const auto& frame : arr.ec_frames) {
    if (frame.size() != arr.num_extra_channels) {
      fprintf(stderr, "Invalid extra channel count\n");
      return false;
    }
    for (const auto& ch : frame) {
      if (ch.size() != ec_size) {
        fprintf(stderr, "Invalid extra channel size\n");
        return false;
      }
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
    uint32_t num_channels = arr.num_color_channels + arr.num_extra_channels;
    std::stringstream ss;
    ss << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
       << arr.frames.size() << ", " << arr.ysize << ", " << arr.xsize << ", "
       << num_channels << "), }\n";
    // 16-bit little endian header length.
    uint8_t header_len[2] = {static_cast<uint8_t>(ss.str().size() % 256),
                             static_cast<uint8_t>(ss.str().size() / 256)};
    WRITE_TO_FILE(header_len, 2);
    WRITE_TO_FILE(ss.str().data(), ss.str().size());
  }

  // interleave the samples from color and extra channels
  for (size_t f = 0; f < arr.frames.size(); ++f) {
    size_t pos = 0;
    for (size_t y = 0; y < arr.ysize; ++y) {
      for (size_t x = 0; x < arr.xsize; ++x, pos += sizeof(float)) {
        WRITE_TO_FILE(arr.frames[f].data() + pos * arr.num_color_channels,
                      arr.num_color_channels * sizeof(float));
        for (size_t i = 0; i < arr.num_extra_channels; i++) {
          WRITE_TO_FILE(arr.ec_frames[f][i].data() + pos, sizeof(float));
        }
      }
    }
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

// TODO(veluca): merge this back in DecodeJXL once/if the API supports decoding
// to JPEG and to pixels at the same time.
bool DecodeJXLToJpeg(const char* input_path, const char* output_path) {
  // JPEG output buffer when reconstructing a JPEG file.
  std::vector<uint8_t> jpeg_data;
  std::vector<uint8_t> jpeg_data_chunk(16 * 1024);
  auto dec = JxlDecoderMake(nullptr);

  uint32_t events = JXL_DEC_JPEG_RECONSTRUCTION | JXL_DEC_FULL_IMAGE;
  EXPECT_SUCCESS(JxlDecoderSubscribeEvents(dec.get(), events));

  // TODO(deymo): Consider using a multi-threading decoder for conformance
  // testing as well.

  // Load and set input all at oncee.
  std::vector<uint8_t> jxl_input;
  EXPECT_TRUE(LoadFile(input_path, &jxl_input));
  EXPECT_SUCCESS(
      JxlDecoderSetInput(dec.get(), jxl_input.data(), jxl_input.size()));

  bool has_jpeg_reconstruction = false;

  while (true) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Error decoding.\n");
      return false;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      fprintf(stderr, "Error decoding: expected more input.\n");
      return false;
    } else if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
      has_jpeg_reconstruction = true;
      // Decoding to JPEG.
      EXPECT_SUCCESS(JxlDecoderSetJPEGBuffer(dec.get(), jpeg_data_chunk.data(),
                                             jpeg_data_chunk.size()));
    } else if (status == JXL_DEC_JPEG_NEED_MORE_OUTPUT) {
      // Decoded a chunk to JPEG.
      size_t used_jpeg_output =
          jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec.get());
      jpeg_data.insert(jpeg_data.end(), jpeg_data_chunk.data(),
                       jpeg_data_chunk.data() + used_jpeg_output);
      if (used_jpeg_output == 0) {
        // Chunk is too small.
        jpeg_data_chunk.resize(jpeg_data_chunk.size() * 2);
      }
      EXPECT_SUCCESS(JxlDecoderSetJPEGBuffer(dec.get(), jpeg_data_chunk.data(),
                                             jpeg_data_chunk.size()));
    } else if (status == JXL_DEC_SUCCESS) {
      break;
    } else if (status == JXL_DEC_FULL_IMAGE) {
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      return true;
    } else {
      fprintf(stderr, "Error: unexpected status: %d\n",
              static_cast<int>(status));
      return false;
    }
  }

  if (has_jpeg_reconstruction) {
    size_t used_jpeg_output =
        jpeg_data_chunk.size() - JxlDecoderReleaseJPEGBuffer(dec.get());
    jpeg_data.insert(jpeg_data.end(), jpeg_data_chunk.data(),
                     jpeg_data_chunk.data() + used_jpeg_output);
    EXPECT_TRUE(SaveFile(output_path, jpeg_data));
  }
  return true;
}

bool DecodeJXL(const DecodeOptions& opts) {
  auto dec = JxlDecoderMake(nullptr);

  uint32_t events =
      JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_PREVIEW_IMAGE;
  if (opts.pixel_prefix) events |= JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE;
  // We need to output the frame header info in the metadata.
  if (opts.metadata_path) events |= JXL_DEC_FRAME;

  if (opts.jpeg_path) {
    EXPECT_TRUE(DecodeJXLToJpeg(opts.input, opts.jpeg_path));
  }

  EXPECT_SUCCESS(JxlDecoderSubscribeEvents(dec.get(), events));
  EXPECT_SUCCESS(JxlDecoderSetRenderSpotcolors(dec.get(), JXL_FALSE));

  // TODO(deymo): Consider using a multi-threading decoder for conformance
  // testing as well.

  // Load and set input all at oncee.
  std::vector<uint8_t> jxl_input;
  EXPECT_TRUE(LoadFile(opts.input, &jxl_input));
  EXPECT_SUCCESS(
      JxlDecoderSetInput(dec.get(), jxl_input.data(), jxl_input.size()));

  JxlBasicInfo info{};

  // Pixel data when decoding a frame or a preview frame.
  std::vector<uint8_t> pixels;
  std::vector<uint8_t> preview_pixels;

  std::vector<JxlExtraChannelInfo> extra_channels;
  std::vector<std::vector<uint8_t>> extra_channel_pixels;

  std::vector<JxlFrameHeader> frame_headers;
  std::vector<std::string> frame_names;

  JxlPixelFormat format;

  ImageArray image, preview;

  while (true) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Error decoding.\n");
      return false;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      fprintf(stderr, "Error decoding: expected more input.\n");
      return false;
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
      }

      // Select the output pixel format based on the basic info.
      format = JxlPixelFormat{info.num_color_channels, JXL_TYPE_FLOAT,
                              JXL_LITTLE_ENDIAN, 0};
      image.num_color_channels = info.num_color_channels;
      image.num_extra_channels = info.num_extra_channels;
      image.xsize = info.xsize;
      image.ysize = info.ysize;

      if (info.have_preview) {
        preview.num_color_channels = info.num_color_channels;
        preview.num_extra_channels = info.num_extra_channels;
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
      extra_channel_pixels.resize(info.num_extra_channels);
      for (uint32_t i = 0; i < info.num_extra_channels; ++i) {
        EXPECT_SUCCESS(JxlDecoderExtraChannelBufferSize(dec.get(), &format,
                                                        &buffer_size, i));
        extra_channel_pixels[i].resize(buffer_size);
        memset(extra_channel_pixels[i].data(), 0,
               extra_channel_pixels[i].size());
        EXPECT_SUCCESS(JxlDecoderSetExtraChannelBuffer(
            dec.get(), &format, extra_channel_pixels[i].data(),
            extra_channel_pixels[i].size(), i));
      }
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
        image.ec_frames.emplace_back();
        for (uint32_t i = 0; i < info.num_extra_channels; ++i) {
          image.ec_frames.back().emplace_back();
          swap(image.ec_frames.back().back(), extra_channel_pixels[i]);
        }
      }

      // TODO(deymo): Get the extra channel pixel data an store it.
    } else if (status == JXL_DEC_PREVIEW_IMAGE) {
      // Preview pixel output buffer is set.
      if (opts.pixel_prefix && info.have_preview) {
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
    std::string name = std::string(opts.pixel_prefix) + "_image.npy";
    EXPECT_TRUE(SaveNPYArray(name.c_str(), image));
  }

  if (opts.pixel_prefix && info.have_preview) {
    std::string name = std::string(opts.pixel_prefix) + "_preview.npy";
    EXPECT_TRUE(SaveNPYArray(name.c_str(), preview));
  }

  if (opts.metadata_path) {
    JSONDict meta;
    // Same order as in 18181-3 CD.

    // Frames.
    auto* meta_frames = meta.AddEmpty<JSONArray>("frames");
    for (size_t i = 0; i < frame_headers.size(); i++) {
      auto* frame_i = meta_frames->AddEmpty<JSONDict>();
      if (info.have_animation) {
        frame_i->Add("duration", JSONValue(frame_headers[i].duration * 1.0f *
                                           info.animation.tps_denominator /
                                           info.animation.tps_numerator));
      }

      frame_i->Add("name", JSONValue(frame_names[i]));

      if (info.animation.have_timecodes) {
        frame_i->Add("timecode", JSONValue(frame_headers[i].timecode));
      }
    }

#define METADATA(FIELD) meta.Add(#FIELD, info.FIELD)

    METADATA(intensity_target);
    METADATA(min_nits);
    METADATA(relative_to_max_display);
    METADATA(linear_below);

    if (info.have_preview) {
      meta.AddEmpty<JSONDict>("preview");
      // TODO(veluca): can we have duration/name/timecode here?
    }

    {
      auto ectype = meta.AddEmpty<JSONArray>("extra_channel_type");
      auto bps = meta.AddEmpty<JSONArray>("bits_per_sample");
      auto ebps = meta.AddEmpty<JSONArray>("exp_bits_per_sample");
      bps->Add(info.bits_per_sample);
      ebps->Add(info.exponent_bits_per_sample);
      for (size_t i = 0; i < extra_channels.size(); i++) {
        switch (extra_channels[i].type) {
          case JXL_CHANNEL_ALPHA: {
            ectype->Add(std::string("Alpha"));
            break;
          }
          case JXL_CHANNEL_DEPTH: {
            ectype->Add(std::string("Depth"));
            break;
          }
          case JXL_CHANNEL_SPOT_COLOR: {
            ectype->Add(std::string("SpotColor"));
            break;
          }
          case JXL_CHANNEL_SELECTION_MASK: {
            ectype->Add(std::string("SelectionMask"));
            break;
          }
          case JXL_CHANNEL_BLACK: {
            ectype->Add(std::string("Black"));
            break;
          }
          case JXL_CHANNEL_CFA: {
            ectype->Add(std::string("CFA"));
            break;
          }
          case JXL_CHANNEL_THERMAL: {
            ectype->Add(std::string("Thermal"));
            break;
          }
          default: {
            ectype->Add(std::string("UNKNOWN"));
            break;
          }
        }
        bps->Add(extra_channels[i].bits_per_sample);
        ebps->Add(extra_channels[i].exponent_bits_per_sample);
      }
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
      "Usage: %s INPUT_JXL [-i ORG_ICC] [-p PREFIX] [-m METADATA]\n"
      "\n"
      "  INPUT_JXL: Path to the input .jxl file.\n"
      "  -i ORG_ICC: Path to the output \"original\" ICC profile.\n"
      "  -p PREFIX: Prefix path to generate the pixel numpy data image (with\n"
      "       suffix \".npy\") and ICC profile (with suffix \".icc\"). The \n"
      "       image data will be a 4D numpy array with dimensions (number of \n"
      "       frames, height, width, number of channels).\n"
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
      optind++;
      EXPECT_ARG("-i");
      opts.icc_path = argv[optind++];
    } else if (!strcmp(argv[optind], "-p")) {
      optind++;
      EXPECT_ARG("-p");
      opts.pixel_prefix = argv[optind++];
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
  if (!opts.input) {
    fprintf(stderr, "JXL decoder for conformance testing.\n");
    return Usage(argv[0]);
  }

  return DecodeJXL(opts) ? 0 : 1;
}
