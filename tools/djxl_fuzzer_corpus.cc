// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>
#include <jxl/types.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

#include "lib/extras/packed_image.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/modular/options.h"
#include "tools/no_memory_manager.h"
#if defined(_WIN32) || defined(_WIN64)
#include "third_party/dirent.h"
#else
#endif

#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

#include "lib/extras/enc/encode.h"
#include "lib/extras/enc/jpg.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/jpeg/enc_jpeg_data.h"
#include "lib/jxl/test_utils.h"  // TODO(eustas): cut this dependency
#include "tools/file_io.h"
#include "tools/thread_pool_internal.h"

namespace {

const size_t kMaxWidth = 50000;
const size_t kMaxHeight = 50000;
const size_t kMaxPixels = 20 * (1 << 20);  // 20 MP
const size_t kMaxBitDepth = 24;  // The maximum reasonable bit depth supported.

std::mutex stderr_mutex;

typedef std::function<uint8_t()> PixelGenerator;

// ImageSpec needs to be a packed struct to allow us to use the raw memory of
// the struct for hashing to create a consistent.
#pragma pack(push, 1)
struct ImageSpec {
  bool Validate() const {
    if (width > kMaxWidth || height > kMaxHeight ||
        width * height > kMaxPixels) {
      return false;
    }
    if (bit_depth > kMaxBitDepth || bit_depth == 0) return false;
    if (num_frames == 0) return false;
    // JPEG doesn't support all formats, so reconstructible JPEG isn't always
    // valid.
    if (is_reconstructible_jpeg && (bit_depth != 8 || num_channels != 3 ||
                                    alpha_bit_depth != 0 || num_frames != 1))
      return false;
    return true;
  }

  friend std::ostream& operator<<(std::ostream& o, const ImageSpec& spec) {
    o << "ImageSpec<"
      << "size=" << spec.width << "x" << spec.height
      << " * chan=" << spec.num_channels << " depth=" << spec.bit_depth
      << " alpha=" << spec.alpha_bit_depth
      << " (premult=" << spec.alpha_is_premultiplied
      << ") x frames=" << spec.num_frames << " seed=" << spec.seed
      << ", speed=" << static_cast<int>(spec.params.speed_tier)
      << ", butteraugli=" << spec.params.butteraugli_distance
      << ", modular_mode=" << spec.params.modular_mode
      << ", lossy_palette=" << spec.params.lossy_palette
      << ", noise=" << spec.params.noise << ", preview=" << spec.params.preview
      << ", fuzzer_friendly=" << spec.fuzzer_friendly
      << ", is_reconstructible_jpeg=" << spec.is_reconstructible_jpeg
      << ", orientation=" << static_cast<int>(spec.orientation) << ">";
    return o;
  }

  void SpecHash(uint8_t hash[16]) const {
    const uint8_t* from = reinterpret_cast<const uint8_t*>(this);
    std::seed_seq hasher(from, from + sizeof(*this));
    uint32_t* to = reinterpret_cast<uint32_t*>(hash);
    hasher.generate(to, to + 4);
  }

  uint64_t width = 256;
  uint64_t height = 256;
  // Number of channels *not* including alpha.
  uint64_t num_channels = 3;
  uint64_t bit_depth = 8;
  // Bit depth for the alpha channel. A value of 0 means no alpha channel.
  uint64_t alpha_bit_depth = 8;
  int32_t alpha_is_premultiplied = JXL_FALSE;

  // Whether the ANS fuzzer friendly setting is currently enabled.
  uint32_t fuzzer_friendly = JXL_FALSE;

  // Number of frames, all the frames will have the same size.
  uint64_t num_frames = 1;

  // The seed for the PRNG.
  uint32_t seed = 7777;

  // Flags used for compression. These are mapped to the CompressedParams.
  struct CjxlParams {
    float butteraugli_distance = 1.f;
    // Must not use Weighted - see force_no_wp
    jxl::Predictor modular_predictor = jxl::Predictor::Gradient;
    jxl::ColorTransform color_transform = jxl::ColorTransform::kXYB;
    jxl::SpeedTier speed_tier = jxl::SpeedTier::kTortoise;
    bool modular_mode = false;
    bool lossy_palette = false;
    bool noise = false;
    bool preview = false;
    // CjxlParams is packed; re-add padding when sum of sizes of members is not
    // multiple of 4.
    // uint8_t padding_[0] = {};
  } params;

  uint32_t is_reconstructible_jpeg = JXL_FALSE;
  // Use 0xFFFFFFFF if any random spec is good; otherwise set the desired value.
  uint32_t override_decoder_spec = 0xFFFFFFFF;
  // Orientation.
  uint8_t orientation = 0;
  uint8_t padding_[3] = {};
};
#pragma pack(pop)
static_assert(sizeof(ImageSpec) % 4 == 0, "Add padding to ImageSpec.");

bool GenerateFile(const char* output_dir, const ImageSpec& spec,
                  bool regenerate, bool quiet) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  // Compute a checksum of the ImageSpec to name the file. This is just to keep
  // the output of this program repeatable.
  uint8_t checksum[16];
  spec.SpecHash(checksum);
  std::string hash_str(sizeof(checksum) * 2, ' ');
  static const char* hex_chars = "0123456789abcdef";
  for (size_t i = 0; i < sizeof(checksum); i++) {
    hash_str[2 * i] = hex_chars[checksum[i] >> 4];
    hash_str[2 * i + 1] = hex_chars[checksum[i] % 0x0f];
  }
  std::string output_fn = std::string(output_dir) + "/" + hash_str + ".jxl";

  // Don't regenerate files if they already exist on disk to speed-up
  // consecutive calls when --regenerate is not used.
  struct stat st;
  if (!regenerate && stat(output_fn.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
    return true;
  }

  if (!quiet) {
    std::unique_lock<std::mutex> lock(stderr_mutex);
    std::cerr << "Generating " << spec << " as " << hash_str << "\n";
  }

  jxl::CodecInOut io{memory_manager};
  if (spec.bit_depth == 32) {
    io.metadata.m.SetFloat32Samples();
  } else {
    io.metadata.m.SetUintSamples(spec.bit_depth);
  }
  io.metadata.m.SetAlphaBits(spec.alpha_bit_depth,
                             FROM_JXL_BOOL(spec.alpha_is_premultiplied));
  io.metadata.m.orientation = spec.orientation;
  io.frames.clear();
  io.frames.reserve(spec.num_frames);

  jxl::ColorEncoding c;
  if (spec.num_channels == 1) {
    c = jxl::ColorEncoding::LinearSRGB(true);
  } else if (spec.num_channels == 3) {
    c = jxl::ColorEncoding::SRGB();
  }

  uint8_t hash[16];
  spec.SpecHash(hash);
  std::mt19937 mt(spec.seed);

  // Compress the image.
  jxl::PaddedBytes compressed{memory_manager};

  std::uniform_int_distribution<> dis(1, 6);
  PixelGenerator gen = [&]() -> uint8_t { return dis(mt); };

  jxl::extras::PackedPixelFile ppf;
  ppf.info.xsize = spec.width;
  ppf.info.ysize = spec.height;
  ppf.info.num_color_channels = spec.num_channels ? 1 : 3;
  ppf.info.bits_per_sample = spec.bit_depth;
  for (uint32_t frame = 0; frame < spec.num_frames; frame++) {
    jxl::ImageBundle ib(memory_manager, &io.metadata.m);
    const bool has_alpha = (spec.alpha_bit_depth != 0);
    const int alpha_channels = (has_alpha ? 1 : 0);
    const size_t bytes_per_sample =
        jxl::DivCeil(io.metadata.m.bit_depth.bits_per_sample, 8);
    const size_t bytes_per_pixel =
        bytes_per_sample *
        (io.metadata.m.color_encoding.Channels() + alpha_channels);
    const size_t row_size = spec.width * bytes_per_pixel;
    std::vector<uint8_t> img_data(row_size * spec.height, 0);
    for (size_t y = 0; y < spec.height; y++) {
      size_t pos = row_size * y;
      for (size_t x = 0; x < spec.width; x++) {
        for (size_t b = 0; b < bytes_per_pixel; b++) {
          img_data[pos++] = gen();
        }
      }
    }
    uint32_t num_channels = bytes_per_pixel / bytes_per_sample;
    JxlDataType data_type =
        bytes_per_sample == 1 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16;
    JxlPixelFormat format = {num_channels, data_type, JXL_LITTLE_ENDIAN, 0};
    const jxl::Span<const uint8_t> span(img_data.data(), img_data.size());
    JXL_RETURN_IF_ERROR(ConvertFromExternal(
        span, spec.width, spec.height, io.metadata.m.color_encoding,
        io.metadata.m.bit_depth.bits_per_sample, format, nullptr, &ib));
    io.frames.push_back(std::move(ib));
    JXL_ASSIGN_OR_RETURN(
        jxl::extras::PackedFrame packed_frame,
        jxl::extras::PackedFrame::Create(spec.width, spec.height, format));
    JXL_ENSURE(packed_frame.color.pixels_size == img_data.size());
    memcpy(packed_frame.color.pixels(0, 0, 0), img_data.data(),
           img_data.size());
    ppf.frames.emplace_back(std::move(packed_frame));
  }

  jxl::CompressParams params;
  params.speed_tier = spec.params.speed_tier;

  if (spec.is_reconstructible_jpeg) {
    // If this image is supposed to be a reconstructible JPEG, collect the JPEG
    // metadata and encode it in the beginning of the compressed bytes.
    std::vector<uint8_t> jpeg_bytes;
    io.jpeg_quality = 70;
    auto encoder = jxl::extras::GetJPEGEncoder();
    encoder->SetOption("quality", "70");
    jxl::extras::EncodedImage encoded;
    JXL_RETURN_IF_ERROR(encoder->Encode(ppf, &encoded, nullptr));
    jpeg_bytes = encoded.bitstreams[0];
    JXL_RETURN_IF_ERROR(jxl::jpeg::DecodeImageJPG(
        jxl::Bytes(jpeg_bytes.data(), jpeg_bytes.size()), &io));
    std::vector<uint8_t> jpeg_data;
    JXL_RETURN_IF_ERROR(EncodeJPEGData(memory_manager, *io.Main().jpeg_data,
                                       &jpeg_data, params));
    std::vector<uint8_t> header;
    header.insert(header.end(), jxl::kContainerHeader.begin(),
                  jxl::kContainerHeader.end());
    jxl::AppendBoxHeader(jxl::MakeBoxType("jbrd"), jpeg_data.size(), false,
                         &header);
    jxl::Bytes(jpeg_data).AppendTo(header);
    jxl::AppendBoxHeader(jxl::MakeBoxType("jxlc"), 0, true, &header);
    JXL_RETURN_IF_ERROR(compressed.append(header));
  }

  params.modular_mode = spec.params.modular_mode;
  params.color_transform = spec.params.color_transform;
  params.butteraugli_distance = spec.params.butteraugli_distance;
  params.options.predictor = {spec.params.modular_predictor};
  params.lossy_palette = spec.params.lossy_palette;
  if (spec.params.preview) params.preview = jxl::Override::kOn;
  if (spec.params.noise) params.noise = jxl::Override::kOn;

  // EncodeFile replaces output; pass a temporary storage for it.
  std::vector<uint8_t> compressed_image;
  bool ok = jxl::test::EncodeFile(params, &io, &compressed_image);
  if (!ok) return false;
  JXL_RETURN_IF_ERROR(compressed.append(compressed_image));

  // Append 4 bytes with the flags used by djxl_fuzzer to select the decoding
  // output.
  std::uniform_int_distribution<> dis256(0, 255);
  if (spec.override_decoder_spec == 0xFFFFFFFF) {
    for (size_t i = 0; i < 4; ++i) {
      JXL_RETURN_IF_ERROR(compressed.push_back(dis256(mt)));
    }
  } else {
    for (size_t i = 0; i < 4; ++i) {
      JXL_RETURN_IF_ERROR(
          compressed.push_back(spec.override_decoder_spec >> (8 * i)));
    }
  }

  if (!jpegxl::tools::WriteFile(output_fn, compressed)) return false;
  if (!quiet) {
    std::unique_lock<std::mutex> lock(stderr_mutex);
    std::cerr << "Stored " << output_fn << " size: " << compressed.size()
              << "\n";
  }

  return true;
}

std::vector<ImageSpec::CjxlParams> CompressParamsList() {
  std::vector<ImageSpec::CjxlParams> ret;

  {
    ImageSpec::CjxlParams params;
    params.butteraugli_distance = 1.5;
    ret.push_back(params);
  }

  {
    // Lossless
    ImageSpec::CjxlParams params;
    params.modular_mode = true;
    params.color_transform = jxl::ColorTransform::kNone;
    params.butteraugli_distance = 0.f;
    params.modular_predictor = {jxl::Predictor::Weighted};
    ret.push_back(params);
  }

  return ret;
}

void Usage() {
  fprintf(stderr,
          "Use: fuzzer_corpus [-r] [-q] [-j THREADS] [output_dir]\n"
          "\n"
          "  -r Regenerate files if already exist.\n"
          "  -q Be quiet.\n"
          "  -j THREADS Number of parallel jobs to run.\n");
}

}  // namespace

int main(int argc, const char** argv) {
  const char* dest_dir = nullptr;
  bool regenerate = false;
  bool quiet = false;
  size_t num_threads = std::thread::hardware_concurrency();
  for (int optind = 1; optind < argc;) {
    if (!strcmp(argv[optind], "-r")) {
      regenerate = true;
      optind++;
    } else if (!strcmp(argv[optind], "-q")) {
      quiet = true;
      optind++;
    } else if (!strcmp(argv[optind], "-j")) {
      optind++;
      if (optind < argc) {
        num_threads = atoi(argv[optind++]);
      } else {
        fprintf(stderr, "-j needs an argument value.\n");
        Usage();
        return 1;
      }
    } else if (dest_dir == nullptr) {
      dest_dir = argv[optind++];
    } else {
      fprintf(stderr, "Unknown parameter: \"%s\".\n", argv[optind]);
      Usage();
      return 1;
    }
  }
  if (!dest_dir) {
    dest_dir = "corpus";
  }

  struct stat st;
  memset(&st, 0, sizeof(st));
  if (stat(dest_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
    fprintf(stderr, "Output path \"%s\" is not a directory.\n", dest_dir);
    Usage();
    return 1;
  }

  // Create the corpus directory if doesn't already exist.
  std::mt19937 mt(77777);

  std::vector<std::pair<uint32_t, uint32_t>> image_sizes = {
      {8, 8},
      {32, 32},
      {128, 128},
      // Degenerated cases.
      {10000, 1},
      {10000, 2},
      {1, 10000},
      {2, 10000},
      // Large case.
      {555, 256},
      {257, 513},
  };
  const std::vector<ImageSpec::CjxlParams> params_list = CompressParamsList();

  ImageSpec spec;
  // The ans_fuzzer_friendly setting is not thread safe and therefore done in
  // an outer loop. This determines whether to use fuzzer-friendly ANS encoding.
  for (bool fuzzer_friendly : {false, true}) {
    jxl::SetANSFuzzerFriendly(fuzzer_friendly);
    spec.fuzzer_friendly = TO_JXL_BOOL(fuzzer_friendly);

    std::vector<ImageSpec> specs;
    for (auto img_size : image_sizes) {
      spec.width = img_size.first;
      spec.height = img_size.second;
      for (uint32_t bit_depth : {1, 2, 8, 16}) {
        spec.bit_depth = bit_depth;
        for (uint32_t num_channels : {1, 3}) {
          spec.num_channels = num_channels;
          for (uint32_t alpha_bit_depth : {0, 8, 16}) {
            spec.alpha_bit_depth = alpha_bit_depth;
            if (bit_depth == 16 && alpha_bit_depth == 8) {
              // This mode is not supported in CopyTo().
              continue;
            }
            for (uint32_t num_frames : {1, 3}) {
              spec.num_frames = num_frames;
              for (bool preview : {false, true}) {
                for (JXL_BOOL reconstructible_jpeg : {JXL_FALSE, JXL_TRUE}) {
                  spec.is_reconstructible_jpeg = reconstructible_jpeg;
                  for (const auto& params : params_list) {
                    spec.params = params;

                    spec.params.preview = preview;
                    if (alpha_bit_depth) {
                      spec.alpha_is_premultiplied = mt() % 2;
                    }
                    if (spec.width * spec.height > 1000) {
                      // Increase the encoder speed for larger images.
                      spec.params.speed_tier = jxl::SpeedTier::kWombat;
                    }
                    spec.seed = mt() % 777777;
                    // Pick the orientation at random. It is orthogonal to all
                    // other features. Valid values are 1 to 8.
                    spec.orientation = 1 + (mt() % 8);
                    if (!spec.Validate()) {
                      if (!quiet) {
                        std::cerr << "Skipping " << spec << "\n";
                      }
                    } else {
                      specs.push_back(spec);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    specs.emplace_back();
    specs.back().params.lossy_palette = true;
    specs.back().override_decoder_spec = 0;

    specs.emplace_back();
    specs.back().params.noise = true;
    specs.back().override_decoder_spec = 0;

    jpegxl::tools::ThreadPoolInternal pool{num_threads};
    const auto generate = [&specs, dest_dir, regenerate, quiet](
                              const uint32_t task,
                              size_t /* thread */) -> jxl::Status {
      const ImageSpec& spec = specs[task];
      GenerateFile(dest_dir, spec, regenerate, quiet);
      return true;
    };
    if (!RunOnPool(pool.get(), 0, specs.size(), jxl::ThreadPool::NoInit,
                   generate, "FuzzerCorpus")) {
      std::cerr << "Error generating fuzzer corpus\n";
      return 1;
    }
  }
  std::cerr << "Finished generating fuzzer corpus\n";
  return 0;
}
