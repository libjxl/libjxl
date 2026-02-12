// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This example prints information from the main codestream header.

#include <jxl/color_encoding.h>
#include <jxl/compressed_icc.h>
#include <jxl/decode.h>
#include <jxl/gain_map.h>

#include <array>
#include <cinttypes>  // PRIu64
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

template <size_t N, typename E>
static const char* EnumToString(E e, const std::array<const char*, N>& names) {
  size_t idx = static_cast<size_t>(e);
  if (idx >= N || names[idx] == nullptr) return "ERROR";
  return names[idx];
}

static void PrintColorEncoding(const char* padding,
                               const JxlColorEncoding* color_encoding) {
  static const std::array<const char*, 4> cs_names = {"RGB", "Grayscale", "XYB",
                                                      "Unknown"};
  static const std::array<const char*, 12> wp_names = {
      nullptr, "D65",   "Custom", nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,  nullptr, "E",     "P3"};
  static const std::array<const char*, 12> pr_names = {
      nullptr, "sRGB",  "Custom", nullptr,    nullptr, nullptr,
      nullptr, nullptr, nullptr,  "Rec.2100", nullptr, "P3"};
  static const std::array<const char*, 19> tf_names = {
      nullptr, "709",    "Unknown", nullptr, nullptr, nullptr, nullptr,
      nullptr, "Linear", nullptr,   nullptr, nullptr, nullptr, "sRGB",
      "",      "",       "PQ",      "DCI",   "HLG"};
  static const std::array<const char*, 4> ri_names = {"Perceptual", "Relative",
                                                      "Saturation", "Absolute"};

  const auto& color_space = color_encoding->color_space;
  const auto& white_point = color_encoding->white_point;
  const auto& primaries = color_encoding->primaries;
  const auto& tf = color_encoding->transfer_function;

  const auto printXy = [](const char* prefix, const double xy[2]) {
    printf("%s(x=%f,y=%f)", prefix, xy[0], xy[1]);
  };

  printf("%sColor space: %s\n", padding, EnumToString(color_space, cs_names));

  printf("%sWhite point: %s", padding, EnumToString(white_point, wp_names));
  if (white_point == JXL_WHITE_POINT_CUSTOM) {
    printXy(" ", color_encoding->white_point_xy);
  }
  printf("\n");

  if (color_space == JXL_COLOR_SPACE_RGB ||
      color_space == JXL_COLOR_SPACE_UNKNOWN) {
    printf("%sPrimaries: %s", padding, EnumToString(primaries, pr_names));
    if (primaries == JXL_PRIMARIES_CUSTOM) {
      printXy(" - red", color_encoding->primaries_red_xy);
      printXy(", green", color_encoding->primaries_green_xy);
      printXy(", blue", color_encoding->primaries_blue_xy);
    }
    printf("\n");
  }

  printf("%sTransfer function: ", padding);
  if (tf == JXL_TRANSFER_FUNCTION_GAMMA) {
    printf("gamma(%f)", color_encoding->gamma);
  } else {
    printf("%s", EnumToString(tf, tf_names));
  }
  printf("\n");

  printf("%sRendering intent: %s\n", padding,
         EnumToString(color_encoding->rendering_intent, ri_names));
}

static int PrintBasicInfo(FILE* file, int verbose) {
  uint8_t* data = nullptr;
  size_t data_size = 0;
  // In how large chunks to read from the file and try decoding the basic info.
  const size_t chunk_size = 2048;
  uint8_t* box_data = nullptr;
  size_t box_size = 0;
  size_t box_index = 0;
  JxlBoxType box_type = {0};

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  if (!dec) {
    fprintf(stderr, "JxlDecoderCreate failed\n");
    return 0;
  }

  JxlDecoderSetKeepOrientation(dec, 1);
  JxlDecoderSetCoalescing(dec, JXL_FALSE);

  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(
          dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME |
                   JXL_DEC_BOX | JXL_DEC_BOX_COMPLETE)) {
    fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
    JxlDecoderDestroy(dec);
    return 0;
  }

  JxlBasicInfo info;
  int seen_basic_info = 0;
  JxlFrameHeader frame_header;
  int framecount = 0;
  float total_duration = 0.f;

  for (;;) {
    // The first time, this will output JXL_DEC_NEED_MORE_INPUT because no
    // input is set yet, this is ok since the input is set when handling this
    // event.
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);

    if (status == JXL_DEC_ERROR) {
      fprintf(stderr, "Decoder error\n");
      break;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      // The first time there is nothing to release and it returns 0, but that
      // is ok.
      size_t remaining = JxlDecoderReleaseInput(dec);
      // move any remaining bytes to the front if necessary
      if (remaining != 0) {
        memmove(data, data + data_size - remaining, remaining);
      }
      // resize the buffer to append one more chunk of data
      // TODO(lode): avoid unnecessary reallocations
      uint8_t* temp =
          reinterpret_cast<uint8_t*>(realloc(data, remaining + chunk_size));
      if (temp == nullptr) {
        fprintf(stderr, "Memory reallocation failed\n");
        break;
      }
      data = temp;
      // append bytes read from the file behind the remaining bytes
      size_t read_size = fread(data + remaining, 1, chunk_size, file);
      if (read_size == 0 && feof(file)) {
        fprintf(stderr, "Unexpected EOF\n");
        break;
      }
      data_size = remaining + read_size;
      JxlDecoderSetInput(dec, data, data_size);
      if (feof(file)) JxlDecoderCloseInput(dec);
    } else if (status == JXL_DEC_SUCCESS) {
      // Finished all processing.
      break;
    } else if (status == JXL_DEC_BASIC_INFO) {
      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec, &info)) {
        fprintf(stderr, "JxlDecoderGetBasicInfo failed\n");
        break;
      }

      seen_basic_info = 1;

      printf("JPEG XL %s, %ux%u, %s",
             info.have_animation ? "animation" : "image", info.xsize,
             info.ysize,
             info.uses_original_profile ? "(possibly) lossless" : "lossy");
      printf(", %d-bit ", info.bits_per_sample);
      if (info.exponent_bits_per_sample) {
        printf("float (%d exponent bits) ", info.exponent_bits_per_sample);
      }
      int cmyk = 0;
      const char* const ec_type_names[] = {
          "Alpha",     "Depth",     "Spotcolor", "Selection", "Black",
          "CFA",       "Thermal",   "Reserved0", "Reserved1", "Reserved2",
          "Reserved3", "Reserved4", "Reserved5", "Reserved6", "Reserved7",
          "Unknown",   "Optional"};
      const size_t ec_type_names_size =
          sizeof(ec_type_names) / sizeof(ec_type_names[0]);
      for (uint32_t i = 0; i < info.num_extra_channels; i++) {
        JxlExtraChannelInfo extra;
        if (JXL_DEC_SUCCESS != JxlDecoderGetExtraChannelInfo(dec, i, &extra)) {
          fprintf(stderr, "JxlDecoderGetExtraChannelInfo failed\n");
          break;
        }
        if (extra.type == JXL_CHANNEL_BLACK) cmyk = 1;
      }
      if (info.num_color_channels == 1) {
        printf("Grayscale");
      } else {
        if (cmyk) {
          printf("CMY");
        } else {
          printf("RGB");
        }
      }
      for (uint32_t i = 0; i < info.num_extra_channels; i++) {
        JxlExtraChannelInfo extra;
        if (JXL_DEC_SUCCESS != JxlDecoderGetExtraChannelInfo(dec, i, &extra)) {
          fprintf(stderr, "JxlDecoderGetExtraChannelInfo failed\n");
          break;
        }
        printf("+%s", (extra.type < ec_type_names_size
                           ? ec_type_names[extra.type]
                           : "Unknown, please update your libjxl"));
      }
      printf("\n");
      if (verbose) {
        printf("Number of color channels: %d\n", info.num_color_channels);
        printf("Number of extra channels: %d\n", info.num_extra_channels);

        for (uint32_t i = 0; i < info.num_extra_channels; i++) {
          JxlExtraChannelInfo extra;
          if (JXL_DEC_SUCCESS !=
              JxlDecoderGetExtraChannelInfo(dec, i, &extra)) {
            fprintf(stderr, "JxlDecoderGetExtraChannelInfo failed\n");
            break;
          }
          printf("Extra channel %u:\n", i);
          printf("  type: %s\n", (extra.type < ec_type_names_size
                                      ? ec_type_names[extra.type]
                                      : "Unknown, please update your libjxl"));
          printf("  bits per sample: %u\n", extra.bits_per_sample);
          if (extra.exponent_bits_per_sample > 0) {
            printf("  float, with exponent_bits_per_sample: %u\n",
                   extra.exponent_bits_per_sample);
          }
          if (extra.dim_shift > 0) {
            printf("  upsampled %ux\n", 1 << extra.dim_shift);
          }
          if (extra.name_length) {
            std::vector<char> name(extra.name_length + 1);
            if (JXL_DEC_SUCCESS !=
                JxlDecoderGetExtraChannelName(dec, i, name.data(),
                                              extra.name_length + 1)) {
              fprintf(stderr, "JxlDecoderGetExtraChannelName failed\n");
              break;
            }
            printf("  name: %s\n", name.data());
          }
          if (extra.type == JXL_CHANNEL_ALPHA) {
            printf("  alpha premultiplied: %d (%s)\n",
                   extra.alpha_premultiplied,
                   extra.alpha_premultiplied ? "Premultiplied"
                                             : "Non-premultiplied");
          }
          if (extra.type == JXL_CHANNEL_SPOT_COLOR) {
            printf("  spot color: (%f, %f, %f) with opacity %f\n",
                   extra.spot_color[0], extra.spot_color[1],
                   extra.spot_color[2], extra.spot_color[3]);
          }
          if (extra.type == JXL_CHANNEL_CFA)
            printf("  cfa_channel: %u\n", extra.cfa_channel);
        }
      }

      if (info.intensity_target != 255.f || info.min_nits != 0.f ||
          info.relative_to_max_display != 0 ||
          info.relative_to_max_display != 0.f) {
        printf("Intensity target: %f nits\n", info.intensity_target);
        printf("Min nits: %f\n", info.min_nits);
        printf("Relative to max display: %d\n", info.relative_to_max_display);
        printf("Linear below: %f\n", info.linear_below);
      }
      if (verbose) printf("Have preview: %d\n", info.have_preview);
      if (info.have_preview) {
        printf("Preview image: %ux%u\n", info.preview.xsize,
               info.preview.ysize);
      }
      if (verbose) printf("Have animation: %d\n", info.have_animation);
      if (verbose && info.have_animation) {
        printf("Ticks per second (numerator / denominator): %u / %u\n",
               info.animation.tps_numerator, info.animation.tps_denominator);
        printf("Num loops: %u\n", info.animation.num_loops);
        printf("Have timecodes: %d\n", info.animation.have_timecodes);
      }
      if (info.xsize != info.intrinsic_xsize ||
          info.ysize != info.intrinsic_ysize || verbose) {
        printf("Intrinsic dimensions: %ux%u\n", info.intrinsic_xsize,
               info.intrinsic_ysize);
      }
      const char* const orientation_string[8] = {
          "Normal",          "Flipped horizontally",
          "Upside down",     "Flipped vertically",
          "Transposed",      "90 degrees clockwise",
          "Anti-Transposed", "90 degrees counter-clockwise"};
      if (info.orientation > 0 && info.orientation < 9) {
        if (verbose || info.orientation > 1) {
          printf("Orientation: %d (%s)\n", info.orientation,
                 orientation_string[info.orientation - 1]);
        }
      } else {
        fprintf(stderr, "Invalid orientation\n");
      }
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      JxlColorEncoding color_encoding;
      if (JXL_DEC_SUCCESS ==
          JxlDecoderGetColorAsEncodedProfile(
              dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &color_encoding)) {
        PrintColorEncoding("", &color_encoding);
      } else {
        // The profile is not in JPEG XL encoded form, get as ICC profile
        // instead.
        size_t profile_size;
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetICCProfileSize(dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                        &profile_size)) {
          fprintf(stderr, "JxlDecoderGetICCProfileSize failed\n");
          continue;
        }
        printf("%" PRIu64 "-byte ICC profile, ",
               static_cast<uint64_t>(profile_size));
        if (profile_size < 132) {
          fprintf(stderr, "ICC profile too small\n");
          continue;
        }
        std::vector<uint8_t> profile(profile_size);
        if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
                                   dec, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                   profile.data(), profile_size)) {
          fprintf(stderr, "JxlDecoderGetColorAsICCProfile failed\n");
          continue;
        }
        printf("CMM type: \"%.4s\", ", profile.data() + 4);
        printf("Color space: \"%.4s\", ", profile.data() + 16);
        printf("Rendering intent: %d\n", static_cast<int>(profile[67]));
      }
    } else if (status == JXL_DEC_FRAME) {
      if (JXL_DEC_SUCCESS != JxlDecoderGetFrameHeader(dec, &frame_header)) {
        fprintf(stderr, "JxlDecoderGetFrameHeader failed\n");
        break;
      }
      if (frame_header.duration == 0) {
        if (frame_header.is_last && framecount == 0 &&
            frame_header.name_length == 0)
          continue;
        printf("Layer: ");
      } else {
        printf("Frame: ");
      }
      framecount++;
      if (frame_header.layer_info.have_crop) {
        printf("%ux%u at position (%i,%i)", frame_header.layer_info.xsize,
               frame_header.layer_info.ysize, frame_header.layer_info.crop_x0,
               frame_header.layer_info.crop_y0);
      } else {
        printf("Full image size");
      }
      if (info.have_animation) {
        float ms = frame_header.duration * 1000.f *
                   info.animation.tps_denominator /
                   info.animation.tps_numerator;
        total_duration += ms;
        printf(", duration: %.1f ms", ms);
        if (info.animation.have_timecodes) {
          printf(", time code: %X", frame_header.timecode);
        }
      }
      if (frame_header.name_length) {
        std::vector<char> name(frame_header.name_length + 1);
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetFrameName(dec, name.data(),
                                   frame_header.name_length + 1)) {
          fprintf(stderr, "JxlDecoderGetFrameName failed\n");
          break;
        }
        printf(", name: \"%s\"", name.data());
      }
      printf("\n");
    } else if (status == JXL_DEC_BOX) {
      uint64_t size;
      uint64_t contents_size;
      JxlDecoderGetBoxType(dec, box_type, JXL_FALSE);
      JxlDecoderGetBoxSizeRaw(dec, &size);
      JxlDecoderGetBoxSizeContents(dec, &contents_size);
      if (verbose) {
        printf("Box:\n  type: \"%.4s\"\n  size: %" PRIu64
               "\n  contents size: %" PRIu64 "\n",
               box_type, size, contents_size);
      }
      if (!strncmp(box_type, "JXL ", 4)) {
        printf("JPEG XL file format container (ISO/IEC 18181-2)\n");
      } else if (!strncmp(box_type, "ftyp", 4)) {
      } else if (!strncmp(box_type, "jxlc", 4)) {
      } else if (!strncmp(box_type, "jxlp", 4)) {
      } else if (!strncmp(box_type, "jxll", 4)) {
      } else if (!strncmp(box_type, "jxli", 4)) {
        printf("Frame index box present\n");
      } else if (!strncmp(box_type, "jbrd", 4)) {
        printf("JPEG bitstream reconstruction data available\n");
      } else if (!strncmp(box_type, "jumb", 4) ||
                 !strncmp(box_type, "Exif", 4) ||
                 !strncmp(box_type, "xml ", 4)) {
        printf("Uncompressed %.4s metadata: %" PRIu64 " bytes\n", box_type,
               size);

      } else if (!strncmp(box_type, "brob", 4)) {
        JxlDecoderGetBoxType(dec, box_type, JXL_TRUE);
        printf("Brotli-compressed %.4s metadata: %" PRIu64
               " compressed bytes\n",
               box_type, size);
      } else if (!strncmp(box_type, "jhgm", 4)) {
        box_data = reinterpret_cast<uint8_t*>(malloc(chunk_size));
        if (box_data == nullptr) {
          fprintf(stderr, "Memory reallocation failed\n");
          break;
        }
        box_size = chunk_size;
        JxlDecoderSetBoxBuffer(dec, box_data, box_size);
      } else {
        printf("Unknown box:\n  type: \"%.4s\"\n  size: %" PRIu64 "\n",
               box_type, size);
      }
    } else if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
      const size_t remaining = JxlDecoderReleaseBoxBuffer(dec);
      box_size += chunk_size;
      box_index += chunk_size - remaining;
      uint8_t* temp = reinterpret_cast<uint8_t*>(realloc(box_data, box_size));
      if (temp == nullptr) {
        box_size = 0;
        box_index = 0;
        fprintf(stderr, "Memory reallocation failed\n");
        break;
      }
      box_data = temp;
      JxlDecoderSetBoxBuffer(dec, box_data + box_index, box_size - box_index);
    } else if (status == JXL_DEC_BOX_COMPLETE) {
      if (!strncmp(box_type, "jhgm", 4)) {
        size_t remaining = JxlDecoderReleaseBoxBuffer(dec);
        box_size -= remaining;
        JxlGainMapBundle gain_map_bundle;
        size_t bytes_read;
        if (!JxlGainMapReadBundle(&gain_map_bundle, box_data, box_size,
                                  &bytes_read)) {
          fprintf(stderr, "Invalid gain map box found\n");
        } else {
          uint8_t* icc = nullptr;
          size_t icc_size = 0;
          JxlMemoryManager manager = {/*opaque*/ nullptr, /*alloc*/ nullptr,
                                      /*free*/ nullptr};
          if (gain_map_bundle.alt_icc_size > 0 &&
              !JxlICCProfileDecode(&manager, gain_map_bundle.alt_icc,
                                   gain_map_bundle.alt_icc_size, &icc,
                                   &icc_size)) {
            fprintf(stderr,
                    "Invalid gain map box found (ICC profile does not "
                    "decompress)\n");
          }
          printf("Gain map (jhgm) box: version = %u",
                 gain_map_bundle.jhgm_version);
          if (icc_size > 0) {
            printf("  %lu-byte ICC profile\n",
                   static_cast<unsigned long>(icc_size));
          }
          printf("  %u-byte gain map, %u-byte metadata\n",
                 gain_map_bundle.gain_map_size,
                 gain_map_bundle.gain_map_metadata_size);
          if (gain_map_bundle.has_color_encoding) {
            PrintColorEncoding("  ", &gain_map_bundle.color_encoding);
          }
          free(icc);
        }
        free(box_data);
        box_data = nullptr;
        box_size = 0;
        box_index = 0;
      } else {
        fprintf(
            stderr,
            "Unexpected JXL_DEC_BOX_COMPLETE event received for box \"%.4s\"\n",
            box_type);
        continue;
      }
    } else {
      fprintf(stderr, "Unexpected decoder status\n");
      break;
    }
  }
  if (info.animation.num_loops > 1) total_duration *= info.animation.num_loops;
  if (info.have_animation) {
    printf("Animation length: %.3f seconds%s\n", total_duration * 0.001f,
           (info.animation.num_loops ? "" : " (looping)"));
  }
  JxlDecoderDestroy(dec);
  if (box_data != nullptr) free(box_data);
  if (data != nullptr) free(data);

  return seen_basic_info;
}

static void print_usage(const char* name) {
  fprintf(stderr,
          "Usage: %s [-v] [-h] INPUT\n"
          "  INPUT                  input JPEG XL image filename(s)\n"
          "  -v (or --verbose)      more verbose output\n"
          "  -h (or --help or -?)   this help)\n",
          name);
}

static int print_basic_info_filename(const char* jxl_filename, int verbose) {
  FILE* file = fopen(jxl_filename, "rb");
  if (!file) {
    fprintf(stderr, "Failed to read file: %s\n", jxl_filename);
    return 1;
  }
  int status = PrintBasicInfo(file, verbose);
  fclose(file);
  if (!status) {
    fprintf(stderr, "Error reading file: %s\n", jxl_filename);
    return status;
  }

  return 0;
}

template <size_t N>
static bool is_flag(const char* arg, std::array<const char*, N> opts) {
  for (auto opt : opts) {
    if (!strcmp(opt, arg)) {
      return true;
    }
  }
  return false;
}

int main(int argc, char* argv[]) {
  int verbose = 0;
  int status = 0;
  const char* const name = argv[0];
  std::array<const char*, 3> help_opts = {"--help", "-h", "-?"};
  std::array<const char*, 2> verbose_opts = {"--verbose", "-v"};
  if (argc < 2) {
    print_usage(name);
    return 2;
  }

  // First pass: Check for flags
  for (int i = 1; i < argc; i++) {
    if (!verbose && is_flag(argv[i], verbose_opts)) {
      verbose = 1;
    }
    if (is_flag(argv[i], help_opts)) {
      print_usage(name);
      return 0;
    }
  }

  // Second pass: print info
  while (argc-- >= 2) {
    if (is_flag(*(argv + 1), verbose_opts) || is_flag(*(argv + 1), help_opts)) {
      ++argv;
    } else {
      status |= print_basic_info_filename(*++argv, verbose);
    }
  }
  return status;
}
