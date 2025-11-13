// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/cms_interface.h>
#include <jxl/decode.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/apng.h"
#include "lib/extras/metrics.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/jpeg/enc_jpeg_data.h"
#include "tools/benchmark/benchmark_args.h"
#include "tools/benchmark/benchmark_codec.h"
#include "tools/benchmark/benchmark_file_io.h"
#include "tools/benchmark/benchmark_stats.h"
#include "tools/benchmark/benchmark_utils.h"
#include "tools/cmdline.h"
#include "tools/codec_config.h"
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"
#include "tools/speed_stats.h"
#include "tools/ssimulacra2.h"
#include "tools/thread_pool_internal.h"
#include "tools/tracking_memory_manager.h"

namespace jpegxl {
namespace tools {
namespace {

#define QUIT(M) JPEGXL_TOOLS_ABORT(M)

using ::jxl::ButteraugliParams;
using ::jxl::Bytes;
using ::jxl::CodecInOut;
using ::jxl::ColorEncoding;
using ::jxl::Image3F;
using ::jxl::ImageBundle;
using ::jxl::ImageF;
using ::jxl::JxlButteraugliComparator;
using ::jxl::Rng;
using ::jxl::Status;
using ::jxl::StatusOr;
using ::jxl::ThreadPool;
using ::jxl::extras::PackedPixelFile;

Status WriteImage(const Image3F& image, ThreadPool* pool,
                  const std::string& filename) {
  JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_BIG_ENDIAN, 0};
  JXL_ASSIGN_OR_RETURN(PackedPixelFile ppf,
                       jxl::extras::ConvertImage3FToPackedPixelFile(
                           image, ColorEncoding::SRGB(), format, pool));
  std::vector<uint8_t> encoded;
  return jxl::Encode(ppf, filename, &encoded, pool) &&
         WriteFile(filename, encoded);
}

void PrintStats(const TrackingMemoryManager& memory_manager) {
  fprintf(stderr,
          "Allocation count: %" PRIuS ", total: %E (max bytes in use: %E)\n",
          static_cast<size_t>(memory_manager.total_allocations),
          static_cast<double>(memory_manager.total_bytes_allocated),
          static_cast<double>(memory_manager.max_bytes_in_use));
}

Status ReadPNG(const std::string& filename, Image3F* image) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  CodecInOut io{memory_manager};
  std::vector<uint8_t> encoded;
  JXL_RETURN_IF_ERROR(ReadFile(filename, &encoded));
  JXL_RETURN_IF_ERROR(
      jxl::SetFromBytes(Bytes(encoded), jxl::extras::ColorHints(), &io));
  JXL_ASSIGN_OR_RETURN(*image,
                       Image3F::Create(memory_manager, io.xsize(), io.ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(*io.Main().color(), image));
  return true;
}

Status CreateNonSRGBICCProfile(PackedPixelFile* ppf) {
  ColorEncoding color_encoding;
  JXL_RETURN_IF_ERROR(color_encoding.FromExternal(ppf->color_encoding));
  if (color_encoding.ICC().empty()) {
    return JXL_FAILURE("Invalid color encoding.");
  }
  if (!color_encoding.IsSRGB()) {
    ppf->icc.assign(color_encoding.ICC().begin(), color_encoding.ICC().end());
  }
  return true;
}

std::string CodecToExtension(const std::string& codec_name, char sep) {
  std::string result;
  // Add in the parameters of the codec_name in reverse order, so that the
  // name of the file format (e.g. jxl) is last.
  int pos = static_cast<int>(codec_name.size()) - 1;
  while (pos > 0) {
    int prev = codec_name.find_last_of(sep, pos);
    if (prev > pos) prev = -1;
    result += '.' + codec_name.substr(prev + 1, pos - prev);
    pos = prev - 1;
  }
  return result;
}

Status DoCompress(const std::string& filename, const PackedPixelFile& ppf,
                  const std::vector<std::string>& extra_metrics_commands,
                  ImageCodec* codec, ThreadPool* inner_pool,
                  std::vector<uint8_t>* compressed, BenchmarkStats* s) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  ++s->total_input_files;

  if (ppf.frames.size() != 1) {
    // Multiple frames not supported.
    if (!Args()->silent_errors) {
      JXL_WARNING("multiframe input image not supported %s", filename.c_str());
    }
    return false;
  }
  const size_t xsize = ppf.info.xsize;
  const size_t ysize = ppf.info.ysize;
  const size_t input_pixels = xsize * ysize;

  jpegxl::tools::SpeedStats speed_stats;
  jpegxl::tools::SpeedStats::Summary summary;

  bool valid = true;  // false if roundtrip, encoding or decoding errors occur.

  if (!Args()->decode_only && (xsize == 0 || ysize == 0)) {
    // This means the benchmark couldn't load the image, e.g. due to invalid
    // ICC profile. Warning message about that was already printed. Continue
    // this function to indicate it as error in the stats.
    valid = false;
  }
  const PackedPixelFile* ppf1 = &ppf;
  PackedPixelFile ppf2;

  for (size_t generation = 0; generation <= Args()->generations; generation++) {
    std::string ext = FileExtension(filename);
    if (valid && !Args()->decode_only) {
      for (size_t i = 0; i < Args()->encode_reps; ++i) {
        if (codec->CanRecompressJpeg() && (ext == ".jpg" || ext == ".jpeg")) {
          std::vector<uint8_t> data_in;
          JXL_RETURN_IF_ERROR(ReadFile(filename, &data_in));
          JXL_RETURN_IF_ERROR(codec->RecompressJpeg(filename, data_in,
                                                    compressed, &speed_stats));
        } else {
          Status status = codec->Compress(filename, *ppf1, inner_pool,
                                          compressed, &speed_stats);
          if (!status) {
            valid = false;
            if (!Args()->silent_errors) {
              std::string message = codec->GetErrorMessage();
              if (!message.empty()) {
                fprintf(stderr, "Error in %s codec: %s\n",
                        codec->description().c_str(), message.c_str());
              } else {
                fprintf(stderr, "Error in %s codec\n",
                        codec->description().c_str());
              }
            }
          }
        }
      }
      JXL_RETURN_IF_ERROR(speed_stats.GetSummary(&summary));
      s->total_time_encode += summary.central_tendency;
    }

    if (valid && Args()->decode_only) {
      std::vector<uint8_t> data_in;
      JXL_RETURN_IF_ERROR(ReadFile(filename, &data_in));
      compressed->insert(compressed->end(), data_in.begin(), data_in.end());
    }

    // Decompress
    if (valid) {
      speed_stats = jpegxl::tools::SpeedStats();
      for (size_t i = 0; i < Args()->decode_reps; ++i) {
        if (!codec->Decompress(filename, Bytes(*compressed), inner_pool, &ppf2,
                               &speed_stats)) {
          if (!Args()->silent_errors) {
            fprintf(stderr,
                    "%s failed to decompress encoded image. Original source:"
                    " %s\n",
                    codec->description().c_str(), filename.c_str());
          }
          valid = false;
        }
      }
      JXL_RETURN_IF_ERROR(speed_stats.GetSummary(&summary));
      s->total_time_decode += summary.central_tendency;
    }
    ppf1 = &ppf2;
  }

  std::string name = FileBaseName(filename);
  std::string codec_name = codec->description();

  if (!valid) {
    s->total_errors++;
  }

  if (valid) {
    for (const auto& frame : ppf2.frames) {
      s->total_input_pixels += frame.color.xsize * frame.color.ysize;
    }
  }

  if (ppf.frames.size() != ppf2.frames.size()) {
    if (!Args()->silent_errors) {
      // Animated gifs not supported yet?
      fprintf(stderr,
              "Frame sizes not equal, is this an animated gif? %s %s %" PRIuS
              " %" PRIuS "\n",
              codec_name.c_str(), name.c_str(), ppf.frames.size(),
              ppf2.frames.size());
    }
    valid = false;
  }

  bool skip_butteraugli = Args()->skip_butteraugli || Args()->decode_only;
  ImageF distmap;
  float distance = 1.0f;

  if (valid && !skip_butteraugli) {
    CodecInOut ppf_io{memory_manager};
    JXL_RETURN_IF_ERROR(
        ConvertPackedPixelFileToCodecInOut(ppf, inner_pool, &ppf_io));
    CodecInOut ppf2_io{memory_manager};
    JXL_RETURN_IF_ERROR(
        ConvertPackedPixelFileToCodecInOut(ppf2, inner_pool, &ppf2_io));
    const ImageBundle& ib1 = ppf_io.Main();
    const ImageBundle& ib2 = ppf2_io.Main();
    if (jxl::SameSize(ppf, ppf2)) {
      ButteraugliParams params;
      // Hack the default intensity target value for SDR images to be 80.0, the
      // intensity target of sRGB images and a more reasonable viewing default
      // than JPEG XL file format's default.
      // TODO(szabadka) Support different intensity targets as well.
      const auto& transfer_function = ib1.c_current().Tf();
      params.intensity_target = transfer_function.IsPQ()    ? 10000.f
                                : transfer_function.IsHLG() ? 1000.f
                                                            : 80.f;

      const JxlCmsInterface& cms = *JxlGetDefaultCms();
      JxlButteraugliComparator comparator(params, cms);
      JXL_RETURN_IF_ERROR(ComputeScore(ib1, ib2, &comparator, cms, &distance,
                                       &distmap, inner_pool,
                                       codec->IgnoreAlpha()));
    } else {
      // TODO(veluca): re-upsample and compute proper distance.
      distance = 1e+4f;
      JXL_ASSIGN_OR_RETURN(distmap, ImageF::Create(memory_manager, 1, 1));
      distmap.Row(0)[0] = distance;
    }
    // Update stats
    s->psnr +=
        compressed->empty()
            ? 0
            : jxl::ComputePSNR(ib1, ib2, *JxlGetDefaultCms()) * input_pixels;
    double pnorm =
        ComputeDistanceP(distmap, ButteraugliParams(), Args()->error_pnorm);
    s->distance_p_norm += pnorm * input_pixels;
    JXL_ASSIGN_OR_RETURN(Msssim msssim, ComputeSSIMULACRA2(ib1, ib2));
    double ssimulacra2 = msssim.Score();
    s->ssimulacra2 += ssimulacra2 * input_pixels;
    s->max_distance = std::max(s->max_distance, distance);
    s->distances.push_back(distance);
    s->pnorms.push_back(pnorm);
    s->ssimulacra2s.push_back(ssimulacra2);
  }

  s->total_compressed_size += compressed->size();
  s->total_adj_compressed_size += compressed->size() * std::max(1.0f, distance);
  codec->GetMoreStats(s);

  if (Args()->save_compressed || Args()->save_decompressed) {
    std::string dir = FileDirName(filename);
    std::string outdir =
        Args()->output_dir.empty() ? dir + "/out" : Args()->output_dir;
    std::string compressed_fn =
        outdir + "/" + name + CodecToExtension(codec_name, ':');
    std::string decompressed_fn = compressed_fn + Args()->output_extension;
    std::string heatmap_fn;
    if (jxl::extras::GetAPNGEncoder()) {
      heatmap_fn = compressed_fn + ".heatmap.png";
    } else {
      heatmap_fn = compressed_fn + ".heatmap.ppm";
    }
    JXL_RETURN_IF_ERROR(MakeDir(outdir));
    if (Args()->save_compressed) {
      JXL_RETURN_IF_ERROR(WriteFile(compressed_fn, *compressed));
    }
    if (Args()->save_decompressed && valid) {
      // TODO(szabadka): Handle Args()->mul_output
      std::vector<uint8_t> encoded;
      JXL_RETURN_IF_ERROR(jxl::Encode(ppf2, decompressed_fn, &encoded));
      JXL_RETURN_IF_ERROR(WriteFile(decompressed_fn, encoded));
      if (!skip_butteraugli) {
        float good = Args()->heatmap_good > 0.0f
                         ? Args()->heatmap_good
                         : jxl::ButteraugliFuzzyInverse(1.5);
        float bad = Args()->heatmap_bad > 0.0f
                        ? Args()->heatmap_bad
                        : jxl::ButteraugliFuzzyInverse(0.5);
        if (Args()->save_heatmap) {
          JXL_ASSIGN_OR_RETURN(Image3F heatmap,
                               CreateHeatMapImage(distmap, good, bad));
          JXL_RETURN_IF_ERROR(WriteImage(heatmap, inner_pool, heatmap_fn));
        }
      }
    }
  }
  if (!extra_metrics_commands.empty()) {
    TemporaryFile tmp_in("original", "pfm");
    TemporaryFile tmp_out("decoded", "pfm");
    TemporaryFile tmp_res("result", "txt");
    std::string tmp_in_fn;
    std::string tmp_out_fn;
    std::string tmp_res_fn;
    JXL_RETURN_IF_ERROR(tmp_in.GetFileName(&tmp_in_fn));
    JXL_RETURN_IF_ERROR(tmp_out.GetFileName(&tmp_out_fn));
    JXL_RETURN_IF_ERROR(tmp_res.GetFileName(&tmp_res_fn));

    std::vector<uint8_t> encoded;
    JXL_RETURN_IF_ERROR(jxl::Encode(ppf, tmp_in_fn, &encoded));
    JXL_RETURN_IF_ERROR(WriteFile(tmp_in_fn, encoded));
    JXL_RETURN_IF_ERROR(jxl::Encode(ppf2, tmp_out_fn, &encoded));
    JXL_RETURN_IF_ERROR(WriteFile(tmp_out_fn, encoded));
    // TODO(szabadka) Handle custom intensity target.
    std::string intensity_target = "255";
    for (const auto& extra_metrics_command : extra_metrics_commands) {
      float res = nanf("");
      bool error = false;
      if (RunCommand(extra_metrics_command,
                     {tmp_in_fn, tmp_out_fn, tmp_res_fn, intensity_target})) {
        FILE* f = fopen(tmp_res_fn.c_str(), "r");
        if (fscanf(f, "%f", &res) != 1) {
          error = true;
        }
        fclose(f);
      } else {
        error = true;
      }
      if (error) {
        fprintf(stderr,
                "WARNING: Computation of metric with command %s failed\n",
                extra_metrics_command.c_str());
      }
      s->extra_metrics.push_back(res);
    }
  }

  if (Args()->show_progress) {
    fprintf(stderr, ".");
    fflush(stderr);
  }
  return true;
}

// Makes a base64 data URI for embedded image in HTML
std::string Base64Image(const std::string& filename) {
  std::vector<uint8_t> bytes;
  if (!ReadFile(filename, &bytes)) {
    return "";
  }
  static const char* symbols =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  for (size_t i = 0; i < bytes.size(); i += 3) {
    uint8_t o0 = bytes[i + 0];
    uint8_t o1 = (i + 1 < bytes.size()) ? bytes[i + 1] : 0;
    uint8_t o2 = (i + 2 < bytes.size()) ? bytes[i + 2] : 0;
    uint32_t value = (o0 << 16) | (o1 << 8) | o2;
    for (size_t j = 0; j < 4; j++) {
      result += (i + j <= bytes.size()) ? symbols[(value >> (6 * (3 - j))) & 63]
                                        : '=';
    }
  }
  // NOTE: Chrome supports max 2MB of data this way for URLs, but appears to
  // support larger images anyway as long as it's embedded in the HTML file
  // itself. If more data is needed, use createObjectURL.
  return "data:image;base64," + result;
}

struct Task {
  ImageCodecPtr codec;
  size_t idx_image;
  size_t idx_method;
  const PackedPixelFile* image;
  BenchmarkStats stats;
};

Status WriteHtmlReport(const std::string& codec_desc,
                       const std::vector<std::string>& fnames,
                       const std::vector<const Task*>& tasks,
                       const std::vector<const PackedPixelFile*>& images,
                       bool add_heatmap, bool self_contained) {
  std::string toggle_js =
      "<script type=\"text/javascript\">\n"
      "  var codecname = '" +
      codec_desc + "';\n";
  if (add_heatmap) {
    toggle_js += R"(
  var maintitle = codecname + ' - click images to toggle, press space to' +
      ' toggle all, h to toggle all heatmaps. Zoom in with CTRL+wheel or' +
      ' CTRL+plus.';
  document.title = maintitle;
  var counter = [];
  function setState(i, s) {
    var preview = document.getElementById("preview" + i);
    var orig = document.getElementById("orig" + i);
    var hm = document.getElementById("hm" + i);
    if (s == 0) {
      preview.style.display = 'none';
      orig.style.display = 'block';
      hm.style.display = 'none';
    } else if (s == 1) {
      preview.style.display = 'block';
      orig.style.display = 'none';
      hm.style.display = 'none';
    } else if (s == 2) {
      preview.style.display = 'none';
      orig.style.display = 'none';
      hm.style.display = 'block';
    }
  }
  function toggle(i) {
    for (index = counter.length; index <= i; index++) {
      counter.push(1);
    }
    setState(i, counter[i]);
    counter[i] = (counter[i] + 1) % 3;
    document.title = maintitle;
  }
  var toggleall_state = 1;
  document.body.onkeydown = function(e) {
    // space (32) to toggle orig/compr, 'h' (72) to toggle heatmap/compr
    if (e.keyCode == 32 || e.keyCode == 72) {
      var divs = document.getElementsByTagName('div');
      var key_state = (e.keyCode == 32) ? 0 : 2;
      toggleall_state = (toggleall_state == key_state) ? 1 : key_state;
      document.title = codecname + ' - ' + (toggleall_state == 0 ?
          'originals' : (toggleall_state == 1 ? 'compressed' : 'heatmaps'));
      for (var i = 0; i < divs.length; i++) {
        setState(i, toggleall_state);
      }
      return false;
    }
  };
</script>
)";
  } else {
    toggle_js += R"(
  var maintitle = codecname + ' - click images to toggle, press space to' +
      ' toggle all. Zoom in with CTRL+wheel or CTRL+plus.';
  document.title = maintitle;
  var counter = [];
  function setState(i, s) {
    var preview = document.getElementById("preview" + i);
    var orig = document.getElementById("orig" + i);
    if (s == 0) {
      preview.style.display = 'none';
      orig.style.display = 'block';
    } else if (s == 1) {
      preview.style.display = 'block';
      orig.style.display = 'none';
    }
  }
  function toggle(i) {
    for (index = counter.length; index <= i; index++) {
      counter.push(1);
    }
    setState(i, counter[i]);
    counter[i] = 1 - counter[i];
    document.title = maintitle;
  }
  var toggleall_state = 1;
  document.body.onkeydown = function(e) {
    // space (32) to toggle orig/compr
    if (e.keyCode == 32) {
      var divs = document.getElementsByTagName('div');
      toggleall_state = 1 - toggleall_state;
      document.title = codecname + ' - ' + (toggleall_state == 0 ?
          'originals' : 'compressed');
      for (var i = 0; i < divs.length; i++) {
        setState(i, toggleall_state);
      }
      return false;
    }
  };
</script>
)";
  }
  std::string out_html;
  std::string outdir;
  out_html.append("<body bgcolor=\"#000\">\n");
  out_html.append("<style>img { image-rendering: pixelated; }</style>\n");
  std::string codec_name = codec_desc;
  // Make compatible for filename
  std::replace(codec_name.begin(), codec_name.end(), ':', '_');
  for (size_t i = 0; i < fnames.size(); ++i) {
    std::string name = FileBaseName(fnames[i]);
    std::string dir = FileDirName(fnames[i]);
    outdir = Args()->output_dir.empty() ? dir + "/out" : Args()->output_dir;
    std::string name_out = name + CodecToExtension(codec_name, '_');
    if (Args()->html_report_use_decompressed) {
      name_out += Args()->output_extension;
    }
    std::string heatmap_out =
        name + CodecToExtension(codec_name, '_') + ".heatmap.png";

    const std::string& fname_orig = fnames[i];
    std::string fname_out = std::string(outdir).append("/").append(name_out);
    std::string fname_heatmap =
        std::string(outdir).append("/").append(heatmap_out);
    std::string url_orig = Args()->originals_url.empty()
                               ? ("file://" + fnames[i])
                               : (Args()->originals_url + "/" + name);
    std::string url_out = name_out;
    std::string url_heatmap = heatmap_out;
    if (self_contained) {
      url_orig = Base64Image(fname_orig);
      url_out = Base64Image(fname_out);
      url_heatmap = Base64Image(fname_heatmap);
    }
    std::string number = StringPrintf("%" PRIuS, i);
    const PackedPixelFile& image = *images[i];
    size_t xsize = image.frames.size() == 1 ? image.info.xsize : 0;
    size_t ysize = image.frames.size() == 1 ? image.info.ysize : 0;
    std::string html_width = StringPrintf("%" PRIuS "px", xsize);
    std::string html_height = StringPrintf("%" PRIuS "px", ysize);
    double bpp = tasks[i]->stats.total_compressed_size * 8.0 /
                 tasks[i]->stats.total_input_pixels;
    double pnorm =
        tasks[i]->stats.distance_p_norm / tasks[i]->stats.total_input_pixels;
    double max_dist = tasks[i]->stats.max_distance;
    std::string compressed_title = StringPrintf(
        "compressed. bpp: %f, pnorm: %f, max dist: %f", bpp, pnorm, max_dist);
    out_html.append("<div onclick=\"toggle(")
        .append(number)
        .append(");\" style=\"display:inline-block;width:")
        .append(html_width)
        .append(";height:")
        .append(html_height)
        .append(";\">\n  <img title=\"")
        .append(compressed_title)
        .append("\" id=\"preview")
        .append(number)
        .append("\" src=")
        .append("\"")
        .append(url_out)
        .append("\"style=\"display:block;\"/>\n")
        .append(R"(  <img title="original" id="orig)")
        .append(number)
        .append("\" src=")
        .append("\"")
        .append(url_orig)
        .append("\"style=\"display:none;\"/>\n");
    if (add_heatmap) {
      out_html.append(R"(  <img title="heatmap" id="hm)")
          .append(number)
          .append("\" src=")
          .append("\"")
          .append(url_heatmap)
          .append("\"style=\"display:none;\"/>\n");
    }
    out_html.append("</div>\n");
  }
  out_html.append("</body>\n").append(toggle_js);
  std::string fname_index =
      std::string(outdir).append("/index.").append(codec_name).append(".html");
  JXL_RETURN_IF_ERROR(WriteFile(fname_index, out_html));
  return true;
}

// Prints the detailed and aggregate statistics, in the correct order but as
// soon as possible when multithreaded tasks are done.
struct StatPrinter {
  StatPrinter(const std::vector<std::string>& methods,
              const std::vector<std::string>& extra_metrics_names,
              const std::vector<std::string>& fnames,
              const std::vector<Task>& tasks)
      : methods_(&methods),
        extra_metrics_names_(&extra_metrics_names),
        fnames_(&fnames),
        tasks_(&tasks),
        tasks_done_(0),
        stats_printed_(0),
        details_printed_(0) {
    stats_done_.resize(methods.size(), 0);
    details_done_.resize(tasks.size(), 0);
    max_fname_width_ = 0;
    for (const auto& fname : fnames) {
      max_fname_width_ = std::max(max_fname_width_, FileBaseName(fname).size());
    }
    max_method_width_ = 0;
    for (const auto& method : methods) {
      max_method_width_ =
          std::max(max_method_width_, FileBaseName(method).size());
    }
  }

  Status TaskDone(size_t task_index, const Task& t) {
    std::lock_guard<std::mutex> guard(mutex);
    tasks_done_++;
    if (Args()->print_details || Args()->show_progress) {
      if (Args()->print_details) {
        // Render individual results as soon as they are ready and all previous
        // ones in task order are ready.
        details_done_[task_index] = 1;
        if (task_index == details_printed_) {
          while (details_printed_ < tasks_->size() &&
                 details_done_[details_printed_]) {
            PrintDetails((*tasks_)[details_printed_]);
            details_printed_++;
          }
        }
      }
      // When using "show_progress" or "print_details", the table must be
      // rendered at the very end, else the details or progress would be
      // rendered in-between the table rows.
      if (tasks_done_ == tasks_->size()) {
        JXL_RETURN_IF_ERROR(PrintStatsHeader());
        for (size_t i = 0; i < methods_->size(); i++) {
          JXL_RETURN_IF_ERROR(PrintStats((*methods_)[i], i));
        }
        JXL_RETURN_IF_ERROR(PrintStatsFooter());
      }
    } else {
      if (tasks_done_ == 1) {
        JXL_RETURN_IF_ERROR(PrintStatsHeader());
      }
      // Render lines of the table as soon as it is ready and all previous
      // lines have been printed.
      stats_done_[t.idx_method]++;
      if (stats_done_[t.idx_method] == fnames_->size() &&
          t.idx_method == stats_printed_) {
        while (stats_printed_ < stats_done_.size() &&
               stats_done_[stats_printed_] == fnames_->size()) {
          JXL_RETURN_IF_ERROR(
              PrintStats((*methods_)[stats_printed_], stats_printed_));
          stats_printed_++;
        }
      }
      if (tasks_done_ == tasks_->size()) {
        JXL_RETURN_IF_ERROR(PrintStatsFooter());
      }
    }
    return true;
  }

  void PrintDetails(const Task& t) const {
    double comp_bpp =
        t.stats.total_compressed_size * 8.0 / t.stats.total_input_pixels;
    double p_norm = t.stats.distance_p_norm / t.stats.total_input_pixels;
    double psnr = t.stats.psnr / t.stats.total_input_pixels;
    double ssimulacra2 = t.stats.ssimulacra2 / t.stats.total_input_pixels;
    double bpp_p_norm = p_norm * comp_bpp;

    const double adj_comp_bpp =
        t.stats.total_adj_compressed_size * 8.0 / t.stats.total_input_pixels;

    size_t pixels = t.stats.total_input_pixels;

    const double enc_mps =
        t.stats.total_input_pixels / (1000000.0 * t.stats.total_time_encode);
    const double dec_mps =
        t.stats.total_input_pixels / (1000000.0 * t.stats.total_time_decode);
    if (Args()->print_details_csv) {
      printf("%s,%s,%" PRIdS ",%" PRIdS ",%" PRIdS
             ",%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f",
             (*methods_)[t.idx_method].c_str(),
             FileBaseName((*fnames_)[t.idx_image]).c_str(),
             t.stats.total_errors, t.stats.total_compressed_size, pixels,
             enc_mps, dec_mps, comp_bpp, t.stats.max_distance, ssimulacra2,
             psnr, p_norm, bpp_p_norm, adj_comp_bpp);
      for (float m : t.stats.extra_metrics) {
        printf(",%.8f", m);
      }
      printf("\n");
    } else {
      printf("%s", (*methods_)[t.idx_method].c_str());
      for (size_t i = (*methods_)[t.idx_method].size(); i <= max_method_width_;
           i++) {
        printf(" ");
      }
      printf("%s", FileBaseName((*fnames_)[t.idx_image]).c_str());
      for (size_t i = FileBaseName((*fnames_)[t.idx_image]).size();
           i <= max_fname_width_; i++) {
        printf(" ");
      }
      printf(
          "error:%" PRIdS "    size:%8" PRIdS "    pixels:%9" PRIdS
          "    enc_speed:%8.8f    dec_speed:%8.8f    bpp:%10.8f    dist:%10.8f"
          "    psnr:%10.8f    ssimulacra2:%.2f   p:%10.8f    bppp:%10.8f    "
          "qabpp:%10.8f ",
          t.stats.total_errors, t.stats.total_compressed_size, pixels, enc_mps,
          dec_mps, comp_bpp, t.stats.max_distance, psnr, ssimulacra2, p_norm,
          bpp_p_norm, adj_comp_bpp);
      for (size_t i = 0; i < t.stats.extra_metrics.size(); i++) {
        printf(" %s:%.8f", (*extra_metrics_names_)[i].c_str(),
               t.stats.extra_metrics[i]);
      }
      printf("\n");
    }
    fflush(stdout);
  }

  Status PrintStats(const std::string& method, size_t idx_method) {
    // Assimilate all tasks with the same idx_method.
    BenchmarkStats method_stats;
    std::vector<const PackedPixelFile*> images;
    std::vector<const Task*> tasks;
    for (const Task& t : *tasks_) {
      if (t.idx_method == idx_method) {
        method_stats.Assimilate(t.stats);
        images.push_back(t.image);
        tasks.push_back(&t);
      }
    }
    JXL_ENSURE(method_stats.total_input_files == fnames_->size());

    std::string out;

    JXL_RETURN_IF_ERROR(method_stats.PrintMoreStats());  // not concurrent
    out += method_stats.PrintLine(method);

    if (Args()->write_html_report) {
      JXL_RETURN_IF_ERROR(WriteHtmlReport(
          method, *fnames_, tasks, images,
          Args()->save_heatmap && Args()->html_report_add_heatmap,
          Args()->html_report_self_contained));
    }

    stats_aggregate_.push_back(method_stats.ComputeColumns(method));

    printf("%s", out.c_str());
    fflush(stdout);
    return true;
  }

  Status PrintStatsHeader() const {
    if (Args()->markdown) {
      if (Args()->show_progress) {
        fprintf(stderr, "\n");
        fflush(stderr);
      }
      printf("```\n");
    }
    if (fnames_->size() == 1) {
      printf("%s\n", (*fnames_)[0].c_str());
    } else {
      printf("%" PRIuS " images\n", fnames_->size());
    }
    JXL_ASSIGN_OR_RETURN(std::string header,
                         PrintHeader(*extra_metrics_names_));
    printf("%s", header.c_str());
    fflush(stdout);
    return true;
  }

  Status PrintStatsFooter() const {
    JXL_ASSIGN_OR_RETURN(
        std::string aggregate,
        PrintAggregate(extra_metrics_names_->size(), stats_aggregate_));
    printf("%s", aggregate.c_str());
    if (Args()->markdown) printf("```\n");
    printf("\n");
    fflush(stdout);
    return true;
  }

  const std::vector<std::string>* methods_;
  const std::vector<std::string>* extra_metrics_names_;
  const std::vector<std::string>* fnames_;
  const std::vector<Task>* tasks_;

  size_t tasks_done_;

  size_t stats_printed_;
  std::vector<size_t> stats_done_;

  size_t details_printed_;
  std::vector<size_t> details_done_;

  size_t max_fname_width_;
  size_t max_method_width_;

  std::vector<std::vector<ColumnValue>> stats_aggregate_;

  std::mutex mutex;
};

class Benchmark {
  using StringVec = std::vector<std::string>;

 public:
  // Return the exit code of the program.
  static Status Run() {
    TrackingMemoryManager memory_manager{};
    bool ok = true;
    {
      const StringVec methods = GetMethods();
      const StringVec extra_metrics_names = GetExtraMetricsNames();
      const StringVec extra_metrics_commands = GetExtraMetricsCommands();
      const StringVec fnames = GetFilenames();
      // (non-const because Task.stats are updated)
      JXL_ASSIGN_OR_RETURN(std::vector<Task> tasks,
                           CreateTasks(methods, fnames, memory_manager.get()));

      std::unique_ptr<ThreadPoolInternal> pool;
      std::vector<std::unique_ptr<ThreadPoolInternal>> inner_pools;
      InitThreads(tasks.size(), &pool, &inner_pools);
      if (Args()->generations > 0) {
        fprintf(stderr,
                "Generation loss testing with %" PRIuS
                " intermediate generations\n",
                Args()->generations);
      }
      std::vector<PackedPixelFile> loaded_images =
          LoadImages(fnames, pool->get());

      if (RunTasks(methods, extra_metrics_names, extra_metrics_commands, fnames,
                   loaded_images, pool->get(), inner_pools, &tasks) != 0) {
        ok = false;
        if (!Args()->silent_errors) {
          fprintf(stderr, "There were error(s) in the benchmark.\n");
        }
      }
    }

    PrintStats(memory_manager);
    if (!ok) return JXL_FAILURE("RunTasks error");
    return true;
  }

 private:
  static size_t NumOuterThreads(const size_t num_hw_threads,
                                const size_t num_tasks) {
    // Default to #cores
    size_t num_threads = num_hw_threads;
    if (Args()->num_threads >= 0) {
      num_threads = static_cast<size_t>(Args()->num_threads);
    }

    // As a safety precaution, limit the number of threads to 4x the number of
    // available CPUs.
    num_threads =
        std::min<size_t>(num_threads, 4 * std::thread::hardware_concurrency());

    // Don't create more threads than there are tasks (pointless/wasteful).
    num_threads = std::min(num_threads, num_tasks);

    // Just one thread is counterproductive.
    if (num_threads == 1) num_threads = 0;

    return num_threads;
  }

  static int NumInnerThreads(const size_t num_hw_threads,
                             const size_t num_threads) {
    size_t num_inner;

    // Default: distribute remaining cores among tasks.
    if (Args()->inner_threads < 0) {
      if (num_threads == 0) {
        num_inner = num_hw_threads;
      } else if (num_hw_threads <= num_threads) {
        num_inner = 1;
      } else {
        num_inner = (num_hw_threads - num_threads) / num_threads;
      }
    } else {
      num_inner = static_cast<size_t>(Args()->inner_threads);
    }

    // Just one thread is counterproductive.
    if (num_inner == 1) num_inner = 0;

    return num_inner;
  }

  static void InitThreads(
      size_t num_tasks, std::unique_ptr<ThreadPoolInternal>* pool,
      std::vector<std::unique_ptr<ThreadPoolInternal>>* inner_pools) {
    const size_t num_hw_threads = std::thread::hardware_concurrency();
    const size_t num_threads = NumOuterThreads(num_hw_threads, num_tasks);
    const size_t num_inner = NumInnerThreads(num_hw_threads, num_threads);

    fprintf(stderr,
            "%" PRIuS " total threads, %" PRIuS " tasks, %" PRIuS
            " threads, %" PRIuS " inner threads\n",
            num_hw_threads, num_tasks, num_threads, num_inner);

    *pool = jxl::make_unique<ThreadPoolInternal>(num_threads);
    // Main thread OR worker threads in pool each get a possibly empty nested
    // pool (helps use all available cores when #tasks < #threads)
    for (size_t i = 0; i < std::max<size_t>(num_threads, 1); ++i) {
      inner_pools->emplace_back(new ThreadPoolInternal(num_inner));
    }
  }

  static StringVec GetMethods() {
    StringVec methods = SplitString(Args()->codec, ',');
    for (auto it = methods.begin(); it != methods.end();) {
      if (it->empty()) {
        it = methods.erase(it);
      } else {
        ++it;
      }
    }
    return methods;
  }

  static StringVec GetExtraMetricsNames() {
    StringVec metrics = SplitString(Args()->extra_metrics, ',');
    for (auto it = metrics.begin(); it != metrics.end();) {
      if (it->empty()) {
        it = metrics.erase(it);
      } else {
        *it = SplitString(*it, ':')[0];
        ++it;
      }
    }
    return metrics;
  }

  static StringVec GetExtraMetricsCommands() {
    StringVec metrics = SplitString(Args()->extra_metrics, ',');
    for (auto it = metrics.begin(); it != metrics.end();) {
      if (it->empty()) {
        it = metrics.erase(it);
      } else {
        auto s = SplitString(*it, ':');
        JPEGXL_TOOLS_CHECK(s.size() == 2);
        *it = s[1];
        ++it;
      }
    }
    return metrics;
  }

  static StringVec SampleFromInput(const StringVec& fnames,
                                   const std::string& sample_tmp_dir,
                                   int num_samples, size_t size) {
    JPEGXL_TOOLS_CHECK(!sample_tmp_dir.empty());
    fprintf(stderr, "Creating samples of %" PRIuS "x%" PRIuS " tiles...\n",
            size, size);
    StringVec fnames_out;
    std::vector<Image3F> images;
    std::vector<size_t> offsets;
    size_t total_num_tiles = 0;
    for (const auto& fname : fnames) {
      Image3F img;
      JPEGXL_TOOLS_CHECK(ReadPNG(fname, &img));
      JPEGXL_TOOLS_CHECK(img.xsize() >= size);
      JPEGXL_TOOLS_CHECK(img.ysize() >= size);
      total_num_tiles += (img.xsize() - size + 1) * (img.ysize() - size + 1);
      offsets.push_back(total_num_tiles);
      images.emplace_back(std::move(img));
    }
    JPEGXL_TOOLS_CHECK(MakeDir(sample_tmp_dir));
    Rng rng(0);
    for (int i = 0; i < num_samples; ++i) {
      int val = rng.UniformI(0, offsets.back());
      size_t idx = (std::lower_bound(offsets.begin(), offsets.end(), val) -
                    offsets.begin());
      JPEGXL_TOOLS_CHECK(idx < images.size());
      const Image3F& img = images[idx];
      int x0 = rng.UniformI(0, img.xsize() - size);
      int y0 = rng.UniformI(0, img.ysize() - size);
      JXL_ASSIGN_OR_QUIT(
          Image3F sample,
          Image3F::Create(jpegxl::tools::NoMemoryManager(), size, size),
          "Allocation failure.");
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = 0; y < size; ++y) {
          const float* JXL_RESTRICT row_in = img.PlaneRow(c, y0 + y);
          float* JXL_RESTRICT row_out = sample.PlaneRow(c, y);
          memcpy(row_out, &row_in[x0], size * sizeof(row_out[0]));
        }
      }
      std::string fn_output = StringPrintf(
          "%s/%s.crop_%" PRIuS "x%" PRIuS "+%d+%d.png", sample_tmp_dir.c_str(),
          FileBaseName(fnames[idx]).c_str(), size, size, x0, y0);
      ThreadPool* null_pool = nullptr;
      JPEGXL_TOOLS_CHECK(WriteImage(sample, null_pool, fn_output));
      fnames_out.push_back(fn_output);
    }
    fprintf(stderr, "Created %d sample tiles\n", num_samples);
    return fnames_out;
  }

  static StringVec GetFilenames() {
    StringVec fnames;
    JPEGXL_TOOLS_CHECK(MatchFiles(Args()->input, &fnames));
    if (fnames.empty()) {
      JPEGXL_TOOLS_ABORT("No input file matches pattern");
    }
    if (Args()->print_details) {
      std::sort(fnames.begin(), fnames.end());
    }

    if (Args()->num_samples > 0) {
      fnames = SampleFromInput(fnames, Args()->sample_tmp_dir,
                               Args()->num_samples, Args()->sample_dimensions);
    }
    return fnames;
  }

  // (Load only once, not for every codec)
  static std::vector<PackedPixelFile> LoadImages(const StringVec& fnames,
                                                 ThreadPool* pool) {
    std::vector<PackedPixelFile> loaded_images;
    loaded_images.resize(fnames.size());
    const auto process_image = [&](const uint32_t task,
                                   size_t /*thread*/) -> Status {
      const size_t i = static_cast<size_t>(task);
      Status ret = true;

      if (!Args()->decode_only) {
        std::vector<uint8_t> encoded;
        ret = ReadFile(fnames[i], &encoded);
        if (ret) {
          ret = jxl::extras::DecodeBytes(Bytes(encoded), Args()->color_hints,
                                         &loaded_images[i]);
        }
        if (ret && loaded_images[i].icc.empty()) {
          // Add ICC profile if the image is not in sRGB, because not all codecs
          // can handle the color_encoding enum.
          ret = CreateNonSRGBICCProfile(&loaded_images[i]);
        }
        if (ret && Args()->intensity_target != 0) {
          // TODO(szabadka) Respect Args()->intensity_target
        }
      }
      if (!ret) {
        if (!Args()->silent_errors) {
          fprintf(stderr, "Failed to load image %s\n", fnames[i].c_str());
        }
        return JXL_FAILURE("Failed to load image");
      }

      if (!Args()->decode_only && Args()->override_bitdepth != 0) {
        // TODO(szabadla) Respect Args()->override_bitdepth
      }
      return true;
    };
    JPEGXL_TOOLS_CHECK(
        jxl::RunOnPool(pool, 0, static_cast<uint32_t>(fnames.size()),
                       ThreadPool::NoInit, process_image, "Load images"));
    return loaded_images;
  }

  static StatusOr<std::vector<Task>> CreateTasks(
      const StringVec& methods, const StringVec& fnames,
      JxlMemoryManager* memory_manager) {
    std::vector<Task> tasks;
    tasks.reserve(methods.size() * fnames.size());
    for (size_t idx_image = 0; idx_image < fnames.size(); ++idx_image) {
      for (size_t idx_method = 0; idx_method < methods.size(); ++idx_method) {
        tasks.emplace_back();
        Task& t = tasks.back();
        t.codec = CreateImageCodec(methods[idx_method], memory_manager);
        t.idx_image = idx_image;
        t.idx_method = idx_method;
        // t.stats is default-initialized.
      }
    }
    JXL_ENSURE(tasks.size() == tasks.capacity());
    return tasks;
  }

  // Return the total number of errors.
  static size_t RunTasks(
      const StringVec& methods, const StringVec& extra_metrics_names,
      const StringVec& extra_metrics_commands, const StringVec& fnames,
      const std::vector<PackedPixelFile>& loaded_images, ThreadPool* pool,
      const std::vector<std::unique_ptr<ThreadPoolInternal>>& inner_pools,
      std::vector<Task>* tasks) {
    StatPrinter printer(methods, extra_metrics_names, fnames, *tasks);
    if (Args()->print_details_csv) {
      // Print CSV header
      printf(
          "method,image,error,size,pixels,enc_speed,dec_speed,"
          "bpp,maxnorm,ssimulacra2,psnr,pnorm,bppp,qabpp");
      for (const std::string& s : extra_metrics_names) {
        printf(",%s", s.c_str());
      }
      printf("\n");
    }

    std::vector<uint64_t> errors_thread;

    const auto init = [&](const size_t num_threads) -> Status {
      // Reduce false sharing by only writing every 8th slot (64 bytes).
      errors_thread.resize(8 * num_threads);
      return true;
    };
    const auto do_task = [&](const uint32_t i, const size_t thread) -> Status {
      Task& t = (*tasks)[i];
      const PackedPixelFile& image = loaded_images[t.idx_image];
      t.image = &image;
      std::vector<uint8_t> compressed;
      if (!DoCompress(fnames[t.idx_image], image, extra_metrics_commands,
                      t.codec.get(), inner_pools[thread]->get(), &compressed,
                      &t.stats)) {
        t.stats.total_errors++;
      } else if (!printer.TaskDone(i, t)) {
        t.stats.total_errors++;
      }
      errors_thread[8 * thread] += t.stats.total_errors;
      return true;
    };
    JPEGXL_TOOLS_CHECK(jxl::RunOnPool(pool, 0, tasks->size(), init, do_task,
                                      "Benchmark tasks"));
    if (Args()->show_progress) fprintf(stderr, "\n");
    return std::accumulate(errors_thread.begin(), errors_thread.end(),
                           static_cast<size_t>(0));
  }
};

int BenchmarkMain(int argc, const char** argv) {
  fprintf(stderr, "benchmark_xl %s\n",
          jpegxl::tools::CodecConfigString(JxlDecoderVersion()).c_str());

  JPEGXL_TOOLS_CHECK(Args()->AddCommandLineOptions());

  if (!Args()->Parse(argc, argv)) {
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (Args()->cmdline.HelpFlagPassed()) {
    Args()->PrintHelp();
    return EXIT_SUCCESS;
  }
  if (!Args()->ValidateArgs()) {
    fprintf(stderr, "Use '%s -h' for more information\n", argv[0]);
    return EXIT_FAILURE;
  }
  return Benchmark::Run() ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace
}  // namespace tools
}  // namespace jpegxl

int main(int argc, const char** argv) {
  return jpegxl::tools::BenchmarkMain(argc, argv);
}
