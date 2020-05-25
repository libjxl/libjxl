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

#ifndef JXL_MODULAR_TRANSFORM_SUBSAMPLE_H_
#define JXL_MODULAR_TRANSFORM_SUBSAMPLE_H_

#include "jxl/base/status.h"
#include "jxl/modular/image/image.h"

namespace jxl {

// JPEG-style (chroma) subsampling. Parameters are: [begin_channel],
// [end_channel], [sample_ratio_h], [sample_ratio_v], ... e.g. 1, 2, 2
// corresponds to 4:2:0

Status CheckSubsampleParameters(TransformParams* parameters, int num_channels) {
  if (parameters->size() == 0) {
    parameters->push_back(0);
  }
  if (parameters->size() == 1) {
    // special case: abbreviated parameters for some common cases
    switch ((*parameters)[0]) {
      case 0:  // 4:2:0
        (*parameters)[0] = 1;
        parameters->push_back(2);
        parameters->push_back(1);
        parameters->push_back(1);
        break;
      case 1:  // 4:2:2
        (*parameters)[0] = 1;
        parameters->push_back(2);
        parameters->push_back(1);
        parameters->push_back(0);
        break;
      case 2:  // 4:4:0
        (*parameters)[0] = 1;
        parameters->push_back(2);
        parameters->push_back(0);
        parameters->push_back(1);
        break;
      case 3:  // 4:1:1
        (*parameters)[0] = 1;
        parameters->push_back(2);
        parameters->push_back(2);
        parameters->push_back(0);
        break;
      default:
        return JXL_FAILURE("Invalid abbreviated value");
        break;
    }
  }
  if (parameters->size() % 4) {
    return JXL_FAILURE("Error: invalid parameters for subsampling.\n");
  }
  for (size_t i = 0; i < parameters->size(); i += 4) {
    int c1 = (*parameters)[i];
    int c2 = (*parameters)[i + 1];
    // The range is including c1 and c2, so c2 may not be num_channels.
    if (c1 < 0 || c1 > num_channels || c2 < 0 || c2 >= num_channels ||
        c2 < c1) {
      return JXL_FAILURE("Invalid channel range");
    }
    int shift1 = (*parameters)[i + 2];
    int shift2 = (*parameters)[i + 3];
    if (shift1 < 0 || shift1 > 30 || shift2 < 0 || shift2 > 30) {
      return JXL_FAILURE("Invalid shift value");
    }
  }

  return true;
}

Status InvSubsample(Image& input, const TransformParams& parameters) {
  TransformParams copy_parameters(parameters);
  JXL_RETURN_IF_ERROR(
      CheckSubsampleParameters(&copy_parameters, input.channel.size()));

  for (size_t i = 0; i < copy_parameters.size(); i += 4) {
    uint32_t c1 = copy_parameters[i + 0];
    uint32_t c2 = copy_parameters[i + 1];
    uint32_t tsrh = copy_parameters[i + 2];
    uint32_t tsrv = copy_parameters[i + 3];
    while (tsrh || tsrv) {
      uint32_t srh = 1, srv = 1;
      if (tsrh > 0) {
        srh = 2;
        tsrh--;
      }
      if (tsrv > 0) {
        srv = 2;
        tsrv--;
      }
      for (uint32_t c = c1; c <= c2; c++) {
        size_t ow = input.channel[c].w;
        size_t oh = input.channel[c].h;
        if (ow >= input.channel[input.nb_meta_channels].w &&
            oh >= input.channel[input.nb_meta_channels].h) {
          // this can happen in case of LQIP and 1:16 scale decodes
          JXL_DEBUG_V(
              5,
              "Skipping upscaling of channel %d because it is already as "
              "large as channel %zu.",
              c, input.nb_meta_channels);
          continue;
        }
        Channel channel(ow * srh, oh * srv);

        // 'fancy' horizontal upscale
        if (srh == 2) {
          for (size_t y = 0; y < oh; y++) {
            const pixel_type* JXL_RESTRICT in_p = input.channel[c].Row(y);
            pixel_type* JXL_RESTRICT out_p = channel.Row(y * srv);
            for (size_t x = 0; x < ow; x++) {
              out_p[x * srh] = (3 * in_p[x] + in_p[x ? x - 1 : 0] + 1) >> 2;
              out_p[x * srh + 1] =
                  (3 * in_p[x] + in_p[x + 1 < ow ? x + 1 : x] + 2) >> 2;
            }
          }
        } else {
          for (size_t y = 0; y < oh; y++) {
            const pixel_type* JXL_RESTRICT in_p = input.channel[c].Row(y);
            pixel_type* JXL_RESTRICT out_p = channel.Row(y * srv);
            for (size_t x = 0; x < ow; x++) {
              out_p[x] = in_p[x];
            }
          }
        }
        if (srv == 2) {
          Channel nchannel(ow * srh, oh * srv);
          intptr_t onerow =
              channel.plane
                  .PixelsPerRow();  // is equal for channel and nchannel since
                                    // they have the same width
          for (size_t y = 0; y < oh; y++) {
            const pixel_type* JXL_RESTRICT in_p = channel.Row(y * srv);
            pixel_type* JXL_RESTRICT out_p = nchannel.Row(y * srv);
            for (size_t x = 0; x < ow * srh; x++) {
              out_p[x] =
                  (3 * in_p[x] +
                   in_p[y ? static_cast<ssize_t>(x) - onerow * srv : x] + 1) >>
                  2;
              out_p[x + onerow] =
                  (3 * in_p[x] + in_p[y + 1 < oh ? x + onerow * srv : x] + 2) >>
                  2;
            }
          }
          channel = std::move(nchannel);
        }

        JXL_DEBUG_V(5, "Upscaled channel %i from %zux%zu to %zux%zu", c,
                    input.channel[c].w, input.channel[c].h, channel.w,
                    channel.h);
        input.channel[c] = std::move(channel);
      }
    }
  }
  return true;
}

Status FwdSubsample(Image& /* input */,
                    const TransformParams& /* parameters */) {
  return false;  // TODO (not really needed though; subsampling is useful if the
                 // input data is a JPEG or YUV, but then the transform is
                 // already done) for non-subsampled input it's probably better
                 // to just stick to 4:4:4 (and quantize most of the chroma
                 // details away)
}

Status MetaSubsample(Image& input, const TransformParams& parameters) {
  TransformParams copy_parameters(parameters);
  JXL_RETURN_IF_ERROR(
      CheckSubsampleParameters(&copy_parameters, input.channel.size()));
  for (size_t i = 0; i < copy_parameters.size(); i += 4) {
    uint32_t c1 = copy_parameters[i + 0];
    uint32_t c2 = copy_parameters[i + 1];
    uint32_t srh = copy_parameters[i + 2];
    uint32_t srv = copy_parameters[i + 3];
    for (uint32_t c = c1; c <= c2; c++) {
      input.channel[c].w += (1u << srh) - 1;
      input.channel[c].w <<= srh;
      input.channel[c].h += (1u << srv) - 1;
      input.channel[c].h <<= srv;
      input.channel[c].hshift += srh;
      input.channel[c].vshift += srv;
    }
  }
  return true;
}

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_SUBSAMPLE_H_
