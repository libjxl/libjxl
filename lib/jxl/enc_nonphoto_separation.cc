// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_nonphoto_separation.h"

#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

#include <algorithm>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

constexpr size_t kSeparationRes = 32;
constexpr uint16_t kNonPhotoThreshold =
    0.2 * 5 * kSeparationRes * kSeparationRes;

float FindSeparation(const Image3F& opsin, ImageU* separation,
                     const CompressParams& cparams, ThreadPool* pool) {
  ImageU sep(DivCeil(opsin.xsize(), kSeparationRes),
             DivCeil(opsin.ysize(), kSeparationRes));
  separation->Swap(sep);
  ZeroFillImage(separation);
  static const float scale_factor[3] = {1.f / 32768.0f, 1.f / 2048.0f,
                                        1.f / 2048.0f};

  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 1; y < opsin.ysize(); y++) {
      const float* JXL_RESTRICT p = opsin.PlaneRow(c, y);
      const float* JXL_RESTRICT ptop = opsin.PlaneRow(c, y - 1);
      uint16_t* JXL_RESTRICT s = separation->Row(y / kSeparationRes);
      for (size_t x = 1; x < opsin.xsize(); x++) {
        // absolute gradient residual
        float agr = std::abs(p[x - 1] + ptop[x] - ptop[x - 1] - p[x]);
        // if not smooth (near-zero) but also not strong edge: likely
        // photographic
        if (agr > 0.5f * scale_factor[c] && agr < 100.f * scale_factor[c]) {
          if (s[x / kSeparationRes] < 0xfffe)
            s[x / kSeparationRes] += (c == 1 ? 1 : 2);
        }
      }
    }
  }
  // avoid small islands and count photo vs nonphoto
  // do a few iterations of 'Game of Life' to remove islands/holes
  constexpr int kIters = 10;
  size_t photo = 0, nonphoto = 0;
  for (int i = 0; i < kIters; i++) {
    for (size_t y = 0; y < separation->ysize(); y++) {
      uint16_t* JXL_RESTRICT s = separation->Row(y);
      uint16_t* JXL_RESTRICT sp = separation->Row(y > 0 ? y - 1 : 0);
      uint16_t* JXL_RESTRICT sn =
          separation->Row(y + 1 < separation->ysize() ? y + 1 : y);
      for (size_t x = 0; x < separation->xsize(); x++) {
        bool W = (x > 0 ? s[x - 1] : s[x]) < kNonPhotoThreshold;
        bool N = sp[x] < kNonPhotoThreshold;
        bool NW = (x > 0 ? sp[x - 1] : sp[x]) < kNonPhotoThreshold;
        bool E = (x + 1 < separation->xsize() ? s[x + 1] : s[x]) <
                 kNonPhotoThreshold;
        bool NE = (x + 1 < separation->xsize() ? sp[x + 1] : sp[x]) <
                  kNonPhotoThreshold;
        bool S = sn[x] < kNonPhotoThreshold;
        bool SW = (x > 0 ? sn[x - 1] : sn[x]) < kNonPhotoThreshold;
        bool SE = (x + 1 < separation->xsize() ? sn[x + 1] : sn[x]) <
                  kNonPhotoThreshold;
        int count = W + N + NW + E + NE + S + SW + SE;
        if (count <= 2) s[x] = 0xffff;  // most neighbors are photo
        if (count >= 7) s[x] = 0;       // most neighbors are nonphoto
        if (!W && !E && count <= 5) s[x] = 0xffff;
        if (!N && !S && count <= 5) s[x] = 0xffff;
        if (i == kIters - 1) {
          if (s[x] < kNonPhotoThreshold) {
            s[x] = 0;
            nonphoto++;
          } else {
            s[x] = 0xffff;
            photo++;
          }
        }
      }
    }
  }

  return (nonphoto * 1.f / (nonphoto + photo));
}

void EncodeAndSubtract(Image3F* JXL_RESTRICT opsin, const ImageU& separation,
                       const CompressParams& orig_cparams,
                       PassesEncoderState* JXL_RESTRICT state, ThreadPool* pool,
                       AuxOut* aux_out) {
  size_t xsize = opsin->xsize();
  size_t ysize = opsin->ysize();
  Image3F nonphoto(xsize, ysize);
  ZeroFillImage(&nonphoto);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; y++) {
      const uint16_t* JXL_RESTRICT s = separation.Row(y / kSeparationRes);
      float* JXL_RESTRICT p = opsin->PlaneRow(c, y);
      float* JXL_RESTRICT np = nonphoto.PlaneRow(c, y);
      for (size_t x = 0; x < xsize; x++) {
        if (s[x / kSeparationRes] < kNonPhotoThreshold) {
          np[x] = p[x];
        }
      }
    }
  }
  CompressParams cparams = orig_cparams;
  // TODO(jon): tweak quality settings
  cparams.quality_pair.first = 75.f - cparams.butteraugli_distance * 12.f;
  cparams.quality_pair.second = 100.f - cparams.butteraugli_distance * 4.f;
  RoundtripPatchFrame(&nonphoto, state, 2, cparams, pool, true);

  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < ysize; y++) {
      float* JXL_RESTRICT p = opsin->PlaneRow(c, y);
      const float* JXL_RESTRICT np =
          state->shared.reference_frames[2].frame->color()->ConstPlaneRow(c, y);
      for (size_t x = 0; x < xsize; x++) {
        p[x] -= np[x];
      }
    }
  }

  std::vector<PatchPosition> position;
  PatchReferencePosition ref_pos;
  ref_pos.xsize = xsize;
  ref_pos.ysize = ysize;
  ref_pos.x0 = 0;
  ref_pos.y0 = 0;
  ref_pos.ref = 2;
  // Add color channels, ignore other channels.
  std::vector<PatchBlending> blending_info(
      state->shared.metadata->m.extra_channel_info.size() + 1,
      PatchBlending{PatchBlendMode::kNone, 0, false});
  blending_info[0].mode = PatchBlendMode::kAdd;
  position.emplace_back(PatchPosition{0, 0, blending_info, ref_pos});

  PatchDictionaryEncoder::SetPositions(&state->shared.image_features.patches,
                                       std::move(position));
}

}  // namespace jxl
