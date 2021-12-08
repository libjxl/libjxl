// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/render_pipeline.h"

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "lib/extras/codec.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/fake_parallel_runner_testonly.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/render_pipeline/test_render_pipeline_stages.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testdata.h"

namespace jxl {
namespace {

TEST(RenderPipelineTest, Build) {
  RenderPipeline::Builder builder(/*num_c=*/1);
  builder.AddStage(jxl::make_unique<UpsampleXSlowStage>());
  builder.AddStage(jxl::make_unique<UpsampleYSlowStage>());
  builder.AddStage(jxl::make_unique<Check0FinalStage>());
  builder.UseSimpleImplementation();
  FrameDimensions frame_dimensions;
  frame_dimensions.Set(/*xsize=*/1024, /*ysize=*/1024, /*group_size_shift=*/0,
                       /*max_hshift=*/0, /*max_vshift=*/0,
                       /*modular_mode=*/false, /*upsampling=*/1);
  std::move(builder).Finalize(frame_dimensions);
}

TEST(RenderPipelineTest, CallAllGroups) {
  RenderPipeline::Builder builder(/*num_c=*/1);
  builder.AddStage(jxl::make_unique<UpsampleXSlowStage>());
  builder.AddStage(jxl::make_unique<UpsampleYSlowStage>());
  builder.AddStage(jxl::make_unique<Check0FinalStage>());
  builder.UseSimpleImplementation();
  FrameDimensions frame_dimensions;
  frame_dimensions.Set(/*xsize=*/1024, /*ysize=*/1024, /*group_size_shift=*/0,
                       /*max_hshift=*/0, /*max_vshift=*/0,
                       /*modular_mode=*/false, /*upsampling=*/1);
  auto pipeline = std::move(builder).Finalize(frame_dimensions);
  pipeline->PrepareForThreads(1);

  for (size_t i = 0; i < frame_dimensions.num_groups; i++) {
    const auto& input_buffers = pipeline->GetInputBuffers(i, 0);
    FillPlane(0.0f, input_buffers.GetBuffer(0).first,
              input_buffers.GetBuffer(0).second);
  }

  EXPECT_TRUE(pipeline->ReceivedAllInput());
}

struct RenderPipelineTestInputSettings {
  // Input image.
  std::string input_path;
  size_t xsize, ysize;
  // Encoding settings.
  CompressParams cparams;
  // Short name for the encoder settings.
  std::string cparams_descr;
};

class RenderPipelineTestParam
    : public ::testing::TestWithParam<RenderPipelineTestInputSettings> {};

TEST_P(RenderPipelineTestParam, PipelineTest) {
  RenderPipelineTestInputSettings config = GetParam();

  // Use a parallel runner that randomly shuffles tasks to detect possible
  // border handling bugs.
  FakeParallelRunner fake_pool(/*order_seed=*/123, /*num_threads=*/8);
  ThreadPool pool(&JxlFakeParallelRunner, &fake_pool);
  const PaddedBytes orig = ReadTestData(config.input_path);

  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  io.ShrinkTo(config.xsize, config.ysize);

  PaddedBytes compressed;

  PassesEncoderState enc_state;
  ASSERT_TRUE(EncodeFile(config.cparams, &io, &enc_state, &compressed,
                         GetJxlCms(), /*aux_out=*/nullptr, &pool));

  DecompressParams dparams;

  CodecInOut io_default;
  ASSERT_TRUE(DecodeFile(dparams, compressed, &io_default, &pool));
  CodecInOut io_slow_pipeline;
  dparams.use_slow_render_pipeline = true;
  ASSERT_TRUE(DecodeFile(dparams, compressed, &io_slow_pipeline, &pool));

  ASSERT_EQ(io_default.frames.size(), io_slow_pipeline.frames.size());
  for (size_t i = 0; i < io_default.frames.size(); i++) {
    VerifyRelativeError(*io_default.frames[i].color(),
                        *io_slow_pipeline.frames[i].color(), 1e-5, 1e-5);
    for (size_t ec = 0; ec < io_default.frames[i].extra_channels().size();
         ec++) {
      VerifyRelativeError(io_default.frames[i].extra_channels()[ec],
                          io_slow_pipeline.frames[i].extra_channels()[ec], 1e-5,
                          1e-5);
    }
  }
}

std::vector<RenderPipelineTestInputSettings> GeneratePipelineTests() {
  std::vector<RenderPipelineTestInputSettings> all_tests;

  for (size_t size : {128, 256, 258, 777}) {
    RenderPipelineTestInputSettings settings;
    settings.input_path = "imagecompression.info/flower_foveon.png";
    settings.xsize = size;
    settings.ysize = size;

    // Base settings.
    settings.cparams.butteraugli_distance = 1.0;
    settings.cparams.patches = Override::kOff;
    settings.cparams.dots = Override::kOff;
    settings.cparams.gaborish = Override::kOff;
    settings.cparams.epf = 0;
    settings.cparams.color_transform = ColorTransform::kXYB;

    {
      auto s = settings;
      s.cparams_descr = "NoGabNoEpfNoPatches";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.color_transform = ColorTransform::kNone;
      s.cparams_descr = "NoGabNoEpfNoPatchesNoXYB";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.gaborish = Override::kOn;
      s.cparams_descr = "GabNoEpfNoPatches";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.epf = 1;
      s.cparams_descr = "NoGabEpf1NoPatches";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.epf = 2;
      s.cparams_descr = "NoGabEpf2NoPatches";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.epf = 3;
      s.cparams_descr = "NoGabEpf3NoPatches";
      all_tests.push_back(s);
    }

    {
      auto s = settings;
      s.cparams.gaborish = Override::kOn;
      s.cparams.epf = 3;
      s.cparams_descr = "GabEpf3NoPatches";
      all_tests.push_back(s);
    }
  }

  return all_tests;
}

std::ostream& operator<<(std::ostream& os,
                         const RenderPipelineTestInputSettings& c) {
  std::string filename;
  size_t pos = c.input_path.find_last_of('/');
  if (pos == std::string::npos) {
    filename = c.input_path;
  } else {
    filename = c.input_path.substr(pos + 1);
  }
  std::replace_if(
      filename.begin(), filename.end(), [](char c) { return !isalnum(c); },
      '_');
  os << filename << "_" << c.xsize << "x" << c.ysize << "_" << c.cparams_descr;
  return os;
}

std::string PipelineTestDescription(
    const testing::TestParamInfo<RenderPipelineTestParam::ParamType>& info) {
  std::stringstream name;
  name << info.param;
  return name.str();
}

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(RenderPipelineTest, RenderPipelineTestParam,
                                   testing::ValuesIn(GeneratePipelineTests()),
                                   PipelineTestDescription);

}  // namespace
}  // namespace jxl
