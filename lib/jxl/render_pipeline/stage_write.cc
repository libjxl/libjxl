// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_write.h"

#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "lib/jxl/alpha.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_write.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Clamp;
using hwy::HWY_NAMESPACE::Div;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::NearestInt;
using hwy::HWY_NAMESPACE::Or;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::RebindToSigned;
using hwy::HWY_NAMESPACE::RebindToUnsigned;
using hwy::HWY_NAMESPACE::ShiftLeftSame;
using hwy::HWY_NAMESPACE::ShiftRightSame;
using hwy::HWY_NAMESPACE::VFromD;

// Custom 32x32 blue noise dithering pattern made by DZgas
// (-0.5,0.5) are scaled to (-0.49219,0.49219) to prevent rounding.
// Rows are padded to 48 (32 + 16) to allow SIMD to wrap around horizontally
const float kDither[48 * 32] = {
	-0.24585, 0.38249, -0.44504, 0.13135, -0.32764, -0.11306, -0.39981, 0.32187,
	-0.25548, -0.04763, 0.14482, -0.45274, 0.33823, 0.23623, -0.10537, 0.08420,
	-0.23816, 0.41232, -0.07361, -0.27857, 0.15444, 0.38442, -0.04859, -0.34015,
	0.26606, 0.00625, -0.19774, 0.49123, -0.34593, 0.31706, -0.45466, -0.04474,
	-0.24585, 0.38249, -0.44504, 0.13135, -0.32764, -0.11306, -0.39981, 0.32187,
	-0.25548, -0.04763, 0.14482, -0.45274, 0.33823, 0.23623, -0.10537, 0.08420,
	0.27472, -0.31417, 0.01010, 0.44889, -0.17850, 0.40077, 0.17368, 0.04474,
	0.43349, -0.31995, 0.37383, -0.09478, -0.27087, 0.01780, -0.38923, 0.34881,
	0.19678, -0.44023, 0.03897, 0.48449, -0.18619, -0.39885, 0.07361, 0.46332,
	-0.17753, 0.35362, -0.43638, 0.05629, 0.21699, -0.17080, 0.16406, -0.29012,
	0.27472, -0.31417, 0.01010, 0.44889, -0.17850, 0.40077, 0.17368, 0.04474,
	0.43349, -0.31995, 0.37383, -0.09478, -0.27087, 0.01780, -0.38923, 0.34881,
	0.08901, -0.14578, 0.25932, -0.27472, 0.22180, -0.46813, -0.05340, -0.37768,
	-0.13808, 0.20255, -0.21410, 0.06399, 0.48738, 0.15829, -0.20255, -0.01684,
	-0.16984, 0.12172, 0.30647, -0.31225, 0.18235, 0.28819, -0.09189, -0.49123,
	0.13904, -0.30070, 0.17561, -0.05822, -0.40270, 0.40462, -0.10344, 0.43926,
	0.08901, -0.14578, 0.25932, -0.27472, 0.22180, -0.46813, -0.05340, -0.37768,
	-0.13808, 0.20255, -0.21410, 0.06399, 0.48738, 0.15829, -0.20255, -0.01684,
	-0.46524, 0.29781, -0.39308, -0.06880, 0.07554, 0.35555, -0.22661, 0.46909,
	0.09767, -0.00625, -0.48353, 0.25163, -0.36710, -0.08323, 0.39789, -0.47679,
	0.44311, -0.34978, -0.21506, -0.03031, -0.42868, -0.15348, 0.40270, -0.22757,
	0.30840, -0.02165, 0.43445, -0.24104, 0.27953, -0.01395, -0.35459, 0.01876,
	-0.46524, 0.29781, -0.39308, -0.06880, 0.07554, 0.35555, -0.22661, 0.46909,
	0.09767, -0.00625, -0.48353, 0.25163, -0.36710, -0.08323, 0.39789, -0.47679,
	0.18812, -0.19485, 0.48160, 0.15252, -0.34304, 0.02839, 0.26798, -0.41713,
	0.29974, -0.28338, 0.41906, -0.15059, 0.31321, -0.30263, 0.10537, 0.26317,
	0.00241, 0.22565, 0.35940, 0.03031, 0.42772, 0.09959, -0.36325, 0.21121,
	0.02357, -0.38634, -0.13904, 0.11787, -0.48064, 0.36806, -0.23142, 0.33727,
	0.18812, -0.19485, 0.48160, 0.15252, -0.34304, 0.02839, 0.26798, -0.41713,
	0.29974, -0.28338, 0.41906, -0.15059, 0.31321, -0.30263, 0.10537, 0.26317,
	-0.37383, 0.06880, -0.03608, -0.21987, 0.41425, -0.12942, -0.30647, -0.02261,
	-0.18331, 0.23142, 0.13423, -0.43060, -0.03416, 0.19004, -0.26029, -0.12365,
	-0.32379, -0.05629, -0.46332, 0.14289, -0.25066, -0.06688, 0.33246, -0.26702,
	-0.11980, 0.25259, 0.39211, -0.33149, 0.22853, -0.16118, 0.12750, -0.07457,
	-0.37383, 0.06880, -0.03608, -0.21987, 0.41425, -0.12942, -0.30647, -0.02261,
	-0.18331, 0.23142, 0.13423, -0.43060, -0.03416, 0.19004, -0.26029, -0.12365,
	-0.25451, 0.37961, -0.42194, 0.24874, -0.48834, 0.18523, 0.34689, 0.11403,
	0.39019, -0.35266, 0.01299, -0.10825, 0.36710, -0.40366, 0.45658, 0.05340,
	0.32668, -0.14289, 0.46717, -0.28627, 0.24778, -0.40847, 0.04859, 0.45177,
	0.14963, -0.45177, 0.07746, -0.08516, 0.03801, 0.32091, -0.31706, 0.46043,
	-0.25451, 0.37961, -0.42194, 0.24874, -0.48834, 0.18523, 0.34689, 0.11403,
	0.39019, -0.35266, 0.01299, -0.10825, 0.36710, -0.40366, 0.45658, 0.05340,
	0.24104, -0.11691, 0.14193, -0.15733, 0.32861, -0.26606, -0.10055, -0.44792,
	-0.21025, 0.04763, 0.47583, -0.24489, 0.08035, 0.27183, -0.19004, -0.42098,
	0.16310, -0.37191, 0.06591, 0.27664, -0.11499, 0.37287, -0.16502, -0.30936,
	-0.04186, 0.34593, -0.27568, 0.47775, -0.20736, -0.41617, 0.20159, 0.00337,
	0.24104, -0.11691, 0.14193, -0.15733, 0.32861, -0.26606, -0.10055, -0.44792,
	-0.21025, 0.04763, 0.47583, -0.24489, 0.08035, 0.27183, -0.19004, -0.42098,
	0.10152, -0.29685, 0.44408, 0.03320, -0.36517, 0.05725, 0.16599, 0.43157,
	-0.06206, 0.27857, -0.46043, 0.17753, -0.33342, -0.07169, 0.21314, 0.38827,
	-0.23623, 0.35074, -0.01107, -0.33630, 0.18716, -0.47872, 0.10729, 0.28723,
	-0.37672, -0.18812, 0.16984, -0.43253, 0.11210, 0.41713, -0.06110, -0.47294,
	0.10152, -0.29685, 0.44408, 0.03320, -0.36517, 0.05725, 0.16599, 0.43157,
	-0.06206, 0.27857, -0.46043, 0.17753, -0.33342, -0.07169, 0.21314, 0.38827,
	0.35747, -0.02646, -0.43830, 0.30551, -0.08035, 0.49026, -0.24008, -0.39596,
	0.20833, -0.28819, 0.32379, -0.16021, 0.40943, -0.29781, 0.02069, -0.45562,
	-0.05052, 0.12846, -0.19967, 0.42098, -0.07939, -0.21891, 0.48641, 0.01395,
	0.19774, 0.44119, -0.00433, 0.26991, -0.12654, -0.26125, 0.29493, -0.18235,
	0.35747, -0.02646, -0.43830, 0.30551, -0.08035, 0.49026, -0.24008, -0.39596,
	0.20833, -0.28819, 0.32379, -0.16021, 0.40943, -0.29781, 0.02069, -0.45562,
	-0.33919, 0.26414, -0.20351, 0.08708, -0.32283, 0.22661, -0.00241, 0.36132,
	-0.12172, 0.12365, -0.38345, -0.03993, 0.14770, -0.17465, 0.29204, 0.09574,
	0.48064, -0.39211, 0.21891, -0.43734, 0.08612, 0.23719, -0.34400, -0.02454,
	-0.45851, -0.24778, -0.10633, -0.35940, 0.38634, 0.06495, -0.39019, 0.15733,
	-0.33919, 0.26414, -0.20351, 0.08708, -0.32283, 0.22661, -0.00241, 0.36132,
	-0.12172, 0.12365, -0.38345, -0.03993, 0.14770, -0.17465, 0.29204, 0.09574,
	0.42675, -0.09671, 0.19485, -0.27953, 0.39596, -0.41040, -0.16791, 0.09189,
	-0.25740, 0.45370, 0.00722, 0.34304, -0.47390, 0.43638, -0.35747, -0.12750,
	-0.27280, 0.31802, -0.15636, 0.39981, -0.29397, -0.09959, 0.30263, -0.16695,
	0.40847, 0.05244, 0.31513, 0.13231, -0.30359, -0.04667, 0.47102, -0.14386,
	0.42675, -0.09671, 0.19485, -0.27953, 0.39596, -0.41040, -0.16791, 0.09189,
	-0.25740, 0.45370, 0.00722, 0.34304, -0.47390, 0.43638, -0.35747, -0.12750,
	-0.00914, -0.37961, 0.36902, -0.13712, 0.13616, -0.04956, 0.29012, -0.47968,
	0.25355, -0.34112, -0.19870, 0.06976, -0.09574, 0.19582, -0.01491, 0.24200,
	0.03608, -0.22372, 0.16791, -0.03801, 0.04378, 0.36421, -0.41906, 0.15155,
	-0.31802, 0.20929, -0.40462, 0.02550, 0.34015, -0.22276, 0.22083, -0.44408,
	-0.00914, -0.37961, 0.36902, -0.13712, 0.13616, -0.04956, 0.29012, -0.47968,
	0.25355, -0.34112, -0.19870, 0.06976, -0.09574, 0.19582, -0.01491, 0.24200,
	0.31129, 0.17176, 0.04956, -0.46236, 0.46428, -0.22950, 0.02261, 0.42291,
	-0.08901, 0.18138, 0.38345, -0.40077, 0.26029, -0.23238, -0.44215, 0.37672,
	-0.32187, 0.44792, -0.48930, 0.28146, -0.38153, -0.24297, 0.09093, 0.46236,
	-0.07650, -0.19678, 0.37479, -0.14963, -0.48449, 0.25644, -0.28434, 0.09478,
	0.31129, 0.17176, 0.04956, -0.46236, 0.46428, -0.22950, 0.02261, 0.42291,
	-0.08901, 0.18138, 0.38345, -0.40077, 0.26029, -0.23238, -0.44215, 0.37672,
	-0.17561, -0.32957, -0.07072, 0.23527, -0.31129, 0.21506, -0.36036, 0.10825,
	-0.43445, -0.02069, -0.27761, 0.30166, -0.14482, 0.46140, 0.13808, -0.08612,
	0.07265, -0.11018, 0.11018, -0.18523, 0.17946, 0.32957, -0.13327, -0.47006,
	0.00529, 0.26125, -0.35074, 0.18331, 0.44985, -0.01780, -0.09093, 0.40655,
	-0.17561, -0.32957, -0.07072, 0.23527, -0.31129, 0.21506, -0.36036, 0.10825,
	-0.43445, -0.02069, -0.27761, 0.30166, -0.14482, 0.46140, 0.13808, -0.08612,
	0.01684, 0.48545, -0.21602, 0.33630, -0.03224, -0.19101, 0.31610, -0.13135,
	0.40174, -0.18042, 0.04186, 0.11691, -0.30840, -0.05918, -0.36902, 0.30936,
	-0.41425, 0.33919, -0.34881, 0.42579, -0.06495, -0.28146, 0.23046, 0.12269,
	0.43253, -0.26414, -0.05244, 0.07072, -0.23334, -0.36421, 0.14578, -0.41232,
	0.01684, 0.48545, -0.21602, 0.33630, -0.03224, -0.19101, 0.31610, -0.13135,
	0.40174, -0.18042, 0.04186, 0.11691, -0.30840, -0.05918, -0.36902, 0.30936,
	0.28338, -0.49219, 0.12076, -0.38730, 0.06303, 0.44600, -0.45370, 0.16021,
	-0.32476, 0.24008, 0.47487, -0.42387, 0.35459, 0.02646, 0.22276, -0.25355,
	0.15636, -0.20833, 0.24585, 0.01107, -0.44985, 0.47294, -0.02839, -0.39500,
	-0.21314, 0.29108, -0.42579, 0.39404, 0.24297, 0.03993, 0.35170, -0.25163,
	0.28338, -0.49219, 0.12076, -0.38730, 0.06303, 0.44600, -0.45370, 0.16021,
	-0.32476, 0.24008, 0.47487, -0.42387, 0.35459, 0.02646, 0.22276, -0.25355,
	-0.15829, 0.07939, 0.38153, -0.11884, 0.26702, -0.26798, -0.07746, 0.34496,
	0.08227, -0.37576, -0.00337, -0.20544, 0.20544, -0.46717, 0.41328, -0.15155,
	0.49219, -0.00722, -0.31321, 0.39115, -0.14097, 0.05533, -0.32668, 0.35651,
	0.08131, -0.11403, 0.16214, -0.29108, -0.03512, -0.45755, -0.10729, 0.20351,
	-0.15829, 0.07939, 0.38153, -0.11884, 0.26702, -0.26798, -0.07746, 0.34496,
	0.08227, -0.37576, -0.00337, -0.20544, 0.20544, -0.46717, 0.41328, -0.15155,
	0.43734, -0.34689, -0.23719, 0.19197, -0.42772, 0.15059, 0.03127, -0.16406,
	-0.24874, 0.28531, -0.10440, 0.42964, -0.26317, 0.09863, -0.04571, -0.29300,
	0.06206, -0.48160, 0.19967, -0.23912, 0.14386, 0.27376, -0.17176, 0.21602,
	-0.37287, 0.48834, -0.17946, 0.33342, -0.13038, 0.45755, 0.27568, -0.30455,
	0.43734, -0.34689, -0.23719, 0.19197, -0.42772, 0.15059, 0.03127, -0.16406,
	-0.24874, 0.28531, -0.10440, 0.42964, -0.26317, 0.09863, -0.04571, -0.29300,
	0.13038, -0.06303, 0.32572, 0.00048, 0.41136, -0.29493, 0.47968, -0.40559,
	0.37094, 0.18908, -0.48641, 0.13327, -0.13520, 0.33149, -0.38827, 0.26510,
	0.36613, -0.11787, 0.29589, -0.05437, -0.26991, -0.42964, 0.32283, -0.09382,
	0.03224, 0.11884, -0.44696, 0.17657, -0.33438, 0.10344, -0.39789, 0.00818,
	0.13038, -0.06303, 0.32572, 0.00048, 0.41136, -0.29493, 0.47968, -0.40559,
	0.37094, 0.18908, -0.48641, 0.13327, -0.13520, 0.33149, -0.38827, 0.26510,
	0.42002, -0.43926, -0.14770, -0.37094, -0.04378, 0.30744, -0.11210, 0.22468,
	0.05437, -0.30166, -0.06784, 0.38730, -0.33823, -0.02550, 0.17465, -0.19293,
	-0.35362, 0.12557, -0.40751, 0.45274, 0.10055, 0.38057, -0.00048, -0.46621,
	0.40366, -0.29974, -0.01588, 0.41521, 0.05918, -0.19389, 0.36325, -0.20448,
	0.42002, -0.43926, -0.14770, -0.37094, -0.04378, 0.30744, -0.11210, 0.22468,
	0.05437, -0.30166, -0.06784, 0.38730, -0.33823, -0.02550, 0.17465, -0.19293,
	-0.27183, 0.22950, 0.04571, 0.25066, -0.21121, -0.47775, 0.10633, -0.35555,
	-0.02935, 0.45562, -0.22083, 0.01588, 0.29878, -0.44600, 0.44023, 0.03705,
	-0.07265, 0.40559, -0.16214, 0.01973, -0.36613, -0.20159, 0.19101, -0.23046,
	0.23912, -0.15444, 0.30455, -0.25644, -0.08420, 0.21217, -0.47198, 0.15540,
	-0.27183, 0.22950, 0.04571, 0.25066, -0.21121, -0.47775, 0.10633, -0.35555,
	-0.02935, 0.45562, -0.22083, 0.01588, 0.29878, -0.44600, 0.44023, 0.03705,
	-0.00818, -0.31513, 0.46621, 0.07457, 0.39885, 0.17850, -0.18716, 0.27280,
	-0.15252, -0.42002, 0.25548, 0.16695, -0.24200, 0.08805, -0.17657, 0.23431,
	-0.43542, 0.20736, -0.33246, 0.34208, 0.24970, -0.10248, 0.43830, -0.04282,
	-0.34208, 0.14001, -0.41809, 0.25836, -0.37864, 0.47679, -0.05533, 0.29300,
	-0.00818, -0.31513, 0.46621, 0.07457, 0.39885, 0.17850, -0.18716, 0.27280,
	-0.15252, -0.42002, 0.25548, 0.16695, -0.24200, 0.08805, -0.17657, 0.23431,
	0.34785, -0.10152, -0.45081, -0.08708, -0.33053, 0.01203, 0.44215, -0.28531,
	0.35266, 0.11980, -0.34496, -0.09863, 0.47006, -0.31898, 0.31995, -0.28049,
	0.48257, -0.22565, 0.07650, -0.03320, -0.47583, 0.16502, -0.40174, 0.04667,
	0.46813, -0.06976, 0.37191, -0.21795, 0.02165, 0.09671, -0.35170, -0.16887,
	0.34785, -0.10152, -0.45081, -0.08708, -0.33053, 0.01203, 0.44215, -0.28531,
	0.35266, 0.11980, -0.34496, -0.09863, 0.47006, -0.31898, 0.31995, -0.28049,
	-0.40655, 0.16887, 0.37576, -0.24393, 0.32764, -0.39404, 0.14097, -0.05725,
	-0.46428, 0.06688, 0.42387, -0.01010, -0.45658, 0.15348, 0.00433, -0.08997,
	0.11306, -0.12557, 0.28434, -0.28915, 0.41809, -0.24682, 0.31225, -0.30744,
	0.18619, -0.48738, 0.08323, -0.12076, 0.31898, -0.28723, 0.38923, 0.11499,
	-0.40655, 0.16887, 0.37576, -0.24393, 0.32764, -0.39404, 0.14097, -0.05725,
	-0.46428, 0.06688, 0.42387, -0.01010, -0.45658, 0.15348, 0.00433, -0.08997,
	0.26895, -0.25932, 0.02935, 0.21795, -0.17368, -0.01876, 0.28915, -0.21699,
	0.23238, -0.12269, -0.27376, 0.27761, -0.19582, 0.37864, -0.39693, 0.34978,
	-0.46140, 0.39500, -0.38442, 0.14867, -0.08131, 0.06784, -0.13616, 0.27087,
	-0.18908, 0.34400, -0.27664, 0.43060, -0.43157, 0.20063, -0.14193, -0.04090,
	0.26895, -0.25932, 0.02935, 0.21795, -0.17368, -0.01876, 0.28915, -0.21699,
	0.23238, -0.12269, -0.27376, 0.27761, -0.19582, 0.37864, -0.39693, 0.34978,
	0.05148, 0.48353, -0.38538, -0.12461, 0.42868, -0.43349, 0.09286, 0.48930,
	-0.36806, 0.36036, 0.18427, -0.41521, -0.05148, 0.21410, -0.16599, 0.05052,
	0.19389, -0.25836, -0.01973, 0.45947, -0.21217, 0.36228, -0.44311, 0.10921,
	-0.01203, -0.36228, 0.22372, -0.07842, 0.13520, -0.32091, 0.44696, -0.48257,
	0.05148, 0.48353, -0.38538, -0.12461, 0.42868, -0.43349, 0.09286, 0.48930,
	-0.36806, 0.36036, 0.18427, -0.41521, -0.05148, 0.21410, -0.16599, 0.05052,
	-0.22468, -0.06591, 0.31032, 0.14674, -0.26510, 0.19870, -0.31032, -0.14001,
	-0.07554, 0.02454, -0.22853, 0.31417, 0.05822, -0.30551, 0.41617, -0.36132,
	-0.14674, 0.26221, 0.08997, -0.42291, 0.22757, -0.31610, 0.02742, 0.47872,
	-0.15925, 0.39693, 0.01491, -0.23527, 0.28627, -0.02357, -0.18427, 0.24682,
	-0.22468, -0.06591, 0.31032, 0.14674, -0.26510, 0.19870, -0.31032, -0.14001,
	-0.07554, 0.02454, -0.22853, 0.31417, 0.05822, -0.30551, 0.41617, -0.36132,
	0.18042, -0.33727, 0.08516, -0.47487, -0.03705, 0.39308, 0.04090, 0.33438,
	-0.49026, 0.40751, 0.12654, -0.38057, 0.45081, -0.11595, 0.13712, 0.30359,
	-0.03897, -0.32861, 0.33534, -0.18138, 0.17272, -0.06014, 0.29685, -0.39115,
	-0.25259, 0.15925, -0.41328, 0.45466, -0.45947, 0.07169, 0.35844, -0.36998,
	0.18042, -0.33727, 0.08516, -0.47487, -0.03705, 0.39308, 0.04090, 0.33438,
	-0.49026, 0.40751, 0.12654, -0.38057, 0.45081, -0.11595, 0.13712, 0.30359,
	-0.00144, 0.42194, -0.20063, 0.36517, 0.16118, -0.35844, -0.23431, 0.24489,
	0.10248, -0.29204, -0.15540, 0.25740, -0.24970, 0.03416, -0.47102, -0.20640,
	0.47390, 0.00914, -0.48545, -0.10921, 0.44504, -0.35651, 0.12942, -0.09767,
	0.25451, 0.04282, -0.03127, -0.14867, 0.19293, -0.26895, -0.11114, 0.30070,
	-0.00144, 0.42194, -0.20063, 0.36517, 0.16118, -0.35844, -0.23431, 0.24489,
	0.10248, -0.29204, -0.15540, 0.25740, -0.24970, 0.03416, -0.47102, -0.20640,
	-0.13231, -0.29878, 0.23816, -0.40943, -0.16310, 0.45851, -0.08227, -0.44119,
	0.00144, 0.46524, -0.02742, -0.42675, 0.38538, 0.17080, -0.06399, 0.21987,
	-0.29589, 0.11595, 0.36998, 0.06014, -0.26221, 0.32476, -0.22180, 0.42483,
	-0.46909, 0.37768, -0.32572, 0.33053, -0.38249, 0.41040, 0.12461, -0.42483,
	-0.13231, -0.29878, 0.23816, -0.40943, -0.16310, 0.45851, -0.08227, -0.44119,
	0.00144, 0.46524, -0.02742, -0.42675, 0.38538, 0.17080, -0.06399, 0.21987,
	0.20640, 0.06110, -0.09286, 0.34112, -0.01299, 0.11114, 0.28242, -0.19197,
	0.21025, -0.34785, 0.29397, 0.07842, -0.17272, -0.33534, 0.43542, -0.41136,
	0.28049, -0.13423, -0.37479, 0.24393, -0.00529, -0.44889, 0.20448, -0.12846,
	0.09382, -0.28242, 0.23334, -0.08805, 0.10440, -0.20929, 0.03512, 0.47198,
	0.20640, 0.06110, -0.09286, 0.34112, -0.01299, 0.11114, 0.28242, -0.19197,
	0.21025, -0.34785, 0.29397, 0.07842, -0.17272, -0.33534, 0.43542, -0.41136
};

namespace {
constexpr static size_t kChunkSize = 1024;
}  // namespace

using DF = HWY_FULL(float);

// Converts `v` to an appropriate value for the given unsigned type.
// If the unsigned type is an 8-bit type, performs ordered dithering.
template <typename T>
VFromD<Rebind<T, DF>> MakeUnsigned(VFromD<DF> v, size_t x0, size_t y0,
                                   VFromD<DF> mul, size_t c) {
  static_assert(std::is_unsigned<T>::value, "T must be an unsigned type");
  using DI32 = RebindToSigned<DF>;
  using DU32 = RebindToUnsigned<DF>;
  using DU = Rebind<T, DF>;
  v = Mul(v, mul);
  // TODO(veluca): if constexpr with C++17
  if (sizeof(T) == 1) {
    size_t x_off = (x0 + c * 23) % 32;
    size_t y_off = (y0 + c * 13) % 32;
    size_t pos = y_off * 48 + x_off;
    auto dither = LoadU(DF(), kDither + pos);
    v = Add(v, dither);
  }
  v = Clamp(Zero(DF()), v, mul);
  VFromD<DI32> ni = NearestInt(v);
  VFromD<DU32> nu = BitCast(DU32(), ni);
  return DemoteTo(DU(), nu);
}

class WriteToOutputStage : public RenderPipelineStage {
 public:
  WriteToOutputStage(const ImageOutput& main_output, size_t width,
                     size_t height, bool has_alpha, bool unpremul_alpha,
                     size_t alpha_c, Orientation undo_orientation,
                     const std::vector<ImageOutput>& extra_output,
                     JxlMemoryManager* memory_manager)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        width_(width),
        height_(height),
        main_(main_output),
        num_color_(main_.num_channels_ < 3 ? 1 : 3),
        want_alpha_(main_.num_channels_ == 2 || main_.num_channels_ == 4),
        has_alpha_(has_alpha),
        unpremul_alpha_(unpremul_alpha),
        alpha_c_(alpha_c),
        flip_x_(ShouldFlipX(undo_orientation)),
        flip_y_(ShouldFlipY(undo_orientation)),
        transpose_(ShouldTranspose(undo_orientation)),
        opaque_alpha_(kChunkSize, 1.0f),
        memory_manager_(memory_manager) {
    for (size_t ec = 0; ec < extra_output.size(); ++ec) {
      if (extra_output[ec].callback.IsPresent() || extra_output[ec].buffer) {
        Output extra(extra_output[ec]);
        extra.channel_index_ = 3 + ec;
        extra_channels_.push_back(extra);
      }
    }
  }

  WriteToOutputStage(const WriteToOutputStage&) = delete;
  WriteToOutputStage& operator=(const WriteToOutputStage&) = delete;
  WriteToOutputStage(WriteToOutputStage&&) = delete;
  WriteToOutputStage& operator=(WriteToOutputStage&&) = delete;

  ~WriteToOutputStage() override {
    if (main_.run_opaque_) {
      main_.pixel_callback_.destroy(main_.run_opaque_);
    }
    for (auto& extra : extra_channels_) {
      if (extra.run_opaque_) {
        extra.pixel_callback_.destroy(extra.run_opaque_);
      }
    }
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    JXL_ENSURE(xextra_left == 0 && xextra_right == 0);
    JXL_ENSURE(main_.run_opaque_ || main_.buffer_);
    if (ypos >= height_) return true;
    if (xpos >= width_) return true;
    if (flip_y_) {
      ypos = height_ - 1u - ypos;
    }
    size_t limit = std::min(xsize, width_ - xpos);
    for (size_t x0 = 0; x0 < limit; x0 += kChunkSize) {
      size_t xstart = xpos + x0;
      size_t len = std::min<size_t>(kChunkSize, limit - x0);

      const float* line_buffers[4];
      for (size_t c = 0; c < num_color_; c++) {
        line_buffers[c] = GetInputRow(input_rows, c, 0) + x0;
      }
      if (has_alpha_) {
        line_buffers[num_color_] = GetInputRow(input_rows, alpha_c_, 0) + x0;
      } else {
        // opaque_alpha_ is a way to set all values to 1.0f.
        line_buffers[num_color_] = opaque_alpha_.data();
      }
      if (has_alpha_ && want_alpha_ && unpremul_alpha_) {
        UnpremulAlpha(thread_id, len, line_buffers);
      }
      OutputBuffers(main_, thread_id, ypos, xstart, len, line_buffers);
      for (const auto& extra : extra_channels_) {
        line_buffers[0] = GetInputRow(input_rows, extra.channel_index_, 0) + x0;
        OutputBuffers(extra, thread_id, ypos, xstart, len, line_buffers);
      }
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    if (c < num_color_ || (has_alpha_ && c == alpha_c_)) {
      return RenderPipelineChannelMode::kInput;
    }
    for (const auto& extra : extra_channels_) {
      if (c == extra.channel_index_) {
        return RenderPipelineChannelMode::kInput;
      }
    }
    return RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "WritePixelCB"; }

 private:
  struct Output {
    explicit Output(const ImageOutput& image_out)
        : pixel_callback_(image_out.callback),
          buffer_(image_out.buffer),
          buffer_size_(image_out.buffer_size),
          stride_(image_out.stride),
          num_channels_(image_out.format.num_channels),
          swap_endianness_(SwapEndianness(image_out.format.endianness)),
          data_type_(image_out.format.data_type),
          bits_per_sample_(image_out.bits_per_sample) {}

    Status PrepareForThreads(size_t num_threads) {
      if (pixel_callback_.IsPresent()) {
        run_opaque_ =
            pixel_callback_.Init(num_threads, /*num_pixels=*/kChunkSize);
        JXL_RETURN_IF_ERROR(run_opaque_ != nullptr);
      } else {
        JXL_RETURN_IF_ERROR(buffer_ != nullptr);
      }
      return true;
    }

    PixelCallback pixel_callback_;
    void* run_opaque_ = nullptr;
    void* buffer_ = nullptr;
    size_t buffer_size_;
    size_t stride_;
    size_t num_channels_;
    bool swap_endianness_;
    JxlDataType data_type_;
    size_t bits_per_sample_;
    size_t channel_index_;  // used for extra_channels
  };

  Status PrepareForThreads(size_t num_threads) override {
    JXL_RETURN_IF_ERROR(main_.PrepareForThreads(num_threads));
    for (auto& extra : extra_channels_) {
      JXL_RETURN_IF_ERROR(extra.PrepareForThreads(num_threads));
    }
    temp_out_.resize(num_threads);
    size_t alloc_size = sizeof(float) * kChunkSize;
    for (AlignedMemory& temp : temp_out_) {
      JXL_ASSIGN_OR_RETURN(
          temp, AlignedMemory::Create(memory_manager_,
                                      alloc_size * main_.num_channels_));
    }
    if ((has_alpha_ && want_alpha_ && unpremul_alpha_) || flip_x_) {
      temp_in_.resize(num_threads * main_.num_channels_);
      for (AlignedMemory& temp : temp_in_) {
        JXL_ASSIGN_OR_RETURN(
            temp, AlignedMemory::Create(memory_manager_, alloc_size));
      }
    }
    return true;
  }
  static bool ShouldFlipX(Orientation undo_orientation) {
    return (undo_orientation == Orientation::kFlipHorizontal ||
            undo_orientation == Orientation::kRotate180 ||
            undo_orientation == Orientation::kRotate270 ||
            undo_orientation == Orientation::kAntiTranspose);
  }
  static bool ShouldFlipY(Orientation undo_orientation) {
    return (undo_orientation == Orientation::kFlipVertical ||
            undo_orientation == Orientation::kRotate180 ||
            undo_orientation == Orientation::kRotate90 ||
            undo_orientation == Orientation::kAntiTranspose);
  }
  static bool ShouldTranspose(Orientation undo_orientation) {
    return (undo_orientation == Orientation::kTranspose ||
            undo_orientation == Orientation::kRotate90 ||
            undo_orientation == Orientation::kRotate270 ||
            undo_orientation == Orientation::kAntiTranspose);
  }

  void UnpremulAlpha(size_t thread_id, size_t len,
                     const float** line_buffers) const {
    const HWY_FULL(float) d;
    auto one = Set(d, 1.0f);
    float* temp_in[4];
    for (size_t c = 0; c < main_.num_channels_; ++c) {
      size_t tix = thread_id * main_.num_channels_ + c;
      temp_in[c] = temp_in_[tix].address<float>();
      memcpy(temp_in[c], line_buffers[c], sizeof(float) * len);
    }
    auto small_alpha = Set(d, kSmallAlpha);
    for (size_t ix = 0; ix < len; ix += Lanes(d)) {
      auto alpha = LoadU(d, temp_in[num_color_] + ix);
      auto mul = Div(one, Max(small_alpha, alpha));
      for (size_t c = 0; c < num_color_; ++c) {
        auto val = LoadU(d, temp_in[c] + ix);
        StoreU(Mul(val, mul), d, temp_in[c] + ix);
      }
    }
    for (size_t c = 0; c < main_.num_channels_; ++c) {
      line_buffers[c] = temp_in[c];
    }
  }

  void OutputBuffers(const Output& out, size_t thread_id, size_t ypos,
                     size_t xstart, size_t len, const float* input[4]) const {
    if (flip_x_) {
      FlipX(out, thread_id, len, &xstart, input);
    }
    if (out.data_type_ == JXL_TYPE_UINT8) {
      uint8_t* JXL_RESTRICT temp = temp_out_[thread_id].address<uint8_t>();
      StoreUnsignedRow(out, input, len, temp, xstart, ypos);
      WriteToOutput(out, thread_id, ypos, xstart, len, temp);
    } else if (out.data_type_ == JXL_TYPE_UINT16 ||
               out.data_type_ == JXL_TYPE_FLOAT16) {
      uint16_t* JXL_RESTRICT temp = temp_out_[thread_id].address<uint16_t>();
      if (out.data_type_ == JXL_TYPE_UINT16) {
        StoreUnsignedRow(out, input, len, temp, xstart, ypos);
      } else {
        StoreFloat16Row(out, input, len, temp);
      }
      if (out.swap_endianness_) {
        const HWY_FULL(uint16_t) du;
        size_t output_len = len * out.num_channels_;
        for (size_t j = 0; j < output_len; j += Lanes(du)) {
          auto v = LoadU(du, temp + j);
          auto vswap = Or(ShiftRightSame(v, 8), ShiftLeftSame(v, 8));
          StoreU(vswap, du, temp + j);
        }
      }
      WriteToOutput(out, thread_id, ypos, xstart, len, temp);
    } else if (out.data_type_ == JXL_TYPE_FLOAT) {
      float* JXL_RESTRICT temp = temp_out_[thread_id].address<float>();
      StoreFloatRow(out, input, len, temp);
      if (out.swap_endianness_) {
        size_t output_len = len * out.num_channels_;
        for (size_t j = 0; j < output_len; ++j) {
          temp[j] = BSwapFloat(temp[j]);
        }
      }
      WriteToOutput(out, thread_id, ypos, xstart, len, temp);
    }
  }

  void FlipX(const Output& out, size_t thread_id, size_t len, size_t* xstart,
             const float** line_buffers) const {
    float* temp_in[4];
    for (size_t c = 0; c < out.num_channels_; ++c) {
      size_t tix = thread_id * main_.num_channels_ + c;
      temp_in[c] = temp_in_[tix].address<float>();
      if (temp_in[c] != line_buffers[c]) {
        memcpy(temp_in[c], line_buffers[c], sizeof(float) * len);
      }
    }
    size_t last = (len - 1u);
    size_t num = (len / 2);
    for (size_t i = 0; i < num; ++i) {
      for (size_t c = 0; c < out.num_channels_; ++c) {
        std::swap(temp_in[c][i], temp_in[c][last - i]);
      }
    }
    for (size_t c = 0; c < out.num_channels_; ++c) {
      line_buffers[c] = temp_in[c];
    }
    *xstart = width_ - *xstart - len;
  }

  template <typename T>
  void StoreUnsignedRow(const Output& out, const float* input[4], size_t len,
                        T* output, size_t xstart, size_t ypos) const {
    const HWY_FULL(float) d;
    auto mul = Set(d, (1u << (out.bits_per_sample_)) - 1);
    const Rebind<T, decltype(d)> du;
    const size_t padding = RoundUpTo(len, Lanes(d)) - len;
    for (size_t c = 0; c < out.num_channels_; ++c) {
      msan::UnpoisonMemory(input[c] + len, sizeof(input[c][0]) * padding);
    }
    if (out.num_channels_ == 1) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreU(MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul, 0),
               du, &output[i]);
      }
    } else if (out.num_channels_ == 2) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved2(
            MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul, 0),
            MakeUnsigned<T>(LoadU(d, &input[1][i]), xstart + i, ypos, mul, 1), du,
            &output[2 * i]);
      }
    } else if (out.num_channels_ == 3) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved3(
            MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul, 0),
            MakeUnsigned<T>(LoadU(d, &input[1][i]), xstart + i, ypos, mul, 1),
            MakeUnsigned<T>(LoadU(d, &input[2][i]), xstart + i, ypos, mul, 2), du,
            &output[3 * i]);
      }
    } else if (out.num_channels_ == 4) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved4(
            MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul, 0),
            MakeUnsigned<T>(LoadU(d, &input[1][i]), xstart + i, ypos, mul, 1),
            MakeUnsigned<T>(LoadU(d, &input[2][i]), xstart + i, ypos, mul, 2),
            MakeUnsigned<T>(LoadU(d, &input[3][i]), xstart + i, ypos, mul, 3), du,
            &output[4 * i]);
      }
    }
    msan::PoisonMemory(output + out.num_channels_ * len,
                       sizeof(output[0]) * out.num_channels_ * padding);
  }

  static void StoreFloat16Row(const Output& out, const float* input[4],
                              size_t len, uint16_t* output) {
    const HWY_FULL(float) d;
    const Rebind<uint16_t, decltype(d)> du;
    const Rebind<hwy::float16_t, decltype(d)> df16;
    const size_t padding = RoundUpTo(len, Lanes(d)) - len;
    for (size_t c = 0; c < out.num_channels_; ++c) {
      msan::UnpoisonMemory(input[c] + len, sizeof(input[c][0]) * padding);
    }
    if (out.num_channels_ == 1) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        StoreU(BitCast(du, DemoteTo(df16, v0)), du, &output[i]);
      }
    } else if (out.num_channels_ == 2) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        auto v1 = LoadU(d, &input[1][i]);
        StoreInterleaved2(BitCast(du, DemoteTo(df16, v0)),
                          BitCast(du, DemoteTo(df16, v1)), du, &output[2 * i]);
      }
    } else if (out.num_channels_ == 3) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        auto v1 = LoadU(d, &input[1][i]);
        auto v2 = LoadU(d, &input[2][i]);
        StoreInterleaved3(BitCast(du, DemoteTo(df16, v0)),
                          BitCast(du, DemoteTo(df16, v1)),
                          BitCast(du, DemoteTo(df16, v2)), du, &output[3 * i]);
      }
    } else if (out.num_channels_ == 4) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        auto v1 = LoadU(d, &input[1][i]);
        auto v2 = LoadU(d, &input[2][i]);
        auto v3 = LoadU(d, &input[3][i]);
        StoreInterleaved4(BitCast(du, DemoteTo(df16, v0)),
                          BitCast(du, DemoteTo(df16, v1)),
                          BitCast(du, DemoteTo(df16, v2)),
                          BitCast(du, DemoteTo(df16, v3)), du, &output[4 * i]);
      }
    }
    msan::PoisonMemory(output + out.num_channels_ * len,
                       sizeof(output[0]) * out.num_channels_ * padding);
  }

  static void StoreFloatRow(const Output& out, const float* input[4],
                            size_t len, float* output) {
    const HWY_FULL(float) d;
    if (out.num_channels_ == 1) {
      memcpy(output, input[0], len * sizeof(output[0]));
    } else if (out.num_channels_ == 2) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved2(LoadU(d, &input[0][i]), LoadU(d, &input[1][i]), d,
                          &output[2 * i]);
      }
    } else if (out.num_channels_ == 3) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved3(LoadU(d, &input[0][i]), LoadU(d, &input[1][i]),
                          LoadU(d, &input[2][i]), d, &output[3 * i]);
      }
    } else {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved4(LoadU(d, &input[0][i]), LoadU(d, &input[1][i]),
                          LoadU(d, &input[2][i]), LoadU(d, &input[3][i]), d,
                          &output[4 * i]);
      }
    }
  }

  template <typename T>
  void WriteToOutput(const Output& out, size_t thread_id, size_t ypos,
                     size_t xstart, size_t len, T* output) const {
    if (transpose_) {
      // TODO(szabadka) Buffer 8x8 chunks and transpose with SIMD.
      if (out.run_opaque_) {
        for (size_t i = 0, j = 0; i < len; ++i, j += out.num_channels_) {
          out.pixel_callback_.run(out.run_opaque_, thread_id, ypos, xstart + i,
                                  1, output + j);
        }
      } else {
        const size_t pixel_stride = out.num_channels_ * sizeof(T);
        const size_t offset = xstart * out.stride_ + ypos * pixel_stride;
        for (size_t i = 0, j = 0; i < len; ++i, j += out.num_channels_) {
          const size_t ix = offset + i * out.stride_;
          JXL_DASSERT(ix + pixel_stride <= out.buffer_size_);
          memcpy(reinterpret_cast<uint8_t*>(out.buffer_) + ix, output + j,
                 pixel_stride);
        }
      }
    } else {
      if (out.run_opaque_) {
        out.pixel_callback_.run(out.run_opaque_, thread_id, xstart, ypos, len,
                                output);
      } else {
        const size_t pixel_stride = out.num_channels_ * sizeof(T);
        const size_t offset = ypos * out.stride_ + xstart * pixel_stride;
        JXL_DASSERT(offset + len * pixel_stride <= out.buffer_size_);
        memcpy(reinterpret_cast<uint8_t*>(out.buffer_) + offset, output,
               len * pixel_stride);
      }
    }
  }

  // Process row in chunks to keep per-thread buffers compact.
  size_t width_;
  size_t height_;
  Output main_;  // color + alpha
  size_t num_color_;
  bool want_alpha_;
  bool has_alpha_;
  bool unpremul_alpha_;
  size_t alpha_c_;
  bool flip_x_;
  bool flip_y_;
  bool transpose_;
  std::vector<Output> extra_channels_;
  std::vector<float> opaque_alpha_;
  JxlMemoryManager* memory_manager_;
  std::vector<AlignedMemory> temp_in_;
  std::vector<AlignedMemory> temp_out_;
};

std::unique_ptr<RenderPipelineStage> GetWriteToOutputStage(
    const ImageOutput& main_output, size_t width, size_t height, bool has_alpha,
    bool unpremul_alpha, size_t alpha_c, Orientation undo_orientation,
    std::vector<ImageOutput>& extra_output, JxlMemoryManager* memory_manager) {
  return jxl::make_unique<WriteToOutputStage>(
      main_output, width, height, has_alpha, unpremul_alpha, alpha_c,
      undo_orientation, extra_output, memory_manager);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {

HWY_EXPORT(GetWriteToOutputStage);

namespace {
class WriteToImageBundleStage : public RenderPipelineStage {
 public:
  explicit WriteToImageBundleStage(
      ImageBundle* image_bundle, const OutputEncodingInfo& output_encoding_info)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        image_bundle_(image_bundle),
        color_encoding_(output_encoding_info.color_encoding) {}

  Status SetInputSizes(
      const std::vector<std::pair<size_t, size_t>>& input_sizes) override {
    JxlMemoryManager* memory_manager = image_bundle_->memory_manager();
    JXL_ENSURE(input_sizes.size() >= 3);
    for (size_t c = 1; c < input_sizes.size(); c++) {
      JXL_ENSURE(input_sizes[c].first == input_sizes[0].first);
      JXL_ENSURE(input_sizes[c].second == input_sizes[0].second);
    }
    // TODO(eustas): what should we do in the case of "want only ECs"?
    JXL_ASSIGN_OR_RETURN(Image3F tmp,
                         Image3F::Create(memory_manager, input_sizes[0].first,
                                         input_sizes[0].second));
    JXL_RETURN_IF_ERROR(
        image_bundle_->SetFromImage(std::move(tmp), color_encoding_));
    // TODO(veluca): consider not reallocating ECs if not needed.
    image_bundle_->extra_channels().clear();
    for (size_t c = 3; c < input_sizes.size(); c++) {
      JXL_ASSIGN_OR_RETURN(ImageF ch,
                           ImageF::Create(memory_manager, input_sizes[c].first,
                                          input_sizes[c].second));
      image_bundle_->extra_channels().emplace_back(std::move(ch));
    }
    return true;
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    JXL_ENSURE(xextra_left == 0 && xextra_right == 0);
    for (size_t c = 0; c < 3; c++) {
      memcpy(image_bundle_->color()->PlaneRow(c, ypos) + xpos,
             GetInputRow(input_rows, c, 0), sizeof(float) * xsize);
    }
    for (size_t ec = 0; ec < image_bundle_->extra_channels().size(); ec++) {
      JXL_ENSURE(image_bundle_->extra_channels()[ec].xsize() >= xpos + xsize);
      memcpy(image_bundle_->extra_channels()[ec].Row(ypos) + xpos - xextra_left,
             GetInputRow(input_rows, 3 + ec, 0), sizeof(float) * xsize);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInput;
  }

  const char* GetName() const override { return "WriteIB"; }

 private:
  ImageBundle* image_bundle_;
  ColorEncoding color_encoding_;
};

class WriteToImage3FStage : public RenderPipelineStage {
 public:
  WriteToImage3FStage(JxlMemoryManager* memory_manager, Image3F* image)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        memory_manager_(memory_manager),
        image_(image) {}

  Status SetInputSizes(
      const std::vector<std::pair<size_t, size_t>>& input_sizes) override {
    JXL_ENSURE(input_sizes.size() >= 3);
    for (size_t c = 1; c < 3; ++c) {
      JXL_ENSURE(input_sizes[c].first == input_sizes[0].first);
      JXL_ENSURE(input_sizes[c].second == input_sizes[0].second);
    }
    JXL_ASSIGN_OR_RETURN(*image_,
                         Image3F::Create(memory_manager_, input_sizes[0].first,
                                         input_sizes[0].second));
    return true;
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    JXL_ENSURE(xextra_left == 0 && xextra_right == 0);
    for (size_t c = 0; c < 3; c++) {
      memcpy(image_->PlaneRow(c, ypos) + xpos, GetInputRow(input_rows, c, 0),
             sizeof(float) * xsize);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInput
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "WriteI3F"; }

 private:
  JxlMemoryManager* memory_manager_;
  Image3F* image_;
};

}  // namespace

std::unique_ptr<RenderPipelineStage> GetWriteToImageBundleStage(
    ImageBundle* image_bundle, const OutputEncodingInfo& output_encoding_info) {
  return jxl::make_unique<WriteToImageBundleStage>(image_bundle,
                                                   output_encoding_info);
}

std::unique_ptr<RenderPipelineStage> GetWriteToImage3FStage(
    JxlMemoryManager* memory_manager, Image3F* image) {
  return jxl::make_unique<WriteToImage3FStage>(memory_manager, image);
}

std::unique_ptr<RenderPipelineStage> GetWriteToOutputStage(
    const ImageOutput& main_output, size_t width, size_t height, bool has_alpha,
    bool unpremul_alpha, size_t alpha_c, Orientation undo_orientation,
    std::vector<ImageOutput>& extra_output, JxlMemoryManager* memory_manager) {
  return HWY_DYNAMIC_DISPATCH(GetWriteToOutputStage)(
      main_output, width, height, has_alpha, unpremul_alpha, alpha_c,
      undo_orientation, extra_output, memory_manager);
}

}  // namespace jxl

#endif
