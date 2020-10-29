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

#include "lib/jxl/base/arch_specific.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if JXL_ARCH_X64
#include <xmmintrin.h>
#if !JXL_COMPILER_MSVC
#include <cpuid.h>
#endif
#endif

#if JXL_ARCH_PPC
#include <sys/platform/ppc.h>  // __ppc_get_timebase_freq
#endif

#if JXL_ARCH_ARM
#include <unistd.h>  // sysconf
#endif

#include <string.h>  // memcpy

#include <set>
#include <string>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/os_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace {

#if JXL_ARCH_X64

// For Cpuid.
#pragma pack(push, 1)
struct Regs {
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
};
#pragma pack(pop)

// Calls CPUID instruction with eax=level and ecx=count and fills `r`.
// The caller must ensure `level` <= the max supported.
void Cpuid(const uint32_t level, const uint32_t count, Regs* JXL_RESTRICT r) {
#if JXL_COMPILER_MSVC
  int regs[4];
  __cpuidex(regs, level, count);
  r->a = regs[0];
  r->b = regs[1];
  r->c = regs[2];
  r->d = regs[3];
#else
  // WARNING: avoid using __cpuid_count, which is broken: it lacks volatile and
  // clobber "memory", so the compiler caches CPUID results, not realizing that
  // CPUID:1b (APIC ID) changes across calls to SetThreadAffinity.
  __asm__ __volatile__(
      "xchgq %%rbx,%q1\n"
      "cpuid\n"
      "xchgq %%rbx,%q1"
      : "=a"(r->a), "=r"(r->b), "=c"(r->c), "=d"(r->d)
      : "0"(level), "2"(count)
      : "memory");
#endif
}

class Info {
 public:
  Info() {
    Regs r;
    Cpuid(0, 0, &r);
    max_func_ = r.a;

    char vendor[13];
    // Note unusual order, reverse of ModR/M encoding.
    memcpy(&vendor[0], &r.b, 4);
    memcpy(&vendor[4], &r.d, 4);
    memcpy(&vendor[8], &r.c, 4);
    vendor[12] = '\0';
    intel_ = strcmp(vendor, "GenuineIntel") == 0;
    amd_ = strcmp(vendor, "AuthenticAMD") == 0;

    Cpuid(0x80000000u, 0, &r);
    max_ext_func_ = r.a;
  }

  uint32_t MaxFunc() const { return max_func_; }
  uint32_t MaxExtFunc() const { return max_ext_func_; }
  bool Intel() const { return intel_; }
  bool AMD() const { return amd_; }

  std::string BrandString() const {
    char brand_string[49];
    Regs r;

    // Check if brand string is supported (it is on all reasonable Intel/AMD)
    if (MaxExtFunc() < 0x80000004U) return std::string();

    for (uint32_t i = 0; i < 3; ++i) {
      Cpuid(0x80000002U + i, 0, &r);
      memcpy(brand_string + i * 16, &r, sizeof(r));
    }
    brand_string[48] = 0;
    return brand_string;
  }

 private:
  uint32_t max_func_;
  uint32_t max_ext_func_;
  bool intel_;
  bool amd_;
};

// Detects number of packages/cores/logical processors (hyperthreads).
class X64_Topology {
 public:
  // Enumerates all APIC IDs and partitions them into fields, or returns false
  // if the topology cannot be detected (e.g. due to missing OS support).
  static Status Detect(ProcessorTopology* topology) {
    const Info info;
    const uint32_t core_bits = CoreBits(info);
    const uint32_t logical_bits = LogicalBits(info, core_bits);

    uint32_t total_bits = 0;
    // Order matters:
    Field logical(logical_bits, &total_bits);
    Field core(core_bits, &total_bits);
    Field package(8, &total_bits);

    // Query ProcessorId on each (accessible) logical processor:
    ThreadAffinity* original_affinity = GetThreadAffinity();
    for (int cpu : AvailableCPUs()) {
      if (!PinThreadToCPU(cpu)) {
        free(original_affinity);
        return false;
      }

      const uint32_t id = ProcessorId(info);
      // xAPIC ID of 255 is invalid. Systems with >= 255 logical processors or
      // >= 64 cores require x2APIC (Nehalem) and CPUID:11 detection, which is a
      // separate codepath that we have not yet implemented.
      if (id >= 255) {
        JXL_WARNING("x2APIC ID (%x); TODO: implement CPUID:11", id);
      }

      logical.AddValue(id);
      core.AddValue(id);
      package.AddValue(id);
    }
    JXL_CHECK(SetThreadAffinity(original_affinity));
    free(original_affinity);

    topology->logical_per_core = logical.NumValues();
    topology->cores_per_package = core.NumValues();
    topology->packages = package.NumValues();
    return true;
  }

 public:
  // How many bits in the xAPIC ID identify the core (per package).
  // #active cores <= (1 << bits).
  static uint32_t CoreBits(const Info& info) {
    Regs r;
    if (info.Intel()) {
      if (info.MaxFunc() < 4) return 0;
      Cpuid(4, 0, &r);
      return static_cast<uint32_t>(CeilLog2Nonzero((r.a >> 26) + 1));
    }

    if (info.AMD()) {
      if (info.MaxExtFunc() < 0x80000008u) return 0;
      Cpuid(0x80000008u, 0, &r);
      return static_cast<uint32_t>(CeilLog2Nonzero((r.c & 0xFF) + 1));
    }

    return 0;
  }

  // How many bits in the xAPIC ID identify the logical processor (per core).
  static uint32_t LogicalBits(const Info& info, const uint32_t core_bits) {
    Regs r;
    Cpuid(1, 0, &r);

    // No hyperthreading
    if ((r.d & (1U << 28)) == 0) return 0;

    // Early AMD falsely claim hyperthreading
    if (info.AMD() && (r.c & 2)) return 0;

    const uint32_t logical_per_package = (r.b >> 16) & 0xFF;
    const uint32_t core_and_logical_bits =
        static_cast<uint32_t>(CeilLog2Nonzero(logical_per_package));
    JXL_ASSERT(core_and_logical_bits >= core_bits);
    return core_and_logical_bits - core_bits;
  }

  // Variable-length/position field within an xAPIC ID. Counts the total
  // number of values encountered for all given id.
  class Field {
   public:
    Field(const uint32_t bits, uint32_t* JXL_RESTRICT total_bits)
        : mask_((1U << bits) - 1), shift_(*total_bits) {
      *total_bits += bits;
    }

    void AddValue(const uint32_t id) { values_.insert((id >> shift_) & mask_); }

    size_t NumValues() const { return values_.size(); }

   private:
    const uint32_t mask_;  // zero for zero-width fields
    const uint32_t shift_;
    std::set<uint32_t> values_;
  };

  // Returns initial APIC ID or x2APIC ID, which uniquely identifies the current
  // logical processor. Returns 0 on old CPUs.
  static uint32_t ProcessorId(const Info& info) {
    Regs r;

    // Support 32-bit IDs: we will only use 8 bits, but returning the full ID
    // allows the caller to raise a warning when x2APIC is active.
    if (info.MaxFunc() >= 11) {
      Cpuid(11, 0, &r);
      // Ensure we have CPUID:11 (number of enabled logical processors != 0).
      if (r.b != 0) {
        // r.d is a 32-bit ID; whether or not x2APIC is actually supported and
        // enabled, its lower 8 bits match the initial APIC ID (CPUID:1b).
        return r.d;
      }
    }

    // No CPUID:11 => just return initial APIC ID (8-bit).
    Cpuid(1, 0, &r);
    return r.b >> 24;
  }
};

double X64_DetectNominalClockRate() {
  const Info info;
  const std::string& brand_string = info.BrandString();
  // Brand strings include the maximum configured frequency. These prefixes are
  // defined by Intel CPUID documentation.
  const char* prefixes[3] = {"MHz", "GHz", "THz"};
  const double multipliers[3] = {1E6, 1E9, 1E12};
  for (size_t i = 0; i < 3; ++i) {
    const size_t pos_prefix = brand_string.find(prefixes[i]);
    if (pos_prefix != std::string::npos) {
      const size_t pos_space = brand_string.rfind(' ', pos_prefix - 1);
      if (pos_space != std::string::npos) {
        const std::string digits =
            brand_string.substr(pos_space + 1, pos_prefix - pos_space - 1);
        return std::stod(digits) * multipliers[i];
      }
    }
  }

  return 0.0;
}

#elif JXL_ARCH_PPC

double PPC_DetectNominalClockRate() {
  double freq = -1;
  char line[200];
  char* s;
  char* value;

  FILE* f = fopen("/proc/cpuinfo", "r");
  if (f != nullptr) {
    while (fgets(line, sizeof(line), f) != nullptr) {
      // NOTE: the ':' is the only character we can rely on
      if (!(value = strchr(line, ':'))) continue;
      // terminate the valuename
      *value++ = '\0';
      // skip any leading spaces
      while (*value == ' ') value++;
      if ((s = strchr(value, '\n'))) *s = '\0';

      if (!strncasecmp(line, "clock", strlen("clock")) &&
          sscanf(value, "%lf", &freq) == 1) {
        freq *= 1E6;
        break;
      }
    }
    fclose(f);
    return freq;
  }

  return 0.0;
}

#endif  // JXL_ARCH_*

}  // namespace

Status DetectProcessorTopology(ProcessorTopology* pt) {
  if (GetProcessorTopologyFromOS(pt)) return true;
#if JXL_ARCH_X64
  if (X64_Topology::Detect(pt)) return true;
#elif JXL_ARCH_ARM
  // TODO(deymo): Actually look up the CPU topology and model the big/small
  // core split.
  pt->logical_per_core = 1;
  pt->cores_per_package = sysconf(_SC_NPROCESSORS_ONLN);
  pt->packages = 1;
  return true;
#endif
  return JXL_FAILURE("Unable to detect processor topology");
}

double NominalClockRate() {
// Thread-safe caching - this is called several times.
#if JXL_ARCH_X64
  static const double cycles_per_second = X64_DetectNominalClockRate();
  return cycles_per_second;
#elif JXL_ARCH_PPC
  static const double cycles_per_second = PPC_DetectNominalClockRate();
  return cycles_per_second;
#else
  return 0.0;
#endif
}

double InvariantTicksPerSecond() {
#if JXL_ARCH_PPC
  static const double cycles_per_second = __ppc_get_timebase_freq();
  return cycles_per_second;
#elif JXL_ARCH_X64
  return NominalClockRate();
#else
  return 1E9;  // nanoseconds - matches tsc_timer.h CLOCK_MONOTONIC fallback.
#endif
}

}  // namespace jxl
