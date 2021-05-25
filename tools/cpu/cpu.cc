// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/cpu/cpu.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "lib/jxl/base/arch_macros.h"  // for JXL_ARCH_*

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
#include "lib/jxl/base/status.h"
#include "tools/cpu/os_specific.h"

using jxl::CeilLog2Nonzero;
using jxl::Debug;

namespace jpegxl {
namespace tools {
namespace cpu {
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

// Detects number of packages/cores/logical processors (HT/SMT).
class X64_Topology {
  enum ApicType {
    kCpuid1_8Bit,   // initial APIC ID
    kCpuidB_32Bit,  // x2APIC ID
    kCpuid1E_32Bit  // AMD extended APIC ID
  };

 public:
  // Enumerates all APIC IDs and partitions them into fields, or returns false
  // if the topology cannot be detected (e.g. due to missing OS support).
  static jxl::Status Detect(ProcessorTopology* topology) {
    const Info info;
    if (DetectLegacyAMD(info, topology)) return true;

    const ApicType type = DetectApicType(info);

    uint32_t core_bits;
    uint32_t logical_bits;
    DetectFieldWidths(info, type, &core_bits, &logical_bits);

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

      const uint32_t id = ProcessorId(type);
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

 private:
  // Returns true if this is an old AMD CPU.
  static jxl::Status DetectLegacyAMD(const Info& info,
                                     ProcessorTopology* topology) {
    if (!info.AMD()) return false;

    // "hyperthreads" not set, we have a single logical (no HT nor multicore)
    Regs r;
    Cpuid(1, 0, &r);
    if ((r.d & (1U << 28)) == 0) {
      topology->logical_per_core = 1;
      topology->cores_per_package = 1;
      topology->packages = 1;
      return true;
    }

    // cpuid:8_1.c bit 2 is "legacy multicore" but it is still set in
    // Threadripper 3, so we do not learn anything from it.

    // Use "extended" method like Intel: variable-width fields in APIC ID.
    return false;
  }

  static ApicType DetectApicType(const Info& info) {
    Regs r;
    Cpuid(1, 0, &r);
    if (info.MaxFunc() >= 0xB && (r.c & (1u << 21))) {
      return kCpuidB_32Bit;
    }

    if (info.AMD() && info.MaxExtFunc() >= 0x8000001E) {
      Cpuid(0x80000001u, 0, &r);
      if (r.c & (1u << 22)) {  // topology extensions
        return kCpuid1E_32Bit;
      }
    }

    return kCpuid1_8Bit;
  }

  // `core_bits`: How many bits in the APIC ID identify the core (per package).
  // #active cores <= (1 << core_bits).
  // `logical_bits`: How many bits identify the logical processor (per core).
  static void DetectFieldWidths_Extended(const Info& info, const ApicType type,
                                         uint32_t* JXL_RESTRICT core_bits,
                                         uint32_t* JXL_RESTRICT logical_bits) {
    *core_bits = 0;
    *logical_bits = 0;

    Regs r;
    Cpuid(1, 0, &r);
    const uint32_t logical_per_package = (r.b >> 16) & 0xFF;

    if (info.Intel() && info.MaxFunc() >= 4) {
      const bool hyperthreading_support = (r.d & (1U << 28)) != 0;

      Cpuid(4, 0, &r);
      *core_bits = static_cast<uint32_t>(CeilLog2Nonzero((r.a >> 26) + 1));

      if (hyperthreading_support) {
        const uint32_t logical_per_core = logical_per_package >> *core_bits;
        if (logical_per_core != 0) {
          *logical_bits =
              static_cast<uint32_t>(CeilLog2Nonzero(logical_per_core));
        }
      }
    }

    if (info.AMD()) {
      if (info.MaxExtFunc() >= 0x80000008u) {
        Cpuid(0x80000008u, 0, &r);
        // AMD 54945 Rev 3.03 documents this as total _threads_ per package;
        // previously, this was listed as the number of _cores_.
        uint32_t thread_bits = (r.c >> 12) & 0xF;
        if (thread_bits == 0) {  // Invalid, ignore
          const uint32_t num_threads = (r.c & 0xFF) + 1;
          thread_bits = static_cast<uint32_t>(CeilLog2Nonzero(num_threads));
        }

        if (type == kCpuid1E_32Bit) {
          Cpuid(0x8000001Eu, 0, &r);
          const uint32_t threads_per_core = ((r.b >> 8) & 0xFF) + 1;
          *logical_bits =
              static_cast<uint32_t>(CeilLog2Nonzero(threads_per_core));
          *core_bits = thread_bits - *logical_bits;
        } else {
          // There does not seem to be another way to detect SMT, so
          // assume it is not available.
          *core_bits = thread_bits;
          *logical_bits = 0;
        }
      } else {
        // Old AMD => did not support SMT/HT yet.
        *core_bits =
            static_cast<uint32_t>(CeilLog2Nonzero(logical_per_package));
        *logical_bits = 0;
      }
    }
  }

  // Returns whether the CPUID:B method was successful.
  static bool DetectFieldWidths_B(const Info& info,
                                  uint32_t* JXL_RESTRICT core_bits,
                                  uint32_t* JXL_RESTRICT logical_bits) {
    if (info.MaxFunc() < 0xB) return false;

    bool got_smt = false;
    bool got_core = false;

    // Prevent spurious uninitialized-variable error
    *core_bits = *logical_bits = 0;

    for (uint32_t level = 0; level < 16; ++level) {
      Regs r;
      Cpuid(0xB, level, &r);

      // We finished all levels if this one has no enabled logical
      // processors.
      if ((r.b & 0xFFFF) == 0) break;

      JXL_ASSERT(level == (r.c & 0xFF));  // Sanity check: should match input

      const uint32_t level_type = (r.c >> 8) & 0xFF;
      const uint32_t level_bits = r.a & 0x1F;

      switch (level_type) {
        case 0:
          Debug("Invalid CPUID level %u despite enabled>0", level);
          break;

        case 1:  // SMT
          *logical_bits = level_bits;
          got_smt = true;
          break;

        case 2:  // core
          *core_bits = level_bits;
          got_core = true;
          break;

        default:
          Debug("Ignoring CPUID:B level %u type %u (%u bits)\n", level,
                level_type, level_bits);
          break;
      }
    }

    if (got_core && got_smt) {
      // Core is actually all logical within a package, so subtract now that
      // we also know logical_bits.
      JXL_ASSERT(*core_bits >= *logical_bits);
      *core_bits -= *logical_bits;
      return true;
    }

    // CPUID:B was incomplete
    return false;
  }

  // Assumes the current processor is representative of all others!
  static void DetectFieldWidths(const Info& info, const ApicType type,
                                uint32_t* JXL_RESTRICT core_bits,
                                uint32_t* JXL_RESTRICT logical_bits) {
    // Preferred on Intel, not available on AMD as of TR3.
    if (type == kCpuidB_32Bit) {
      if (DetectFieldWidths_B(info, core_bits, logical_bits)) {
        return;
      }
    }

    // CPUID:B not available or failed
    DetectFieldWidths_Extended(info, type, core_bits, logical_bits);
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

  // Returns unique identifier of the current logical processor (0 on old CPUs).
  static uint32_t ProcessorId(const ApicType type) {
    Regs r;

    switch (type) {
      case kCpuidB_32Bit:
        Cpuid(11, 0, &r);
        JXL_ASSERT(r.b != 0);
        // Note: whether or not x2APIC is actually supported and enabled, its
        // lower 8 bits match the initial APIC ID (CPUID:1.b).
        return r.d;

      case kCpuid1_8Bit:
        Cpuid(1, 0, &r);
        return r.b >> 24;

      case kCpuid1E_32Bit:
        Cpuid(0x8000001E, 0, &r);
        return r.a;
    }

    // Unreachable
    return 0;
  }
};

#endif  // JXL_ARCH_*

}  // namespace

jxl::Status DetectProcessorTopology(ProcessorTopology* pt) {
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

}  // namespace cpu
}  // namespace tools
}  // namespace jpegxl
