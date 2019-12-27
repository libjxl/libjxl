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

#ifndef JXL_BASE_PROFILER_H_
#define JXL_BASE_PROFILER_H_

// High precision, low overhead time measurements. Returns exact call counts and
// total elapsed time for user-defined 'zones' (code regions, i.e. C++ scopes).
//
// Usage: add this header to BUILD srcs; instrument regions of interest:
// { PROFILER_ZONE("name"); /*code*/ } or
// void FuncToMeasure() { PROFILER_FUNC; /*code*/ }.
// After all threads have exited any zones, invoke PROFILER_PRINT_RESULTS() to
// print call counts and average durations [CPU cycles] to stdout, sorted in
// descending order of total duration.

// Configuration settings:

// If zero, this file has no effect and no measurements will be recorded.
#ifndef PROFILER_ENABLED
#define PROFILER_ENABLED 0
#endif

// How many mebibytes to allocate (if PROFILER_ENABLED) per thread that
// enters at least one zone. Once this buffer is full, the thread will analyze
// and discard packets, thus temporarily adding some observer overhead.
// Each zone occupies 16 bytes.
#ifndef PROFILER_THREAD_STORAGE
#define PROFILER_THREAD_STORAGE 32ULL
#endif

#if PROFILER_ENABLED

#define PROFILER_PRINT_OVERHEAD 0

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // memcpy

#include <algorithm>  // min/max
#include <atomic>
#include <cinttypes>  // PRIu64
#include <hwy/static_targets.h>
#include <new>

#include "jxl/base/arch_specific.h"
#include "jxl/base/cache_aligned.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/robust_statistics.h"
#include "jxl/base/status.h"
#include "jxl/base/tsc_timer.h"

// Non-portable aspects:
// - SSE2 128-bit load/store (write-combining, UpdateOrAdd)
// - RDTSCP timestamps (serializing, high-resolution)
// - assumes string literals are stored within an 8 MiB range
// - compiler-specific annotations (restrict, alignment, fences)
#if JXL_ARCH_X64
#include <emmintrin.h>
#if JXL_COMPILER_MSVC
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

namespace jxl {

// Upper bounds for various fixed-size data structures (guarded via JXL_ASSERT):

// How many unique threads can enter a zone (those that don't do not count).
// Memory use is about kMaxThreads * PROFILER_THREAD_STORAGE MiB.
// WARNING: fiber libraries and multiple ThreadPool can spawn >100 threads.
static constexpr size_t kMaxThreads = 256;

// Maximum nesting of zones.
static constexpr size_t kMaxDepth = 64;

// Total number of zones.
static constexpr size_t kMaxZones = 256;

// Represents zone entry/exit events. Stores a full-resolution timestamp plus
// an offset (representing zone name or identifying exit packets). POD.
class Packet {
 public:
  // If offsets do not fit, UpdateOrAdd will overrun our heap allocation
  // (governed by kMaxZones). We have seen ~100 MiB static binaries.
  static constexpr size_t kOffsetBits = 27;
  static constexpr uintptr_t kOffsetBias = 1ULL << (kOffsetBits - 1);

  // We need full-resolution timestamps; at an effective rate of 4 GHz,
  // this permits 34 second zone durations (for longer durations, split into
  // multiple zones). Wraparound is handled by masking.
  static constexpr size_t kTimestampBits = 64 - kOffsetBits;
  static constexpr uintptr_t kTimestampMask = (1ULL << kTimestampBits) - 1;

  static Packet Make(const uintptr_t biased_offset, const uint64_t timestamp) {
    JXL_ASSERT(biased_offset < (1ULL << kOffsetBits));

    Packet packet;
    packet.bits_ =
        (biased_offset << kTimestampBits) + (timestamp & kTimestampMask);
    return packet;
  }

  uint64_t Timestamp() const { return bits_ & kTimestampMask; }

  uintptr_t BiasedOffset() const { return (bits_ >> kTimestampBits); }

 private:
  uint64_t bits_;
};
static_assert(sizeof(Packet) == 8, "Wrong Packet size");

// Returns the address of a string literal. Assuming zone names are also
// literals and stored nearby, we can represent them as offsets, which are
// faster to compute than hashes or even a static index.
//
// This function must not be static - each call (even from other translation
// units) must return the same value.
inline uintptr_t StringOrigin() {
  // Chosen such that no zone name is a prefix nor suffix of this string
  // to ensure they aren't merged (offset 0 identifies zone-exit packets).
  static const char* string_origin = "__#Origin#__";

  return reinterpret_cast<uintptr_t>(string_origin) - Packet::kOffsetBias;
}

// Representation of an active zone, stored in a stack. Used to deduct
// child duration from the parent's self time. POD.
struct ProfilerNode {
  Packet packet;
  uint64_t child_total;
};

// Holds statistics for all zones with the same name. POD.
struct Accumulator {
  static constexpr size_t kNumCallBits = 64 - Packet::kOffsetBits;

  uintptr_t BiasedOffset() const { return num_calls >> kNumCallBits; }
  uint64_t NumCalls() const { return num_calls & ((1ULL << kNumCallBits) - 1); }

  // UpdateOrAdd relies upon this layout.
  uint64_t num_calls = 0;  // upper bits = biased_offset.
  uint64_t total_duration = 0;
};
#if JXL_ARCH_X64
static_assert(sizeof(Accumulator) == 2 * sizeof(uint64_t), "Accumulator size");
#endif

template <typename T>
inline T ClampedSubtract(const T minuend, const T subtrahend) {
  if (subtrahend > minuend) {
    return 0;
  }
  return minuend - subtrahend;
}

// Per-thread call graph (stack) and Accumulator for each zone.
class Results {
 public:
  Results() {
    // Zero-initialize first accumulator to avoid a check for num_zones_ == 0.
    memset(zones_, 0, sizeof(Accumulator));
  }

  // Used for computing overhead when this thread encounters its first Zone.
  // This has no observable effect apart from increasing "analyze_elapsed_".
  uint64_t ZoneDuration(const Packet* packets) {
    JXL_CHECK(depth_ == 0);
    JXL_CHECK(num_zones_ == 0);
    AnalyzePackets(packets, 2);
    const uint64_t duration = zones_[0].total_duration;
    zones_[0].num_calls = 0;
    zones_[0].total_duration = 0;
    JXL_CHECK(depth_ == 0);
    num_zones_ = 0;
    return duration;
  }

  void SetSelfOverhead(const uint64_t self_overhead) {
    self_overhead_ = self_overhead;
  }

  void SetChildOverhead(const uint64_t child_overhead) {
    child_overhead_ = child_overhead;
  }

  // Draw all required information from the packets, which can be discarded
  // afterwards. Called whenever this thread's storage is full.
  void AnalyzePackets(const Packet* packets, const size_t num_packets) {
    const uint64_t t0 = TicksBefore();

    for (size_t i = 0; i < num_packets; ++i) {
      const Packet p = packets[i];
      // Entering a zone
      if (p.BiasedOffset() != Packet::kOffsetBias) {
        JXL_ASSERT(depth_ < kMaxDepth);
        nodes_[depth_].packet = p;
        nodes_[depth_].child_total = 0;
        ++depth_;
        continue;
      }

      JXL_ASSERT(depth_ != 0);
      const ProfilerNode& node = nodes_[depth_ - 1];
      // Masking correctly handles unsigned wraparound.
      const uint64_t duration =
          (p.Timestamp() - node.packet.Timestamp()) & Packet::kTimestampMask;
      const uint64_t self_duration = ClampedSubtract(
          duration, self_overhead_ + child_overhead_ + node.child_total);

      UpdateOrAdd(node.packet.BiasedOffset(), 1, self_duration);
      --depth_;

      // Deduct this nested node's time from its parent's self_duration.
      if (depth_ != 0) {
        nodes_[depth_ - 1].child_total += duration + child_overhead_;
      }
    }

    const uint64_t t1 = TicksAfter();
    analyze_elapsed_ += t1 - t0;
  }

  // Incorporates results from another thread. Call after all threads have
  // exited any zones.
  void Assimilate(const Results& other) {
    const uint64_t t0 = TicksBefore();
    JXL_ASSERT(depth_ == 0);
    JXL_ASSERT(other.depth_ == 0);

    for (size_t i = 0; i < other.num_zones_; ++i) {
      const Accumulator& zone = other.zones_[i];
      UpdateOrAdd(zone.BiasedOffset(), zone.NumCalls(), zone.total_duration);
    }
    const uint64_t t1 = TicksAfter();
    analyze_elapsed_ += t1 - t0 + other.analyze_elapsed_;
  }

  // Single-threaded.
  void Print() {
    const uint64_t t0 = TicksBefore();
    MergeDuplicates();

    // Sort by decreasing total (self) cost.
    std::sort(zones_, zones_ + num_zones_,
              [](const Accumulator& r1, const Accumulator& r2) {
                return r1.total_duration > r2.total_duration;
              });

    const uintptr_t string_origin = StringOrigin();
    uint64_t total_visible_duration = 0;
    for (size_t i = 0; i < num_zones_; ++i) {
      const Accumulator& r = zones_[i];
      const uint64_t num_calls = r.NumCalls();
      const char* name =
          reinterpret_cast<const char*>(string_origin + r.BiasedOffset());
      if (name[0] != '@') {
        total_visible_duration += r.total_duration;
        printf("%-40s: %10" PRIu64 " x %15" PRIu64 "= %15" PRIu64 "\n", name,
               num_calls, r.total_duration / num_calls, r.total_duration);
      }
    }

    const uint64_t t1 = TicksAfter();
    analyze_elapsed_ += t1 - t0;
    printf("Total clocks during analysis: %" PRIu64 "\n", analyze_elapsed_);
    printf("Total clocks measured: %" PRIu64 "\n", total_visible_duration);
  }

  // Single-threaded. Clears all results as if no zones had been recorded.
  void Reset() {
    analyze_elapsed_ = 0;
    JXL_CHECK(depth_ == 0);
    num_zones_ = 0;
    memset(nodes_, 0, sizeof(nodes_));
    memset(zones_, 0, sizeof(zones_));
  }

 private:
#if JXL_ARCH_X64
  static bool SameOffset(const __m128i zone, const uintptr_t biased_offset) {
    const uint64_t num_calls = _mm_cvtsi128_si64(zone);
    return (num_calls >> Accumulator::kNumCallBits) == biased_offset;
  }
#endif

  // Updates an existing Accumulator (uniquely identified by biased_offset) or
  // adds one if this is the first time this thread analyzed that zone.
  // Uses a self-organizing list data structure, which avoids dynamic memory
  // allocations and is far faster than unordered_map. Loads, updates and
  // stores the entire Accumulator with vector instructions.
  void UpdateOrAdd(const uintptr_t biased_offset, const uint64_t num_calls,
                   const uint64_t duration) {
    JXL_ASSERT(biased_offset < (1ULL << Packet::kOffsetBits));

#if JXL_ARCH_X64
    const __m128i num_calls_64 = _mm_cvtsi64_si128(num_calls);
    const __m128i duration_64 = _mm_cvtsi64_si128(duration);
    const __m128i add_duration_call =
        _mm_unpacklo_epi64(num_calls_64, duration_64);

    __m128i* const JXL_RESTRICT zones = reinterpret_cast<__m128i*>(zones_);

    // Special case for first zone: (maybe) update, without swapping.
    __m128i prev = _mm_load_si128(zones);
    if (SameOffset(prev, biased_offset)) {
      prev = _mm_add_epi64(prev, add_duration_call);
      JXL_ASSERT(SameOffset(prev, biased_offset));
      _mm_store_si128(zones, prev);
      return;
    }

    // Look for a zone with the same offset.
    for (size_t i = 1; i < num_zones_; ++i) {
      __m128i zone = _mm_load_si128(zones + i);
      if (SameOffset(zone, biased_offset)) {
        zone = _mm_add_epi64(zone, add_duration_call);
        JXL_ASSERT(SameOffset(zone, biased_offset));
        // Swap with predecessor (more conservative than move to front,
        // but at least as successful).
        _mm_store_si128(zones + i - 1, zone);
        _mm_store_si128(zones + i, prev);
        return;
      }
      prev = zone;
    }

    // Not found; create a new Accumulator.
    const __m128i biased_offset_64 = _mm_slli_epi64(
        _mm_cvtsi64_si128(biased_offset), Accumulator::kNumCallBits);
    const __m128i zone = _mm_add_epi64(biased_offset_64, add_duration_call);
    JXL_ASSERT(SameOffset(zone, biased_offset));

    JXL_ASSERT(num_zones_ < kMaxZones);
    _mm_store_si128(zones + num_zones_, zone);
    ++num_zones_;
#else
    // Special case for first zone: (maybe) update, without swapping.
    if (zones_[0].BiasedOffset() == biased_offset) {
      zones_[0].total_duration += duration;
      zones_[0].num_calls += num_calls;
      JXL_ASSERT(zones_[0].BiasedOffset() == biased_offset);
      return;
    }

    // Look for a zone with the same offset.
    for (size_t i = 1; i < num_zones_; ++i) {
      if (zones_[i].BiasedOffset() == biased_offset) {
        zones_[i].total_duration += duration;
        zones_[i].num_calls += num_calls;
        JXL_ASSERT(zones_[i].BiasedOffset() == biased_offset);
        // Swap with predecessor (more conservative than move to front,
        // but at least as successful).
        const Accumulator prev = zones_[i - 1];
        zones_[i - 1] = zones_[i];
        zones_[i] = prev;
        return;
      }
    }

    // Not found; create a new Accumulator.
    JXL_ASSERT(num_zones_ < kMaxZones);
    Accumulator* JXL_RESTRICT zone = zones_ + num_zones_;
    zone->num_calls = (biased_offset << Accumulator::kNumCallBits) + num_calls;
    zone->total_duration = duration;
    JXL_ASSERT(zone->BiasedOffset() == biased_offset);
    ++num_zones_;
#endif
  }

  // Each instantiation of a function template seems to get its own copy of
  // __func__ and GCC doesn't merge them. An N^2 search for duplicates is
  // acceptable because we only expect a few dozen zones.
  void MergeDuplicates() {
    const uintptr_t string_origin = StringOrigin();
    for (size_t i = 0; i < num_zones_; ++i) {
      const uintptr_t biased_offset = zones_[i].BiasedOffset();
      const char* name =
          reinterpret_cast<const char*>(string_origin + biased_offset);
      // Separate num_calls from biased_offset so we can add them together.
      uint64_t num_calls = zones_[i].NumCalls();

      // Add any subsequent duplicates to num_calls and total_duration.
      for (size_t j = i + 1; j < num_zones_;) {
        if (!strcmp(name, reinterpret_cast<const char*>(
                              string_origin + zones_[j].BiasedOffset()))) {
          num_calls += zones_[j].NumCalls();
          zones_[i].total_duration += zones_[j].total_duration;
          // Fill hole with last item.
          zones_[j] = zones_[--num_zones_];
        } else {  // Name differed, try next Accumulator.
          ++j;
        }
      }

      JXL_ASSERT(num_calls < (1ULL << Accumulator::kNumCallBits));

      // Re-pack regardless of whether any duplicates were found.
      zones_[i].num_calls =
          (biased_offset << Accumulator::kNumCallBits) + num_calls;
    }
  }

  uint64_t analyze_elapsed_ = 0;
  uint64_t self_overhead_ = 0;
  uint64_t child_overhead_ = 0;

  size_t depth_ = 0;      // Number of active zones.
  size_t num_zones_ = 0;  // Number of retired zones.

  alignas(64) ProfilerNode nodes_[kMaxDepth];  // Stack
  alignas(64) Accumulator zones_[kMaxZones];   // Self-organizing list
};

// Per-thread packet storage, allocated via CacheAligned.
class ThreadSpecific {
  static constexpr size_t kBufferCapacity =
      CacheAligned::kCacheLineSize / sizeof(Packet);

 public:
  // "name" is used to sanity-check offsets fit in kOffsetBits.
  explicit ThreadSpecific(const char* name)
      : packets_(static_cast<Packet*>(
            CacheAligned::Allocate(PROFILER_THREAD_STORAGE << 20))),
        num_packets_(0),
        max_packets_(PROFILER_THREAD_STORAGE << 17),
        string_origin_(StringOrigin()) {
    // Even in optimized builds (with NDEBUG), verify that this zone's name
    // offset fits within the allotted space. If not, UpdateOrAdd is likely to
    // overrun zones_[]. We also JXL_ASSERT(), but users often do not run debug
    // builds. Checking here on the cold path (only reached once per thread)
    // is cheap, but it only covers one zone.
    const uintptr_t biased_offset =
        reinterpret_cast<uintptr_t>(name) - string_origin_;
    JXL_CHECK(biased_offset <= (1ULL << Packet::kOffsetBits));
  }

  ~ThreadSpecific() { CacheAligned::Free(packets_); }

  // Depends on Zone => defined below.
  void ComputeOverhead();

  void WriteEntry(const char* name, const uint64_t timestamp) {
    const uintptr_t biased_offset =
        reinterpret_cast<uintptr_t>(name) - string_origin_;
    Write(Packet::Make(biased_offset, timestamp));
  }

  void WriteExit(const uint64_t timestamp) {
    const uintptr_t biased_offset = Packet::kOffsetBias;
    Write(Packet::Make(biased_offset, timestamp));
  }

  void AnalyzeRemainingPackets() {
#if JXL_ARCH_X64
    // Ensures prior weakly-ordered streaming stores are globally visible.
    _mm_sfence();

    // Storage full => empty it.
    if (num_packets_ + buffer_size_ > max_packets_) {
      results_.AnalyzePackets(packets_, num_packets_);
      num_packets_ = 0;
    }
    memcpy(packets_ + num_packets_, buffer_, buffer_size_ * sizeof(Packet));
    num_packets_ += buffer_size_;
    buffer_size_ = 0;
#endif

    results_.AnalyzePackets(packets_, num_packets_);
    num_packets_ = 0;
  }

  Results& GetResults() { return results_; }

 private:
#if JXL_ARCH_X64 && HWY_BITS != 0
  // Overwrites `to` without loading it into cache (read-for-ownership).
  // Copies kCacheLineSize bytes from/to naturally aligned addresses.
  template <typename T>
  static HWY_ATTR void StreamCacheLine(const T* JXL_RESTRICT from,
                                       T* JXL_RESTRICT to) {
    static_assert(16 % sizeof(T) == 0, "T must fit in a lane");
    constexpr size_t kLanes = 16 / sizeof(T);
    const hwy::Desc<T, kLanes> d;
    JXL_COMPILER_FENCE;
    const auto v0 = Load(d, from + 0 * kLanes);
    const auto v1 = Load(d, from + 1 * kLanes);
    const auto v2 = Load(d, from + 2 * kLanes);
    const auto v3 = Load(d, from + 3 * kLanes);
    static_assert(sizeof(v0) * 4 == 64, "Wrong #vectors");
    // Fences prevent the compiler from reordering loads/stores, which may
    // interfere with write-combining.
    JXL_COMPILER_FENCE;
    Stream(v0, d, to + 0 * kLanes);
    Stream(v1, d, to + 1 * kLanes);
    Stream(v2, d, to + 2 * kLanes);
    Stream(v3, d, to + 3 * kLanes);
    JXL_COMPILER_FENCE;
  }
#endif

  // Write packet to buffer/storage, emptying them as needed.
  void Write(const Packet packet) {
#if JXL_ARCH_X64 && HWY_BITS != 0
    // Buffer full => copy to storage.
    if (buffer_size_ == kBufferCapacity) {
      // Storage full => empty it.
      if (num_packets_ + kBufferCapacity > max_packets_) {
        results_.AnalyzePackets(packets_, num_packets_);
        num_packets_ = 0;
      }
      // This buffering halves observer overhead and decreases the overall
      // runtime by about 3%.
      StreamCacheLine(buffer_, packets_ + num_packets_);
      num_packets_ += kBufferCapacity;
      buffer_size_ = 0;
    }
    buffer_[buffer_size_] = packet;
    ++buffer_size_;
#else
    // Write directly to storage.
    if (num_packets_ >= max_packets_) {
      results_.AnalyzePackets(packets_, num_packets_);
      num_packets_ = 0;
    }
    packets_[num_packets_] = packet;
    ++num_packets_;
#endif
  }

  // Write-combining buffer to avoid cache pollution. Must be the first
  // non-static member to ensure cache-line alignment.
#if JXL_ARCH_X64
  Packet buffer_[kBufferCapacity];
  size_t buffer_size_ = 0;
#endif

  // Contiguous storage for zone enter/exit packets.
  Packet* const JXL_RESTRICT packets_;
  size_t num_packets_;
  const size_t max_packets_;
  // Cached here because we already read this cache line on zone entry/exit.
  uintptr_t string_origin_;
  Results results_;
};

class ThreadList {
 public:
  // Thread-safe.
  void Add(ThreadSpecific* const ts) {
    const uint32_t index = num_threads_.fetch_add(1, std::memory_order_relaxed);
    JXL_CHECK(index < kMaxThreads);
    threads_[index] = ts;
  }

  // Single-threaded.
  void PrintResults() {
    const uint32_t num_threads = num_threads_.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < num_threads; ++i) {
      threads_[i]->AnalyzeRemainingPackets();
    }

    // Combine all threads into a single Result.
    for (uint32_t i = 1; i < num_threads; ++i) {
      threads_[0]->GetResults().Assimilate(threads_[i]->GetResults());
    }

    if (num_threads != 0) {
      threads_[0]->GetResults().Print();

      for (uint32_t i = 0; i < num_threads; ++i) {
        threads_[i]->GetResults().Reset();
      }
    }
  }

 private:
  // Owning pointers.
  alignas(64) ThreadSpecific* threads_[kMaxThreads];
  std::atomic<uint32_t> num_threads_{0};
};

// RAII zone enter/exit recorder constructed by the ZONE macro; also
// responsible for initializing ThreadSpecific.
class Zone {
 public:
  // "name" must be a string literal (see StringOrigin).
  JXL_NOINLINE explicit Zone(const char* name) {
    JXL_COMPILER_FENCE;
    ThreadSpecific* JXL_RESTRICT thread_specific = StaticThreadSpecific();
    if (JXL_UNLIKELY(thread_specific == nullptr)) {
      void* mem = CacheAligned::Allocate(sizeof(ThreadSpecific));
      thread_specific = new (mem) ThreadSpecific(name);
      // Must happen before ComputeOverhead, which re-enters this ctor.
      Threads().Add(thread_specific);
      StaticThreadSpecific() = thread_specific;
      thread_specific->ComputeOverhead();
    }

    // (Capture timestamp ASAP, not inside WriteEntry.)
    JXL_COMPILER_FENCE;
    const uint64_t timestamp = TicksBefore();
    thread_specific->WriteEntry(name, timestamp);
  }

  JXL_NOINLINE ~Zone() {
    JXL_COMPILER_FENCE;
    const uint64_t timestamp = TicksAfter();
    StaticThreadSpecific()->WriteExit(timestamp);
    JXL_COMPILER_FENCE;
  }

  // Call exactly once after all threads have exited all zones.
  static void PrintResults() { Threads().PrintResults(); }

 private:
  // Returns reference to the thread's ThreadSpecific pointer (initially null).
  // Function-local static avoids needing a separate definition.
  static ThreadSpecific*& StaticThreadSpecific() {
    static thread_local ThreadSpecific* thread_specific;
    return thread_specific;
  }

  // Returns the singleton ThreadList. Non time-critical.
  static ThreadList& Threads() {
    static ThreadList threads_;
    return threads_;
  }
};

// Creates a zone starting from here until the end of the current scope.
// Timestamps will be recorded when entering and exiting the zone.
// "name" must be a string literal, which is ensured by merging with "".
#define PROFILER_ZONE(name)        \
  JXL_COMPILER_FENCE;              \
  const ::jxl::Zone zone("" name); \
  JXL_COMPILER_FENCE

// Creates a zone for an entire function (when placed at its beginning).
// Shorter/more convenient than ZONE.
#define PROFILER_FUNC               \
  JXL_COMPILER_FENCE;               \
  const ::jxl::Zone zone(__func__); \
  JXL_COMPILER_FENCE

#define PROFILER_PRINT_RESULTS ::jxl::Zone::PrintResults

inline void ThreadSpecific::ComputeOverhead() {
  // Delay after capturing timestamps before/after the actual zone runs. Even
  // with frequency throttling disabled, this has a multimodal distribution,
  // including 32, 34, 48, 52, 59, 62.
  uint64_t self_overhead;
  {
    const size_t kNumSamples = 32;
    uint32_t samples[kNumSamples];
    for (size_t idx_sample = 0; idx_sample < kNumSamples; ++idx_sample) {
      const size_t kNumDurations = 1024;
      uint32_t durations[kNumDurations];

      for (size_t idx_duration = 0; idx_duration < kNumDurations;
           ++idx_duration) {
        {
          PROFILER_ZONE("Dummy Zone (never shown)");
        }
#if JXL_ARCH_X64
        const uint64_t duration = results_.ZoneDuration(buffer_);
        buffer_size_ = 0;
#else
        const uint64_t duration = results_.ZoneDuration(packets_);
        num_packets_ = 0;
#endif
        durations[idx_duration] = static_cast<uint32_t>(duration);
        JXL_CHECK(num_packets_ == 0);
      }
      CountingSort(durations, durations + kNumDurations);
      samples[idx_sample] = HalfSampleMode()(durations, kNumDurations);
    }
    // Median.
    CountingSort(samples, samples + kNumSamples);
    self_overhead = samples[kNumSamples / 2];
#if PROFILER_PRINT_OVERHEAD
    printf("Overhead: %zu\n", self_overhead);
#endif
    results_.SetSelfOverhead(self_overhead);
  }

  // Delay before capturing start timestamp / after end timestamp.
  const size_t kNumSamples = 32;
  uint32_t samples[kNumSamples];
  for (size_t idx_sample = 0; idx_sample < kNumSamples; ++idx_sample) {
    const size_t kNumDurations = 16;
    uint32_t durations[kNumDurations];
    for (size_t idx_duration = 0; idx_duration < kNumDurations;
         ++idx_duration) {
      const size_t kReps = 10000;
      // Analysis time should not be included => must fit within buffer.
      JXL_CHECK(kReps * 2 < max_packets_);
#if JXL_ARCH_X64
      _mm_mfence();
#endif
      const uint64_t t0 = TicksBefore();
      for (size_t i = 0; i < kReps; ++i) {
        PROFILER_ZONE("Dummy");
      }
#if JXL_ARCH_X64
      _mm_sfence();
#endif
      const uint64_t t1 = TicksAfter();
#if JXL_ARCH_X64
      JXL_CHECK(num_packets_ + buffer_size_ == kReps * 2);
      buffer_size_ = 0;
#else
      JXL_CHECK(num_packets_ == kReps * 2);
#endif
      num_packets_ = 0;
      const uint64_t avg_duration = (t1 - t0 + kReps / 2) / kReps;
      durations[idx_duration] =
          static_cast<uint32_t>(ClampedSubtract(avg_duration, self_overhead));
    }
    CountingSort(durations, durations + kNumDurations);
    samples[idx_sample] = HalfSampleMode()(durations, kNumDurations);
  }
  CountingSort(samples, samples + kNumSamples);
  const uint64_t child_overhead = samples[9 * kNumSamples / 10];
#if PROFILER_PRINT_OVERHEAD
  printf("Child overhead: %zu\n", child_overhead);
#endif
  results_.SetChildOverhead(child_overhead);
}

}  // namespace jxl

#else  // !PROFILER_ENABLED
#define PROFILER_ZONE(name)
#define PROFILER_FUNC
#define PROFILER_PRINT_RESULTS()
#endif

#endif  // JXL_BASE_PROFILER_H_
