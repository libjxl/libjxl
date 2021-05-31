// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/cpu/os_specific.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <ctime>
#include <numeric>
#include <random>

#include "lib/jxl/base/os_macros.h"  // for JXL_OS_*
#include "tools/cpu/cpu.h"           // ProcessorTopology

#if JXL_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <windows.h>
#endif  // JXL_OS_WIN

#if JXL_OS_LINUX
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif  // _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif  // JXL_OS_LINUX

#if JXL_OS_MAC
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif  // JXL_OS_MAC

#if JXL_OS_FREEBSD
#include <sys/cpuset.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif  // JXL_OS_FREEBSD

#if JXL_OS_HAIKU
#include <OS.h>
#endif  // JXL_OS_HAIKU

using jxl::Status;

namespace jpegxl {
namespace tools {
namespace cpu {

// Emulate Linux type (cpu_set_t) + interface on other platforms

#if JXL_OS_FREEBSD
using cpu_set_t = cpuset_t;
#elif JXL_OS_WIN || JXL_OS_MAC || JXL_OS_HAIKU
using cpu_set_t = uint64_t;

static inline void CPU_ZERO(cpu_set_t* set) { *set = 0; }

static inline int CPU_ISSET(int cpu, const cpu_set_t* set) {
  return (*set & (1ULL << cpu)) != 0;
}

static inline void CPU_SET(int cpu, cpu_set_t* set) { *set |= (1ULL << cpu); }
#endif

struct ThreadAffinity {
  cpu_set_t set;
};

#if JXL_OS_MAC
namespace {
Status GetSystemValue(const char* name, size_t* value) {
  int64_t value_i64 = 0;
  size_t size = sizeof(value_i64);
  const int err = sysctlbyname(name, &value_i64, &size, nullptr, 0);
  if (err != 0) return JXL_FAILURE("sysctl packages failed");
  JXL_ASSERT(value_i64 >= 0);
  *value = static_cast<size_t>(value_i64);
  return true;
}

// Returns mask with the lowest N bits set, one per logical processor.
cpu_set_t SetOfAllLogicalProcessors() {
  size_t logical;
  // On failure, assume there is at least one logical processor.
  if (!GetSystemValue("machdep.cpu.thread_count", &logical)) return 1;

  if (logical > 64) {
    printf("Warning: more than 64 logical processors, update cpu_set_t");
    return ~0ull;
  }
  if (logical == 64) return ~0ull;

  return (1ull << logical) - 1;
}
}  // namespace

#elif JXL_OS_HAIKU

namespace {
cpu_set_t SetOfAllLogicalProcessors() {
  system_info info;
  get_system_info(&info);

  if (info.cpu_count > 64) {
    printf("Warning: more than 64 logical processors, update cpu_set_t");
    return ~0ull;
  }
  if (info.cpu_count == 64) return ~0ull;

  return (1ull << info.cpu_count) - 1;
}
}  // namespace

#endif

Status GetProcessorTopologyFromOS(ProcessorTopology* pt) {
#if JXL_OS_MAC
  size_t packages, cores, logical;  // totals, not per package/core!
  JXL_RETURN_IF_ERROR(GetSystemValue("hw.packages", &packages));
  JXL_RETURN_IF_ERROR(GetSystemValue("machdep.cpu.core_count", &cores));
  JXL_RETURN_IF_ERROR(GetSystemValue("machdep.cpu.thread_count", &logical));

  // All succeeded: now set `pt`
  pt->packages = packages;
  pt->cores_per_package = cores / packages;
  pt->logical_per_core = logical / cores;

  return true;
#elif JXL_OS_HAIKU
  system_info info;
  get_system_info(&info);
  pt->packages = 1;
  pt->cores_per_package = info.cpu_count;
  pt->logical_per_core = 1;

  return true;

#else
  // Not needed on X64 if the affinity APIs work (DetectProcessorTopology will
  // succeed)
  return false;
#endif
}

ThreadAffinity* GetThreadAffinity() {
  ThreadAffinity* affinity =
      static_cast<ThreadAffinity*>(malloc(sizeof(ThreadAffinity)));
#if JXL_OS_WIN
  DWORD_PTR process_affinity, system_affinity;
  const BOOL ok = GetProcessAffinityMask(GetCurrentProcess(), &process_affinity,
                                         &system_affinity);
  JXL_CHECK(ok);
  affinity->set = process_affinity;
#elif JXL_OS_LINUX
  CPU_ZERO(&affinity->set);
  const int err = sched_getaffinity(0, sizeof(cpu_set_t), &affinity->set);
  JXL_CHECK(err == 0);
#elif JXL_OS_FREEBSD
  const pid_t pid = getpid();  // current thread
  const int err = cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, pid,
                                     sizeof(cpuset_t), &affinity->set);
  JXL_CHECK(err == 0);
#elif JXL_OS_MAC || JXL_OS_HAIKU
  static cpu_set_t all = SetOfAllLogicalProcessors();
  affinity->set = all;
#endif
  return affinity;
}

namespace {

ThreadAffinity* OriginalThreadAffinity() {
  static ThreadAffinity* original = GetThreadAffinity();
  return original;
}

}  // namespace

Status SetThreadAffinity(ThreadAffinity* affinity) {
  // Ensure original is initialized before changing.
  const ThreadAffinity* const original = OriginalThreadAffinity();
  JXL_CHECK(original != nullptr);

#if JXL_OS_WIN
  const HANDLE hThread = GetCurrentThread();
  const DWORD_PTR prev = SetThreadAffinityMask(hThread, affinity->set);
  if (prev == 0) return JXL_FAILURE("SetThreadAffinityMask failed");
  return true;
#elif JXL_OS_LINUX
  const int err = sched_setaffinity(0, sizeof(cpu_set_t), &affinity->set);
  if (err != 0) return JXL_FAILURE("sched_setaffinity failed");
  return true;
#elif JXL_OS_FREEBSD
  const pid_t pid = getpid();  // current thread
  const int err = cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, pid,
                                     sizeof(cpuset_t), &affinity->set);
  if (err != 0) return JXL_FAILURE("cpuset_setaffinity failed");
  return true;
#elif JXL_OS_MAC
  // As of 2019-03 we are unaware of a way to reliably restrict a thread to
  // core(s); THREAD_AFFINITY_POLICY is only a hint.
  (void)affinity;
  return false;
#elif JXL_OS_HAIKU
  // As of 2020-06 Haiku does not support pinning threads to cores.
  (void)affinity;
  return false;
#else
  printf("Don't know how to SetThreadAffinity on this platform.\n");
  return false;
#endif
}

std::vector<int> AvailableCPUs() {
  std::vector<int> cpus;
  cpus.reserve(128);
#if JXL_OS_WIN || JXL_OS_LINUX || JXL_OS_FREEBSD || JXL_OS_MAC || JXL_OS_HAIKU
  const ThreadAffinity* const affinity = OriginalThreadAffinity();
  for (int cpu = 0; cpu < static_cast<int>(sizeof(cpu_set_t)) * 8; ++cpu) {
    if (CPU_ISSET(cpu, &affinity->set)) {
      cpus.push_back(static_cast<int>(cpu));
    }
  }
#else
  cpus.push_back(0);
#endif
  return cpus;
}

Status PinThreadToCPU(const int cpu) {
#ifdef JXL_DISABLE_PINNING
  return false;
#else
#if JXL_OS_WIN || JXL_OS_LINUX || JXL_OS_FREEBSD || JXL_OS_MAC || JXL_OS_HAIKU
  ThreadAffinity affinity;
  CPU_ZERO(&affinity.set);
  CPU_SET(cpu, &affinity.set);
  return SetThreadAffinity(&affinity);
#else
  return false;
#endif
#endif
}

Status PinThreadToRandomCPU() {
  std::vector<int> cpus = AvailableCPUs();

  // Remove first two CPUs because interrupts are often pinned to them.
  JXL_CHECK(cpus.size() > 2);
  cpus.erase(cpus.begin(), cpus.begin() + 2);

  // Random choice to prevent burning up the same core.
  std::random_device device;
  std::ranlux48 generator(device());
  std::shuffle(cpus.begin(), cpus.end(), generator);
  const int cpu = cpus.front();

  return PinThreadToCPU(cpu);
}

namespace {

size_t DetectTotalMemoryMiB() {
#if JXL_OS_LINUX || JXL_OS_FREEBSD || JXL_OS_MAC
  const long page_size = sysconf(_SC_PAGESIZE);
  const long num_pages = sysconf(_SC_PHYS_PAGES);
  if (page_size == -1 || num_pages == -1) {
    JXL_WARNING("Failed to detect page size (%ld) and/or num pages (%ld)",
                page_size, num_pages);
    return 0;
  }
  JXL_ASSERT(page_size > 0 && num_pages > 0);
  const uint64_t bytes =
      static_cast<uint64_t>(num_pages) * static_cast<uint64_t>(page_size);
  return bytes >> 20;
#elif JXL_OS_WIN
  MEMORYSTATUSEX ms;
  ms.dwLength = sizeof(ms);
  if (!GlobalMemoryStatusEx(&ms)) {
    JXL_WARNING("Failed to get memory status");
    return 0;
  }
  const uint64_t bytes = ms.ullTotalPhys;
  // `bytes` excludes nonpaged pool reserved during boot; round up to whole MiB
  // to improve the estimate.
  return (bytes + (1U << 20) - 1) >> 20;
#elif JXL_OS_HAIKU
  system_info info;
  get_system_info(&info);
  return (info.max_pages * B_PAGE_SIZE) >> 20;
#else
  JXL_WARNING("Implement DetectTotalMemoryMiB for this platform");
  return 0;
#endif
}

}  // namespace

size_t TotalMemoryMiB() {
  static size_t mib = DetectTotalMemoryMiB();
  return mib;
}

/*
Status RunCommand(const std::vector<std::string>& args) {
#if _POSIX_VERSION >= 200112L
  // Avoid system(), but do not try to be over-zealous about not passing along
  // some special resources further (such as: inherited-not-marked-FD_CLOEXEC
  // file descriptors).
  std::vector<const char*> c_args;
  c_args.reserve(args.size() + 1);
  for (size_t i = 0; i < args.size(); ++i) {
    c_args.push_back(args[i].c_str());
  }
  c_args.push_back(nullptr);
  const pid_t pid = fork();
  if (pid == -1)  // fork() failed.
    return false;
  if (pid != 0) {  // Parent process.
    int ret_status;
    if (pid != waitpid(pid, &ret_status, 0)) {
      return false;  // waitpid() error.
    }
    return ret_status == 0;
  } else {  // Child process.
    execvp(c_args[0],
           // Address benign-but-annoying execvp() signature weirdness.
           const_cast<char* const*>(c_args.data()));
    JXL_ABORT("Failed to run command.\n");
  }
#elif JXL_OS_WIN
  // Synthesize a string for system(). And warn about it.
  // TODO(user): Fix this - research the safe way to run a command on Windows.
  // Likely, the solution is along these lines:
  // docs.microsoft.com/en-us/windows/desktop/ProcThread/creating-processes
  std::ostringstream cmd;
  std::copy(args.begin(), args.end(),
            std::ostream_iterator<std::string>(cmd, " "));
  printf(stderr, "Warning: Using system() on string: %s\n", cmd.str.c_str());
  int ret = system(cmd.str.c_str());
  if (errno != ENOENT &&  // Windows: Command interpreter not found.
      ret == 0) {
    return true;
  }
  return false;
#else
#error Neither a POSIX-1.2001 nor a Windows System.
#endif
}
*/

}  // namespace cpu
}  // namespace tools
}  // namespace jpegxl
