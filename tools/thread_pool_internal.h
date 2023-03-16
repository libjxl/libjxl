// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_THREAD_POOL_INTERNAL_H_
#define TOOLS_THREAD_POOL_INTERNAL_H_

#include <jxl/parallel_runner.h>
#include <stddef.h>

#include <cmath>

#include "lib/jxl/base/data_parallel.h"
#include "lib/threads/thread_parallel_runner_internal.h"

namespace jpegxl {
namespace tools {

// Helper class to pass an internal ThreadPool-like object using threads.
class ThreadPoolInternal : public jxl::ThreadPool {
 public:
  // Starts the given number of worker threads and blocks until they are ready.
  // "num_worker_threads" defaults to one per hyperthread. If zero, all tasks
  // run on the main thread.
  explicit ThreadPoolInternal(
      int num_worker_threads = std::thread::hardware_concurrency())
      : ThreadPool(&jpegxl::ThreadParallelRunner::Runner,
                   static_cast<void*>(&runner_)),
        runner_(num_worker_threads) {}

  ThreadPoolInternal(const ThreadPoolInternal&) = delete;
  ThreadPoolInternal& operator&(const ThreadPoolInternal&) = delete;

  size_t NumThreads() const { return runner_.NumThreads(); }

 private:
  jpegxl::ThreadParallelRunner runner_;
};

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_THREAD_POOL_INTERNAL_H_
