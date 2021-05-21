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

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

void ProcessInput(const char* filename) {
  std::ifstream ifs(filename, std::ios::binary);
  std::vector<char> contents((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
  ifs.close();
  std::cout << "Processing " << filename << std::endl;
  LLVMFuzzerTestOneInput(reinterpret_cast<uint8_t*>(contents.data()),
                         contents.size());
}

// Read files listed in args and pass their contents to "fuzzer".
int main(int argc, const char* argv[]) {
  if (argc == 2) {
    // No threaded runner for single inputs.
    ProcessInput(argv[1]);
  } else if (argc > 2) {
    auto runner = JxlThreadParallelRunnerMake(
        nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
    return JxlThreadParallelRunner(
        runner.get(), argv,
        /* init= */ +[](void*, size_t) -> JxlParallelRetCode { return 0; },
        /* func= */
        +[](void* opaque, uint32_t value, size_t) {
          const char** proc_argv = static_cast<const char**>(opaque);
          ProcessInput(proc_argv[value]);
        },
        /* start_rage= */ 1, /* end_range= */ argc);
  }
  return 0;
}
