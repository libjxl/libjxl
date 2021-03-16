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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

// Read files listed in args and pass their contents to "fuzzer".
int main(int argc, const char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::ifstream ifs(argv[i], std::ios::binary);
    std::vector<char> contents((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());
    ifs.close();
    std::cout << "Processing " << argv[i] << std::endl;
    LLVMFuzzerTestOneInput(reinterpret_cast<uint8_t*>(contents.data()),
                           contents.size());
  }
}
