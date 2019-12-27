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

#include "tools/cmdline.h"

#include <memory>
#include <string>

namespace jpegxl {
namespace tools {

void CommandLineParser::PrintHelp() const {
  fprintf(stderr, "Usage: %s [OPTIONS]\n",
          program_name_ ? program_name_ : "command");
  for (const auto& option : options_) {
    fprintf(stderr, " %s\n", option->help_flags().c_str());
    const char* help_text = option->help_text();
    if (help_text) {
      fprintf(stderr, "    %s\n", help_text);
    }
  }
  fprintf(stderr, " --help\n    Prints this help message.\n");
}

bool CommandLineParser::Parse(int argc, const char* argv[]) {
  if (argc) program_name_ = argv[0];
  int i = 1;  // argv[0] is the program name.
  while (i < argc) {
    if (!strcmp("--help", argv[i])) {
      // Returning false on Parse() forces to print the help message.
      return false;
    }
    bool found = false;
    for (const auto& option : options_) {
      if (option->Match(argv[i])) {
        // Parsing advances the value i on success.
        const char* arg = argv[i];
        if (!option->Parse(argc, argv, &i)) {
          fprintf(stderr, "Error parsing flag %s\n", arg);
          return false;
        }
        found = true;
        break;
      }
    }
    if (!found) {
      // No option matched argv[i].
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      return false;
    }
  }
  return true;
}

}  // namespace tools
}  // namespace jpegxl
