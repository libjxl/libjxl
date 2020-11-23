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
  // Use stdout, not stderr, so help can easily be grepped.
  FILE* out = stdout;
  fprintf(out, "Usage: %s", program_name_ ? program_name_ : "command");

  for (const auto& option : options_) {
    if (option->positional()) {
      if (option->verbosity_level() > verbosity) continue;
      if (option->required()) {
        fprintf(out, " %s", option->help_flags().c_str());
      } else {
        fprintf(out, " [%s]", option->help_flags().c_str());
      }
    }
  }
  fprintf(out, " [OPTIONS...]\n");

  bool showed_all = true;
  for (const auto& option : options_) {
    if (option->verbosity_level() > verbosity) {
      showed_all = false;
      continue;
    }
    fprintf(out, " %s\n", option->help_flags().c_str());
    const char* help_text = option->help_text();
    if (help_text) {
      fprintf(out, "    %s\n", help_text);
    }
  }
  fprintf(out, " -h, --help\n    Prints this help message%s.\n",
          (showed_all ? "" : " (use -v to see more options)"));
}

bool CommandLineParser::Parse(int argc, const char* argv[]) {
  if (argc) program_name_ = argv[0];
  int i = 1;  // argv[0] is the program name.
  while (i < argc) {
    if (!strcmp("-h", argv[i]) || !strcmp("--help", argv[i])) {
      help_ = true;
      i++;
      continue;
    }
    if (!strcmp("-v", argv[i]) || !strcmp("--verbose", argv[i])) {
      verbosity++;
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
