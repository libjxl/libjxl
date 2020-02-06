# Guidelines for JPEG-XL software

The following rules govern experimental software contributed for JPEG XL. Merge
requests not complying with these rules will not be accepted by the software
coordinators.

## License

The contribution shall be licensed under the Apache 2 license, with each file
beginning with the following header:

```
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
```

## Language

The software shall be compilable as C++ without custom preprocessors, and be
sent in the preferred representation for reading/development (without
obfuscation).

In general, please follow
[guidelines on features and style](https://google.github.io/styleguide/cppguide.html)
for newly added code.

In particular:

*   The code shall compile under clang++ (version 6) without any warnings.
*   Exceptions and RTTI shall not be used.
*   Features from C++11 may be used if desired, but not C++14 or later.
*   Integer types shall be fixed-size types from stdint.h (e.g. uint32_t), or size_t, or [u]intptr_t.
*   NULL shall not be used (use nullptr instead).
*   The code shall be formatted by running clang-format.
*   Macro names shall be UPPERCASE_WITH_UNDERSCORES.
*   External dependencies should be minimized; large dependencies such as Qt or Boost are not allowed.
*   Headers shall be surrounded by include guards of the form

    ```
    #ifndef FILENAME_H_
    #define FILENAME_H_
    #endif // FILENAME_H_
    ```

## Acceptance test

Your patch shall build in the "release" and "debug" configurations of `ci.sh`
and pass all tests (see [building and testing](doc/building_and_testing.md)).

The "opt" configuration will be tested in the Merge Request pipeline which is
equivalent to a "release" build with debug information.

Before merging results of a core experiment, the tests shall pass in the
"asan" and "msan" configurations.

If you added new functions, you should also add tests to these functions in the
same commit.
