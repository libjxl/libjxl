# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

enable_language(OBJC)

add_executable(compare_images
  main.m
  CenteredClipView.h
  CenteredClipView.m
  ImageComparisonWindow.h
  ImageComparisonWindow.m
  SplitImageView.h
  SplitImageView.m
)
target_compile_options(compare_images PRIVATE -fobjc-arc)
target_link_libraries(compare_images PRIVATE "-framework Cocoa -framework QuartzCore")
set_target_properties(compare_images PROPERTIES RUNTIME_OUTPUT_DIRECTORY ..)
