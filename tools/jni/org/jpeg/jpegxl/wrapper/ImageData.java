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

package org.jpeg.jpegxl.wrapper;

import java.nio.Buffer;

/** POJO that contains necessary image data (dimensions, pixels,...). */
public class ImageData {
  final int width;
  final int height;
  final Buffer pixels;
  final Buffer icc;

  ImageData(int width, int height, Buffer pixels, Buffer icc) {
    this.width = width;
    this.height = height;
    this.pixels = pixels;
    this.icc = icc;
  }
}
