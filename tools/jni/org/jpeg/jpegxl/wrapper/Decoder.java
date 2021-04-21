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
import java.nio.ByteBuffer;

/** JPEG XL JNI decoder wrapper. */
public class Decoder {
  /** Utility library, disable object construction. */
  private Decoder() {}

  /** One-shot decoding. */
  public static ImageData decode(Buffer data) {
    DecoderJni.BasicInfo basicInfo = DecoderJni.getBasicInfo(data);
    if (basicInfo.width < 0 || basicInfo.height < 0 || basicInfo.pixelsSize < 0
        || basicInfo.iccSize < 0) {
      throw new IllegalStateException("JNI has returned negative size");
    }
    Buffer pixels = ByteBuffer.allocateDirect(basicInfo.pixelsSize);
    Buffer icc = ByteBuffer.allocateDirect(basicInfo.iccSize);
    DecoderJni.getPixels(data, pixels, icc);
    return new ImageData(basicInfo.width, basicInfo.height, pixels, icc);
  }
}
