// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package org.jpeg.jpegxl.wrapper;

import java.nio.ByteBuffer;
import java.util.Base64;

public class DecoderTest {
  static {
    String jniLibrary = System.getProperty("org.jpeg.jpegxl.wrapper.lib");
    if (jniLibrary != null) {
      try {
        System.load(new java.io.File(jniLibrary).getAbsolutePath());
      } catch (UnsatisfiedLinkError ex) {
        String message =
            "If the nested exception message says that some standard library (stdc++, tcmalloc, etc.) was not found, "
            + "it is likely that JDK discovered by the build system overrides library search path. "
            + "Try specifying a different JDK via JAVA_HOME environment variable and doing a clean build.";
        throw new RuntimeException(message, ex);
      }
    }
  }

  static ByteBuffer makeSimpleImage() {
    byte[] jxl = Base64.getDecoder().decode("/wr6H0GRCAYBAGAASzgkunkeVbaSBu95EXDn0e7ABz2ShAMA");
    ByteBuffer jxlData = ByteBuffer.allocateDirect(jxl.length);
    jxlData.put(jxl);
    return jxlData;
  }

  static void checkSimpleImageData(ImageData imageData) {
    if (imageData.width != 1024) {
      throw new IllegalStateException("invalid width");
    }
    if (imageData.height != 1024) {
      throw new IllegalStateException("invalid height");
    }
    int iccSize = imageData.icc.capacity();
    // Do not expect ICC profile to be some exact size; currently it is 732
    if (iccSize < 300 || iccSize > 1000) {
      throw new IllegalStateException("unexpected ICC profile size");
    }
  }

  static void testRgba() {
    ImageData imageData = Decoder.decode(makeSimpleImage());
    checkSimpleImageData(imageData);
    if (imageData.pixels.limit() != 1024 * 1024 * 4) {
      throw new IllegalStateException("Expected 4 bytes per pixels (RGBA_8888)");
    }
  }

  static void testRgbaF16() {
    ImageData imageData = Decoder.decode(makeSimpleImage(), PixelFormat.RGBA_F16);
    checkSimpleImageData(imageData);
    if (imageData.pixels.limit() != 1024 * 1024 * 8) {
      throw new IllegalStateException("Expected 8 bytes per pixels (RGBA_F16)");
    }
  }
  // Simple executable to avoid extra dependencies.
  public static void main(String[] args) {
    testRgba();
    testRgbaF16();
  }
}
