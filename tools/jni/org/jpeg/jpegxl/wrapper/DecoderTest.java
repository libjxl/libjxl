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

  // Simple executable to avoid extra dependencies.
  public static void main(String[] args) {
    byte[] jxl = Base64.getDecoder().decode("/wr6H0GRCAYBAGAASzgkunkeVbaSBu95EXDn0e7ABz2ShAMA");
    ByteBuffer jxlData = ByteBuffer.allocateDirect(jxl.length);
    jxlData.put(jxl);
    ImageData imageData = Decoder.decode(jxlData);
    if (imageData.width != 1024)
      throw new IllegalStateException("invalid width");
    if (imageData.height != 1024)
      throw new IllegalStateException("invalid height");
    int iccSize = imageData.icc.capacity();
    // Do not expect ICC profile to be some exact size; currently it is 732
    if (iccSize < 300 || iccSize > 1000)
      throw new IllegalStateException("unexpected ICC profile size");
  }
}
