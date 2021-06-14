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

  private static final int SIMPLE_IMAGE_DIM = 1024;
  private static final byte[] SIMPLE_IMAGE_BYTES =
      Base64.getDecoder().decode("/wr6H0GRCAYBAGAASzgkunkeVbaSBu95EXDn0e7ABz2ShAMA");

  private static final int PIXEL_IMAGE_DIM = 1;
  private static final byte[] PIXEL_IMAGE_BYTES =
      Base64.getDecoder().decode("/woAELASCBAQABwASxLFgoUkDA==");

  static ByteBuffer makeByteBuffer(byte[] src, int length) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(length);
    buffer.put(src, 0, length);
    return buffer;
  }

  static ByteBuffer makeSimpleImage() {
    return makeByteBuffer(SIMPLE_IMAGE_BYTES, SIMPLE_IMAGE_BYTES.length);
  }

  static void checkSimpleImageData(ImageData imageData) {
    if (imageData.width != SIMPLE_IMAGE_DIM) {
      throw new IllegalStateException("invalid width");
    }
    if (imageData.height != SIMPLE_IMAGE_DIM) {
      throw new IllegalStateException("invalid height");
    }
    int iccSize = imageData.icc.capacity();
    // Do not expect ICC profile to be some exact size; currently it is 732
    if (iccSize < 300 || iccSize > 1000) {
      throw new IllegalStateException("unexpected ICC profile size");
    }
  }

  static void checkPixelFormat(PixelFormat pixelFormat, int bytesPerPixel) {
    ImageData imageData = Decoder.decode(makeSimpleImage(), pixelFormat);
    checkSimpleImageData(imageData);
    if (imageData.pixels.limit() != SIMPLE_IMAGE_DIM * SIMPLE_IMAGE_DIM * bytesPerPixel) {
      throw new IllegalStateException("Unexpected pixels size");
    }
  }

  static void testRgba() {
    checkPixelFormat(PixelFormat.RGBA_8888, 4);
  }

  static void testRgbaF16() {
    checkPixelFormat(PixelFormat.RGBA_F16, 8);
  }

  static void testRgb() {
    checkPixelFormat(PixelFormat.RGB_888, 3);
  }

  static void testRgbF16() {
    checkPixelFormat(PixelFormat.RGB_F16, 6);
  }

  static void checkGetInfo(ByteBuffer data, int dim, int alphaBits) {
    StreamInfo streamInfo = Decoder.decodeInfo(data);
    if (streamInfo.status != Status.OK) {
      throw new IllegalStateException("Unexpected decoding error");
    }
    if (streamInfo.width != dim || streamInfo.height != dim) {
      throw new IllegalStateException("Invalid width / height");
    }
    if (streamInfo.alphaBits != alphaBits) {
      throw new IllegalStateException("Invalid alphaBits");
    }
  }

  static void testGetInfoNoAlpha() {
    checkGetInfo(makeSimpleImage(), SIMPLE_IMAGE_DIM, 0);
  }

  static void testGetInfoAlpha() {
    checkGetInfo(makeByteBuffer(PIXEL_IMAGE_BYTES, PIXEL_IMAGE_BYTES.length), PIXEL_IMAGE_DIM, 8);
  }

  static void testNotEnoughInput() {
    for (int i = 0; i < 6; ++i) {
      ByteBuffer jxlData = makeByteBuffer(SIMPLE_IMAGE_BYTES, i);
      StreamInfo streamInfo = Decoder.decodeInfo(jxlData);
      if (streamInfo.status != Status.NOT_ENOUGH_INPUT) {
        throw new IllegalStateException(
            "Expected 'not enough input', but got " + streamInfo.status + " " + i);
      }
    }
  }

  // Simple executable to avoid extra dependencies.
  public static void main(String[] args) {
    testRgba();
    testRgbaF16();
    testRgb();
    testRgbF16();
    testGetInfoNoAlpha();
    testGetInfoAlpha();
    testNotEnoughInput();
  }
}
