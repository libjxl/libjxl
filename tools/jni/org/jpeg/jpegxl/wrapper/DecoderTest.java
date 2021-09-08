// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package org.jpeg.jpegxl.wrapper;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

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
  // Base64: "/wr6H0GRCAYBAGAASzgkunkeVbaSBu95EXDn0e7ABz2ShAMA"
  private static final byte[] SIMPLE_IMAGE_BYTES = {-1, 10, -6, 31, 65, -111, 8, 6, 1, 0, 96, 0, 75,
      56, 36, -70, 121, 30, 85, -74, -110, 6, -17, 121, 17, 112, -25, -47, -18, -64, 7, 61, -110,
      -124, 3, 0};

  private static final int PIXEL_IMAGE_DIM = 1;
  // Base64: "/woAELASCBAQABwASxLFgoUkDA=="
  private static final byte[] PIXEL_IMAGE_BYTES = {
      -1, 10, 0, 16, -80, 18, 8, 16, 16, 0, 28, 0, 75, 18, -59, -126, -123, 36, 12};

  // Base64:
  // "/woAAASASAgEAQBIAEsSxYKFFJdtAKxt8+9r/n34Aw=="
  private static final byte[] RGB_RGB_BYTES = {-1, 10, 0, 0, 4, -128, 72, 8, 4, 1, 0, 72, 0, 75, 18,
      -59, -126, -123, 20, -105, 109, 0, -84, 109, -13, -17, 107, -2, 125, -8, 3};

  // Base64:
  // "/woAAASAEC4UBiTMA0RuetFofl5XodBw1/O9QMOv+UIMd2nutmuEYa4HnKjBjtSm9ZyqBjzxD+CTfco69npucZV/s5t5h"
  // "DlG8jOmmWQoZD6RQh0Lk+d5LgmIfgjEliQjBkNuMXAwWuZNiItTZw0EtdwJCAQBAEgASxLFgoUUl20ArG3z72v+ffgD"
  private static final byte[] RGB_BGR_BYTES = {-1, 10, 0, 0, 4, -128, 16, 46, 20, 6, 36, -52, 3, 68,
      110, 122, -47, 104, 126, 94, 87, -95, -48, 112, -41, -13, -67, 64, -61, -81, -7, 66, 12, 119,
      105, -18, -74, 107, -124, 97, -82, 7, -100, -88, -63, -114, -44, -90, -11, -100, -86, 6, 60,
      -15, 15, -32, -109, 125, -54, 58, -10, 122, 110, 113, -107, 127, -77, -101, 121, -124, 57, 70,
      -14, 51, -90, -103, 100, 40, 100, 62, -111, 66, 29, 11, -109, -25, 121, 46, 9, -120, 126, 8,
      -60, -106, 36, 35, 6, 67, 110, 49, 112, 48, 90, -26, 77, -120, -117, 83, 103, 13, 4, -75, -36,
      9, 8, 4, 1, 0, 72, 0, 75, 18, -59, -126, -123, 20, -105, 109, 0, -84, 109, -13, -17, 107, -2,
      125, -8, 3};

  // Base64:
  // "/woAAASAEC4UBiTMA0RuelFjfl5XodBw1/6zQMOv+UIMdzncbdcIw1wbdNRgLbU5errqAJ74B/DJPmUn9s50u8K/2c08x"
  // "0gj+BnTTDIUMp9IodrC5HmeEYMhtxhojJZ5E+Li1FkD4ZKAOA+B2JJELXcCCAQBAEgASxLFgoUUl20ArG3z72v+ffgD"
  private static final byte[] RGB_GRB_BYTES = {-1, 10, 0, 0, 4, -128, 16, 46, 20, 6, 36, -52, 3, 68,
      110, 122, 81, 99, 126, 94, 87, -95, -48, 112, -41, -2, -77, 64, -61, -81, -7, 66, 12, 119, 57,
      -36, 109, -41, 8, -61, 92, 27, 116, -44, 96, 45, -75, 57, 122, -70, -22, 0, -98, -8, 7, -16,
      -55, 62, 101, 39, -10, -50, 116, -69, -62, -65, -39, -51, 60, -57, 72, 35, -8, 25, -45, 76,
      50, 20, 50, -97, 72, -95, -38, -62, -28, 121, -98, 17, -125, 33, -73, 24, 104, -116, -106,
      121, 19, -30, -30, -44, 89, 3, -31, -110, -128, 56, 15, -127, -40, -110, 68, 45, 119, 2, 8, 4,
      1, 0, 72, 0, 75, 18, -59, -126, -123, 20, -105, 109, 0, -84, 109, -13, -17, 107, -2, 125, -8,
      3};

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
    ImageData imageData =
        Decoder.decode(makeSimpleImage(), new Decoder.Options().setPixelFormat(pixelFormat));
    checkSimpleImageData(imageData);
    if (imageData.pixels.limit() != SIMPLE_IMAGE_DIM * SIMPLE_IMAGE_DIM * bytesPerPixel) {
      throw new IllegalStateException("Unexpected pixels size");
    }
  }

  static double parseF16(char raw) {
    int sign = ((raw & 0x8000) == 0) ? 1 : -1;
    // Don't care about subnormals, etc.
    int mantissa = 1024 + (raw & 1023);
    int exponent = ((raw >> 10) & 31) - 15;
    double base = Math.pow(2, exponent);
    return sign * mantissa * base;
  }

  static int getPixelColor(ByteBuffer pixels, PixelFormat pixelFormat, int n) {
    int stride = 0;
    switch (pixelFormat) {
      case RGBA_F16:
        stride = 8;
        break;
      case RGB_F16:
        stride = 6;
        break;
      default:
        throw new IllegalArgumentException("Invalid pixelFormat " + pixelFormat);
    }
    pixels = pixels.order(ByteOrder.LITTLE_ENDIAN);
    double rgb[] = new double[3];
    for (int i = 0; i < 3; ++i) rgb[i] = parseF16(pixels.getChar(n * stride + 2 * i));
    for (int i = 0; i < 3; ++i) {
      boolean match = true;
      for (int j = 0; j < 3; ++j) {
        if (i == j) {
          if ((int) rgb[j] != 1024) {
            match = false;
          }
        } else {
          if (rgb[j] >= 1) {
            match = false;
          }
        }
      }
      if (match) {
        return i;
      }
    }
    return -1;
  }

  static void checkRgbProfile(
      byte[] data, PixelFormat pixelFormat, Colorspace colorspace, int... expectedOrder) {
    ImageData imageData = Decoder.decode(makeByteBuffer(data, data.length),
        new Decoder.Options().setPixelFormat(pixelFormat).setDesiredColorspace(colorspace));
    if ((imageData.width != 3) || (imageData.height != 1)) {
      throw new IllegalStateException("invalid width or height");
    }
    for (int i = 0; i < 3; ++i) {
      int color = getPixelColor(imageData.pixels, pixelFormat, i);
      if (color != expectedOrder[i]) {
        throw new IllegalStateException(
            "Wanted " + expectedOrder[i] + " @ " + i + ", but got " + color);
      }
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

  static void testRgbConverted() {
    checkRgbProfile(RGB_RGB_BYTES, PixelFormat.RGBA_F16, Colorspace.SRGB, 0, 1, 2);
  }

  static void testRgbNoConversion() {
    checkRgbProfile(RGB_RGB_BYTES, PixelFormat.RGBA_F16, null, 0, 1, 2);
  }

  static void testBgrConverted() {
    checkRgbProfile(RGB_BGR_BYTES, PixelFormat.RGBA_F16, Colorspace.SRGB, 2, 1, 0);
  }

  static void testBgrNoConversion() {
    checkRgbProfile(RGB_BGR_BYTES, PixelFormat.RGBA_F16, null, 0, 1, 2);
  }

  static void testGrbConverted() {
    checkRgbProfile(RGB_GRB_BYTES, PixelFormat.RGBA_F16, Colorspace.SRGB, 1, 0, 2);
  }

  static void testGrbNoConversion() {
    checkRgbProfile(RGB_GRB_BYTES, PixelFormat.RGBA_F16, null, 0, 1, 2);
  }

  static boolean probeColorConversion() {
    try {
      Decoder.decode(makeByteBuffer(RGB_RGB_BYTES, RGB_RGB_BYTES.length),
          new Decoder.Options()
              .setPixelFormat(PixelFormat.RGBA_F16)
              .setDesiredColorspace(Colorspace.SRGB));
    } catch (IllegalStateException ex) {
      return false;
    }
    return true;
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

    testRgbNoConversion();
    testBgrNoConversion();
    testGrbNoConversion();

    if (probeColorConversion()) {
      testRgbConverted();
      testBgrConverted();
      testGrbConverted();
    }
  }
}
