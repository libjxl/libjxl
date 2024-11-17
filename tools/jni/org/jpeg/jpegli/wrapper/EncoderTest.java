// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package org.jpeg.jpegli.wrapper;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;

/**
 * Tests for jpegli encoder wrapper.
 */
public class EncoderTest {
  static void checkTrue(boolean condition) {
    if (!condition) {
      throw new IllegalStateException("check failed");
    }
  }

  static void test64x64() throws IOException {
    int width = 64;
    int height = 64;
    int[] pixels = new int[width * height];
    for (int y = 0; y < 64; ++y) {
      for (int x = 0; x < 64; ++x) {
        pixels[x + 64 * y] = (x * 4) + ((y * 4) << 8);
      }
    }
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Encoder.encode(pixels, width, height, 99, out);
    System.err.println("Encoded size: " + out.size());
    checkTrue(out.size() > 0);
    byte[] encoded = Base64.getEncoder().encode(out.toByteArray());
    System.err.println("Base64: " + new String(encoded, UTF_8));
  }

  // Simple executable to avoid extra dependencies.
  public static void main(String[] args) throws IOException {
    test64x64();
  }
}
