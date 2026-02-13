// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package org.jpeg.jpegli.wrapper;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;

/** Jpegli JNI encoder wrapper. */
public class Encoder {
  static {
    JniHelper.ensureInitialized();
  }

  /** Utility library, disable object construction. */
  private Encoder() {}

  // NB(eustas): moving serialized to native might improve performance
  public static class Config {
    private enum Options {
      QUALITY,
    }

    private int[] serialized = new int[33];

    public Config setQuality(int quality) {
      setOption(Options.QUALITY, quality);
      return this;
    }

    private void setOption(Options option, int value) {
      int index = option.ordinal();
      serialized[32] |= 1 << index;
      serialized[index] = value;
    }
  }

  private static native int nativeInit();

  private static native int nativeEncode(
      int width, int height, int[] config, int[] data, WritableByteChannel output);

  private static class InitHelper {
    private static final int STATUS = nativeInit();
  }

  public static boolean ensureInitialized() {
    return (InitHelper.STATUS == 0);
  }

  /** One-shot encoding. */
  public static void encode(int[] color, int width, int height, Config config,
      WritableByteChannel output) throws IOException {
    if (!ensureInitialized()) {
      throw new IllegalStateException("Native library not initialized");
    }
    if (output == null) {
      throw new IllegalArgumentException("output is null");
    }
    if (color == null) {
      throw new IllegalArgumentException("color is null");
    }
    if ((width <= 0) || (height <= 0) || (color.length != width * height)) {
      throw new IllegalArgumentException("invalid image dimensions");
    }

    // TODO(eustas): what is colorspace?

    int status = nativeEncode(width, height, config.serialized, color, output);
    if (status != 0) {
      throw new IOException("Jpegli wrapper nativeProcess return code: " + status);
    }
  }

  /** One-shot encoding. */
  public static void encode(int[] color, int width, int height, int quality, OutputStream output)
      throws IOException {
    if (output == null) {
      throw new IllegalArgumentException("output is null");
    }
    try (WritableByteChannel channel = Channels.newChannel(output)) {
      encode(color, width, height, new Config().setQuality(quality), channel);
    }
  }
}
