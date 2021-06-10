// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package org.jpeg.jpegxl.wrapper;

import java.nio.Buffer;

/**
 * Low level JNI wrapper.
 *
 * This class is package-private, should be only be used by high level wrapper.
 */
class DecoderJni {
  private static native void nativeGetBasicInfo(int[] context, Buffer data);
  private static native void nativeGetPixels(int[] context, Buffer data, Buffer pixels, Buffer icc);

  /** POJO that wraps some fields of JxlBasicInfo */
  static class BasicInfo {
    int width;
    int height;
    int pixelsSize;
    int iccSize;
    BasicInfo(int[] context) {
      checkStatusCode(context[0]);
      this.width = context[1];
      this.height = context[2];
      this.pixelsSize = context[3];
      this.iccSize = context[4];
    }
  }

  private static void checkStatusCode(int statusCode) {
    if (statusCode != 0) {
      // TODO(eustas): extend status code reporting
      throw new IllegalArgumentException("Corrupted JXL input");
    }
  }

  /** One-shot decoding. */
  static BasicInfo getBasicInfo(Buffer data, PixelFormat pixelFormat) {
    if (!data.isDirect()) {
      throw new IllegalArgumentException("data must be direct buffer");
    }
    int[] context = new int[5];
    context[0] = pixelFormat.ordinal();
    nativeGetBasicInfo(context, data);
    return new BasicInfo(context);
  }

  /** One-shot decoding. */
  static void getPixels(Buffer data, Buffer pixels, Buffer icc, PixelFormat pixelFormat) {
    if (!data.isDirect()) {
      throw new IllegalArgumentException("data must be direct buffer");
    }
    if (!pixels.isDirect()) {
      throw new IllegalArgumentException("pixels must be direct buffer");
    }
    if (!icc.isDirect()) {
      throw new IllegalArgumentException("icc must be direct buffer");
    }
    int[] context = new int[1];
    context[0] = pixelFormat.ordinal();
    nativeGetPixels(context, data, pixels, icc);
    checkStatusCode(context[0]);
  }

  /** Utility library, disable object construction. */
  private DecoderJni() {}
}
