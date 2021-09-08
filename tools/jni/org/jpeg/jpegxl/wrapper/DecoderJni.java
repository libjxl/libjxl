// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package org.jpeg.jpegxl.wrapper;

import java.nio.ByteBuffer;

/**
 * Low level JNI wrapper.
 *
 * This class is package-private, should be only be used by high level wrapper.
 */
class DecoderJni {
  private static native void nativeGetBasicInfo(int[] context, ByteBuffer data);
  private static native void nativeGetPixels(
      int[] context, ByteBuffer data, ByteBuffer pixels, ByteBuffer icc);

  static Status makeStatus(int statusCode) {
    switch (statusCode) {
      case 0:
        return Status.OK;
      case -1:
        return Status.INVALID_STREAM;
      case -2:
        return Status.NOT_SUPPORTED;
      case 1:
        return Status.NOT_ENOUGH_INPUT;
      default:
        throw new IllegalStateException("Unknown status code");
    }
  }

  static StreamInfo makeStreamInfo(int[] context) {
    StreamInfo result = new StreamInfo();
    result.status = makeStatus(context[0]);
    result.width = context[1];
    result.height = context[2];
    result.pixelsSize = context[3];
    result.iccSize = context[4];
    result.alphaBits = context[5];
    return result;
  }

  /** Decode stream information. */
  static StreamInfo getBasicInfo(ByteBuffer data, PixelFormat pixelFormat) {
    if (!data.isDirect()) {
      throw new IllegalArgumentException("data must be direct buffer");
    }
    int[] context = new int[6];
    context[0] = (pixelFormat == null) ? -1 : pixelFormat.ordinal();
    nativeGetBasicInfo(context, data);
    return makeStreamInfo(context);
  }

  /** One-shot decoding. */
  static Status getPixels(ByteBuffer data, ByteBuffer pixels, ByteBuffer icc, Colorspace colorspace,
      PixelFormat pixelFormat) {
    if (!data.isDirect()) {
      throw new IllegalArgumentException("data must be direct buffer");
    }
    if (!pixels.isDirect()) {
      throw new IllegalArgumentException("pixels must be direct buffer");
    }
    if (!icc.isDirect()) {
      throw new IllegalArgumentException("icc must be direct buffer");
    }
    if (pixelFormat == null) {
      throw new IllegalArgumentException("pixelFormat must be non-null");
    }
    if (colorspace != null && !pixelFormat.isF16) {
      throw new IllegalArgumentException("for color transform FP16 pixelFormat is expected");
    }
    int[] context = new int[2];
    context[0] = pixelFormat.ordinal();
    context[1] = (colorspace == null) ? -1 : colorspace.ordinal();
    nativeGetPixels(context, data, pixels, icc);
    return makeStatus(context[0]);
  }

  /** Utility library, disable object construction. */
  private DecoderJni() {}
}
