// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package org.jpeg.jpegxl.wrapper;

import java.nio.Buffer;
import java.nio.ByteBuffer;

/** JPEG XL JNI decoder wrapper. */
public class Decoder {
  /** Utility library, disable object construction. */
  private Decoder() {}

  public static class Options {
    Colorspace desiredColorspace;
    PixelFormat pixelFormat;

    Options setDesiredColorspace(Colorspace colorspace) {
      this.desiredColorspace = colorspace;
      return this;
    }

    Options setPixelFormat(PixelFormat pixelFormat) {
      this.pixelFormat = pixelFormat;
      return this;
    }
  }

  /** One-shot decoding. */
  public static ImageData decode(ByteBuffer data, Options options) {
    StreamInfo basicInfo = DecoderJni.getBasicInfo(data, options.pixelFormat);
    if (basicInfo.status != Status.OK) {
      throw new IllegalStateException("Decoding failed");
    }
    if (basicInfo.width < 0 || basicInfo.height < 0 || basicInfo.pixelsSize < 0
        || basicInfo.iccSize < 0) {
      throw new IllegalStateException("JNI has returned negative size");
    }
    ByteBuffer pixels = ByteBuffer.allocateDirect(basicInfo.pixelsSize);
    ByteBuffer icc = ByteBuffer.allocateDirect(basicInfo.iccSize);
    Status status =
        DecoderJni.getPixels(data, pixels, icc, options.desiredColorspace, options.pixelFormat);
    if (status != Status.OK) {
      throw new IllegalStateException("Decoding failed with status " + status);
    }
    return new ImageData(basicInfo.width, basicInfo.height, pixels, icc, options.desiredColorspace,
        options.pixelFormat);
  }

  // TODO(eustas): accept byte-array as input.
  public static StreamInfo decodeInfo(ByteBuffer data) {
    return DecoderJni.getBasicInfo(data, null);
  }
}
