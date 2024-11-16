// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package org.jpeg.jpegxl.wrapper;

import java.io.File;

/**
 * Helper for native library loading.
 *
 * Normally the wrapper is responsible for (lazy) library loading; though it is
 * sometimes necessary to make the moment of loading more deterministic.
 */
public class JniHelper {
  /** Utility library, disable object construction. */
  private JniHelper() {}

  static {
    String jniLibrary = System.getProperty("org.jpeg.jpegxl.wrapper.lib");
    if (jniLibrary != null) {
      try {
        System.load(new File(jniLibrary).getAbsolutePath());
      } catch (UnsatisfiedLinkError ex) {
        String message = "If the nested exception message says that some standard library (stdc++, "
            + "tcmalloc, etc.) was not found, it is likely that JDK discovered by the "
            + "build system overrides library search path. Try specifying a different "
            + "JDK via JAVA_HOME environment variable and doing a clean build.";
        throw new IllegalStateException(message, ex);
      }
    }
  }

  static void ensureInitialized() {
    // Do nothing, just trigger static initializer.
  }
}
