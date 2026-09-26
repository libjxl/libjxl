// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#import "CenteredClipView.h"

@implementation CenteredClipView

- (NSRect)constrainBoundsRect:(NSRect)proposedBounds {
  NSRect bounds = [super constrainBoundsRect:proposedBounds];

  NSView *documentView = self.documentView;
  if (!documentView) return bounds;

  if (documentView.frame.size.width < bounds.size.width) {
    bounds.origin.x =
        documentView.frame.origin.x + (documentView.frame.size.width - bounds.size.width) / 2;
  }

  if (documentView.frame.size.height < bounds.size.height) {
    bounds.origin.y =
        documentView.frame.origin.y + (documentView.frame.size.height - bounds.size.height) / 2;
  }

  return bounds;
}

@end
