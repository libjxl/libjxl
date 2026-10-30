// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#import "ImageComparisonWindow.h"

#import "CenteredClipView.h"
#import "SplitImageView.h"

@implementation ImageComparisonWindow

- (instancetype)initComparisonBetween:(NSImage *)firstImage
                                  and:(NSImage *)secondImage
                        withReference:(NSImage *)referenceImage {
  NSRect contentRect = NSMakeRect(0, 0, 600, 450);
  self = [super initWithContentRect:contentRect
                          styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                    NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                            backing:NSBackingStoreBuffered
                              defer:NO];
  if (self) {
    self.releasedWhenClosed = NO;
    self.title = @"libjxl image comparison tool";

    NSScrollView *scrollView = [[NSScrollView alloc] initWithFrame:contentRect];
    scrollView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    scrollView.hasHorizontalScroller = YES;
    scrollView.hasVerticalScroller = YES;
    scrollView.autohidesScrollers = YES;

    NSView *images = [[SplitImageView alloc] initComparisonBetween:firstImage
                                                               and:secondImage
                                                     withReference:referenceImage];
    NSClipView *clipView = [[CenteredClipView alloc] initWithFrame:scrollView.contentView.frame];
    clipView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    scrollView.contentView = clipView;
    scrollView.documentView = images;
    scrollView.allowsMagnification = YES;
    self.contentView = scrollView;

    NSSize frameSize = [NSScrollView frameSizeForContentSize:images.frame.size
                                     horizontalScrollerClass:scrollView.horizontalScroller.class
                                       verticalScrollerClass:scrollView.verticalScroller.class
                                                  borderType:scrollView.borderType
                                                 controlSize:NSControlSizeRegular
                                               scrollerStyle:scrollView.scrollerStyle];

    [self setFrame:NSIntersectionRect(self.screen.visibleFrame,
                                      NSMakeRect(0, 0, frameSize.width, frameSize.height))
           display:YES];
    [self center];
  }
  return self;
}

- (void)cancelOperation:(id)sender {
  [self close];
}

@end
