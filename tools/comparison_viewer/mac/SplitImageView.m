// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#import "SplitImageView.h"

#import <AvailabilityMacros.h>
#import <QuartzCore/QuartzCore.h>

@implementation SplitImageView

NSSize leftImageSize;
NSSize rightImageSize;
NSSize referenceImageSize;
CALayer *leftImageLayer;
CALayer *rightImageLayer;
CALayer *referenceImageLayer;
CAShapeLayer *firstSeparator;
CAShapeLayer *secondSeparator;

- (instancetype)initComparisonBetween:(NSImage *)firstImage
                                  and:(NSImage *)secondImage
                        withReference:(NSImage *)referenceImage {
  leftImageSize = firstImage.size;
  rightImageSize = secondImage.size;
  CGFloat width = MAX(leftImageSize.width, rightImageSize.width);
  CGFloat height = MAX(leftImageSize.height, rightImageSize.height);
  if (referenceImage) {
    referenceImageSize = referenceImage.size;
    width = MAX(width, referenceImageSize.width);
    height = MAX(height, referenceImageSize.height);
  }

  self = [super initWithFrame:NSMakeRect(0, 0, width, height)];
  if (self) {
    self.layer = [CALayer layer];
    self.wantsLayer = YES;

    leftImageLayer = [CALayer layer];
    leftImageLayer.anchorPoint = NSMakePoint(0, 0);
    leftImageLayer.bounds = NSMakeRect(0, 0, leftImageSize.width, leftImageSize.height);
    leftImageLayer.contentsGravity = kCAGravityLeft;
    leftImageLayer.contents = firstImage;
    rightImageLayer = [CALayer layer];
    rightImageLayer.anchorPoint = NSMakePoint(0, 0);
    rightImageLayer.bounds = NSMakeRect(0, 0, rightImageSize.width, rightImageSize.height);
    rightImageLayer.contentsGravity = kCAGravityRight;
    rightImageLayer.contents = secondImage;
    if (referenceImage) {
      referenceImageLayer = [CALayer layer];
      referenceImageLayer.anchorPoint = NSMakePoint(0, 0);
      referenceImageLayer.bounds =
          NSMakeRect(0, 0, referenceImageSize.width, referenceImageSize.height);
      referenceImageLayer.contents = referenceImage;
    }

#if MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
      leftImageLayer.preferredDynamicRange = CADynamicRangeHigh;
      rightImageLayer.preferredDynamicRange = CADynamicRangeHigh;
      referenceImageLayer.preferredDynamicRange = CADynamicRangeHigh;
    } else
#endif
#if MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
        if (@available(macOS 14.0, *)) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      leftImageLayer.wantsExtendedDynamicRangeContent = YES;
      rightImageLayer.wantsExtendedDynamicRangeContent = YES;
      referenceImageLayer.wantsExtendedDynamicRangeContent = YES;
#pragma clang diagnostic pop
    }
#endif

    if (referenceImageLayer) {
      [self.layer addSublayer:referenceImageLayer];
    }
    [self.layer addSublayer:leftImageLayer];
    [self.layer addSublayer:rightImageLayer];

    NSArray *dashPattern = @[ @3, @1 ];
    CGMutablePathRef path = CGPathCreateMutable();
    CGPathMoveToPoint(path, nil, 0, 0);
    CGPathAddLineToPoint(path, nil, 0, height);
    firstSeparator = [CAShapeLayer layer];
    firstSeparator.strokeColor = NSColor.grayColor.CGColor;
    firstSeparator.lineWidth = 1;
    firstSeparator.lineDashPattern = dashPattern;
    firstSeparator.path = path;
    [self.layer addSublayer:firstSeparator];
    if (referenceImage) {
      secondSeparator = [CAShapeLayer layer];
      secondSeparator.strokeColor = NSColor.grayColor.CGColor;
      secondSeparator.lineWidth = 1;
      secondSeparator.lineDashPattern = dashPattern;
      secondSeparator.path = path;
      [self.layer addSublayer:secondSeparator];
    }
    CGPathRelease(path);

    firstSeparator.position = NSMakePoint(200, 0);

    NSTrackingArea *trackingArea = [[NSTrackingArea alloc]
        initWithRect:NSMakeRect(self.frame.origin.x - 1000, self.frame.origin.y - 1000,
                                self.frame.size.width + 2000, self.frame.size.height + 2000)
             options:NSTrackingMouseMoved | NSTrackingActiveAlways
               owner:self
            userInfo:nil];
    [self addTrackingArea:trackingArea];
  }
  return self;
}

- (void)mouseMoved:(NSEvent *)event {
  NSPoint where = [self convertPoint:[event locationInWindow] fromView:nil];
  [CATransaction begin];
  [CATransaction setDisableActions:YES];
#define CLAMP(x) (MAX(0, MIN(1, (x))))
  firstSeparator.hidden = NO;
  if (referenceImageLayer) {
    leftImageLayer.contentsRect = NSMakeRect(0, 0, CLAMP((where.x - 50) / leftImageSize.width), 1);
    rightImageLayer.contentsRect = NSMakeRect(CLAMP((where.x + 50) / rightImageSize.width), 0,
                                              CLAMP(1 - (where.x + 50) / rightImageSize.width), 1);
    firstSeparator.position = NSMakePoint(where.x - 50, 0);
    secondSeparator.position = NSMakePoint(where.x + 50, 0);
    secondSeparator.hidden = NO;
  } else {
    leftImageLayer.contentsRect = NSMakeRect(0, 0, CLAMP(where.x / leftImageSize.width), 1);
    rightImageLayer.contentsRect = NSMakeRect(CLAMP(where.x / rightImageSize.width), 0,
                                              CLAMP(1 - where.x / rightImageSize.width), 1);
    firstSeparator.position = NSMakePoint(where.x, 0);
  }
#undef CLAMP
  [CATransaction commit];
}

- (BOOL)acceptsFirstResponder {
  return YES;
}

- (void)keyDown:(NSEvent *)event {
  if (!(event.modifierFlags & NSEventModifierFlagNumericPad)) {
    [super keyDown:event];
    return;
  }
  NSString *characters = event.charactersIgnoringModifiers;
  NSUInteger length = characters.length;
  if (length != 1) {
    [super keyDown:event];
    return;
  }
  unichar key = [characters characterAtIndex:0];
  if (!(key == NSLeftArrowFunctionKey || key == NSRightArrowFunctionKey ||
        (referenceImageLayer && (key == NSUpArrowFunctionKey || key == NSDownArrowFunctionKey)))) {
    [super keyDown:event];
    return;
  }
  [CATransaction begin];
  [CATransaction setDisableActions:YES];
  firstSeparator.hidden = YES;
  secondSeparator.hidden = YES;
  switch (key) {
    case NSLeftArrowFunctionKey:
      leftImageLayer.contentsRect = NSMakeRect(0, 0, 1, 1);
      rightImageLayer.contentsRect = NSMakeRect(0, 0, 0, 0);
      break;
    case NSRightArrowFunctionKey:
      leftImageLayer.contentsRect = NSMakeRect(0, 0, 0, 0);
      rightImageLayer.contentsRect = NSMakeRect(0, 0, 1, 1);
      break;
    case NSUpArrowFunctionKey:
    case NSDownArrowFunctionKey:
      leftImageLayer.contentsRect = NSMakeRect(0, 0, 0, 0);
      rightImageLayer.contentsRect = NSMakeRect(0, 0, 0, 0);
      break;
  }
  [CATransaction commit];
}

@end
