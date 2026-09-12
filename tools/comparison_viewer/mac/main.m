// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#import <Cocoa/Cocoa.h>
#import <stdio.h>

#import "ImageComparisonWindow.h"

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
  return YES;
}

@end

static NSImage *loadImage(const char *const arg) {
  return [[NSImage alloc] initByReferencingFile:[[NSString alloc] initWithUTF8String:arg]];
}

int main(int argc, const char **argv) {
  if (argc != 3 && argc != 4) {
    fprintf(stderr, "Usage: %s <first image> <second image> [<reference image>]\n", argv[0]);
    return 1;
  }

  @autoreleasepool {
    [[NSApplication sharedApplication] setActivationPolicy:NSApplicationActivationPolicyAccessory];
    ImageComparisonWindow *window;
    NSImage *firstImage = loadImage(argv[1]);
    NSImage *secondImage = loadImage(argv[2]);
    NSImage *referenceImage = argc > 3 ? loadImage(argv[3]) : nil;
    window = [[ImageComparisonWindow alloc] initComparisonBetween:firstImage
                                                              and:secondImage
                                                    withReference:referenceImage];
    [window makeKeyAndOrderFront:nil];
    [NSApp setDelegate:[[AppDelegate alloc] init]];
    [NSApp activateIgnoringOtherApps:YES];
    [NSApp run];
  }
  return 0;
}
