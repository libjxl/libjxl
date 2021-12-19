#ifndef FAST_LOSSLESS_H
#define FAST_LOSSLESS_H
#include <stdlib.h>

size_t FastLosslessEncode(const unsigned char* rgba, size_t width,
                          size_t row_stride, size_t height, size_t num_threads,
                          unsigned char** output);

#endif
