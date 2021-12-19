#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <thread>

#include "fast_lossless.h"
#include "lodepng.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s in.png out.jxl [num_threads [num_reps]]\n",
            argv[0]);
    return 1;
  }

  const char* in = argv[1];
  const char* out = argv[2];
  size_t num_threads = argc >= 4 && atoi(argv[3]) != 0
                           ? atoi(argv[3])
                           : std::thread::hardware_concurrency();
  size_t num_reps = argc >= 5 ? atoi(argv[4]) : 0;

  unsigned char* png;
  unsigned width, height;

  unsigned error = lodepng_decode32_file(&png, &width, &height, in);

  if (error) {
    fprintf(stderr, "lodepng error %u: %s\n", error, lodepng_error_text(error));
    return 1;
  }

  size_t encoded_size = 0;
  unsigned char* encoded = nullptr;

  if (num_reps > 0) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t _ = 0; _ < num_reps; _++) {
      free(encoded);
      encoded_size = FastLosslessEncode(png, width, width * 4, height,
                                        num_threads, &encoded);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    float us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    size_t pixels = size_t{width} * size_t{height} * num_reps;
    float mps = pixels / us;
    fprintf(stderr, "%10.3f MP/s\n", mps);
    fprintf(stderr, "%10.3f bits/pixel\n",
            encoded_size * 8.0 / float(width) / float(height));
  } else {
    encoded_size = FastLosslessEncode(png, width, width * 4, height,
                                      num_threads, &encoded);
  }

  FILE* o = fopen(out, "w");
  if (!o) {
    fprintf(stderr, "error opening %s: %s\n", out, strerror(errno));
    return 1;
  }
  if (fwrite(encoded, 1, encoded_size, o) != encoded_size) {
    fprintf(stderr, "error writing to %s: %s\n", out, strerror(errno));
  }
  fclose(o);
}
