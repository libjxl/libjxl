#include <iostream>
#include <cstdint>
#include <climits>
#include <algorithm>

// Mock of jxl::DivCeil
size_t DivCeil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// The vulnerable logic from GetMinSize in decode.cc
size_t GetMinSizeVulnerable(size_t xsize, size_t ysize, size_t num_channels, size_t bits) {
    size_t row_size = DivCeil(xsize * num_channels * bits, 8);
    size_t last_row_size = row_size;
    // Simplified: assuming align = 1
    return row_size * (ysize - 1) + last_row_size;
}

int main() {
    // Case 1: 64-bit overflow with large dimensions and stride
    // ysize = 2^30 (allowed by SizeHeader)
    // row_size = 2^34 (16 GiB, e.g. 2^30 pixels * 4 channels * 4 bytes/channel)
    size_t xsize = 1ULL << 30;
    size_t ysize = 1ULL << 30;
    size_t num_channels = 4;
    size_t bits = 32;

    std::cout << "--- 64-bit overflow test ---" << std::endl;
    std::cout << "Dimensions: " << xsize << "x" << ysize << ", " << num_channels << " channels, " << bits << " bits" << std::endl;
    
    // xsize * num_channels * bits = 2^30 * 4 * 32 = 2^37
    // row_size = 2^37 / 8 = 2^34
    // total = 2^34 * (2^30 - 1) + 2^34 = 2^34 * 2^30 = 2^64
    
    size_t min_size = GetMinSizeVulnerable(xsize, ysize, num_channels, bits);
    std::cout << "Calculated min_size: " << min_size << std::endl;
    
    if (min_size == 0) {
        std::cout << "SUCCESS: Demonstrated overflow to 0!" << std::endl;
    } else {
        std::cout << "FAILED: No overflow detected (check if you are on a 32-bit system)." << std::endl;
    }

    // Case 2: 32-bit overflow (even on 32-bit systems)
    // xsize = 2^28, num_channels = 4, bits = 32
    // xsize * num_channels * bits = 2^28 * 4 * 32 = 2^35 (Overflows 32-bit size_t)
    uint32_t xsize32 = 1U << 28;
    uint32_t num_channels32 = 4;
    uint32_t bits32 = 32;
    uint32_t row_size32 = (uint32_t)DivCeil((size_t)xsize32 * num_channels32 * bits32, 8);
    
    std::cout << "\n--- 32-bit overflow test ---" << std::endl;
    std::cout << "xsize32 * num_channels32 * bits32 (as 32-bit): " << (xsize32 * num_channels32 * bits32) << std::endl;
    
    return 0;
}
