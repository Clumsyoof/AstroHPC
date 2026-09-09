#ifndef MORTON_HPP
#define MORTON_HPP

#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace astro {

struct BoundingBox {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    float extent_x() const { return max_x - min_x; }
    float extent_y() const { return max_y - min_y; }
    float extent_z() const { return max_z - min_z; }
    float max_extent() const {
        return std::max({extent_x(), extent_y(), extent_z(), 1e-6f});
    }
};

// Expands a 21-bit integer into 63 bits by inserting 2 zeros between each bit
inline uint64_t expand_bits_21(uint32_t v) {
    uint64_t x = v & 0x1fffffULL; // 21 bits
    x = (x | (x << 32)) & 0x001f00000000ffffULL;
    x = (x | (x << 16)) & 0x001f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

// Compacts 63 bits by extracting every 3rd bit into a 21-bit integer
inline uint32_t compact_bits_21(uint64_t x) {
    x &= 0x1249249249249249ULL;
    x = (x ^ (x >> 2))  & 0x10c30c30c30c30c3ULL;
    x = (x ^ (x >> 4))  & 0x100f00f00f00f00fULL;
    x = (x ^ (x >> 8))  & 0x001f0000ff0000ffULL;
    x = (x ^ (x >> 16)) & 0x001f00000000ffffULL;
    x = (x ^ (x >> 32)) & 0x1fffffULL;
    return static_cast<uint32_t>(x);
}

// Encodes 3D coordinates (x, y, z) into a 64-bit Morton (Z-order) key given a bounding box
inline uint64_t morton_encode_3d(float x, float y, float z, const BoundingBox& bbox) {
    float extent = bbox.max_extent();
    
    // Normalize coordinates to [0.0, 1.0]
    float nx = (x - bbox.min_x) / extent;
    float ny = (y - bbox.min_y) / extent;
    float nz = (z - bbox.min_z) / extent;

    // Clamp to [0, 2^21 - 1] = [0, 2097151]
    const float scale = 2097151.0f;
    uint32_t qx = static_cast<uint32_t>(std::max(0.0f, std::min(scale, nx * scale)));
    uint32_t qy = static_cast<uint32_t>(std::max(0.0f, std::min(scale, ny * scale)));
    uint32_t qz = static_cast<uint32_t>(std::max(0.0f, std::min(scale, nz * scale)));

    // Bit-interleaving: bit 3k = X, bit 3k+1 = Y, bit 3k+2 = Z
    return expand_bits_21(qx) | (expand_bits_21(qy) << 1) | (expand_bits_21(qz) << 2);
}

// Decodes a 64-bit Morton key back into normalized quantized coordinates (qx, qy, qz)
inline void morton_decode_3d(uint64_t morton, uint32_t& qx, uint32_t& qy, uint32_t& qz) {
    qx = compact_bits_21(morton);
    qy = compact_bits_21(morton >> 1);
    qz = compact_bits_21(morton >> 2);
}

} // namespace astro

#endif // MORTON_HPP
