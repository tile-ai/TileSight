#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace tilesight_bench {

constexpr uint32_t kDefaultDataSeed = 1729;

#ifdef __CUDACC__
#define TILESIGHT_PATTERN_INLINE __host__ __device__ __forceinline__
#else
#define TILESIGHT_PATTERN_INLINE inline
#endif

// Versioned, reproducible integer payloads. Every word is nonzero, and nearby
// indices mix across all bits instead of producing zero/constant cache lines.
TILESIGHT_PATTERN_INLINE uint32_t data_word(uint64_t index, uint32_t seed) {
    uint32_t value = uint32_t(index) ^ seed ^ (uint32_t(index >> 32) * 0x9e3779b9u);
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value ? value : 0xa511e9b3u;
}

inline bool parse_data_seed(const char* text, uint32_t* seed) {
    if (!text || !*text) return false;
    for (const char* p = text; *p; ++p)
        if (*p < '0' || *p > '9') return false;
    errno = 0;
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || *end || value > std::numeric_limits<uint32_t>::max()) return false;
    *seed = uint32_t(value);
    return true;
}

#undef TILESIGHT_PATTERN_INLINE
}  // namespace tilesight_bench
