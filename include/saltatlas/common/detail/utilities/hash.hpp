// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// This file contains public domain code from SMHasher
// (https://github.com/aappleby/smhasher/tree/master).
//
// From the MurmurHash3 file:
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#pragma once

#include <cstdint>

namespace saltatlas::detail::murmurhash3 {

#define FORCE_INLINE inline __attribute__((always_inline))

inline uint32_t rotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}

inline uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

FORCE_INLINE uint32_t getblock32(const uint32_t *p, int i) { return p[i]; }

FORCE_INLINE uint64_t getblock64(const uint64_t *p, int i) { return p[i]; }

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche

FORCE_INLINE uint32_t fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

//----------

FORCE_INLINE uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;

  return k;
}

void MurmurHash3_x64_128(const void *key, const int len, const uint32_t seed,
                         void *out) {
  const uint8_t *data    = (const uint8_t *)key;
  const int      nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = 0x87c37b91114253d5ULL;
  const uint64_t c2 = 0x4cf5ad432745937fULL;

  //----------
  // body

  const uint64_t *blocks = (const uint64_t *)(data);

  for (int i = 0; i < nblocks; i++) {
    uint64_t k1 = getblock64(blocks, i * 2 + 0);
    uint64_t k2 = getblock64(blocks, i * 2 + 1);

    k1 *= c1;
    k1 = rotl64(k1, 31);
    k1 *= c2;
    h1 ^= k1;

    h1 = rotl64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = rotl64(k2, 33);
    k2 *= c1;
    h2 ^= k2;

    h2 = rotl64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;
  }

  //----------
  // tail

  const uint8_t *tail = (const uint8_t *)(data + nblocks * 16);

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (len & 15) {
    case 15:
      k2 ^= ((uint64_t)tail[14]) << 48;
      [[fallthrough]];
    case 14:
      k2 ^= ((uint64_t)tail[13]) << 40;
      [[fallthrough]];
    case 13:
      k2 ^= ((uint64_t)tail[12]) << 32;
      [[fallthrough]];
    case 12:
      k2 ^= ((uint64_t)tail[11]) << 24;
      [[fallthrough]];
    case 11:
      k2 ^= ((uint64_t)tail[10]) << 16;
      [[fallthrough]];
    case 10:
      k2 ^= ((uint64_t)tail[9]) << 8;
      [[fallthrough]];
    case 9:
      k2 ^= ((uint64_t)tail[8]) << 0;
      k2 *= c2;
      k2 = rotl64(k2, 33);
      k2 *= c1;
      h2 ^= k2;
      [[fallthrough]];

    case 8:
      k1 ^= ((uint64_t)tail[7]) << 56;
      [[fallthrough]];
    case 7:
      k1 ^= ((uint64_t)tail[6]) << 48;
      [[fallthrough]];
    case 6:
      k1 ^= ((uint64_t)tail[5]) << 40;
      [[fallthrough]];
    case 5:
      k1 ^= ((uint64_t)tail[4]) << 32;
      [[fallthrough]];
    case 4:
      k1 ^= ((uint64_t)tail[3]) << 24;
      [[fallthrough]];
    case 3:
      k1 ^= ((uint64_t)tail[2]) << 16;
      [[fallthrough]];
    case 2:
      k1 ^= ((uint64_t)tail[1]) << 8;
      [[fallthrough]];
    case 1:
      k1 ^= ((uint64_t)tail[0]) << 0;
      k1 *= c1;
      k1 = rotl64(k1, 31);
      k1 *= c2;
      h1 ^= k1;
  };

  //----------
  // finalization

  h1 ^= len;
  h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  ((uint64_t *)out)[0] = h1;
  ((uint64_t *)out)[1] = h2;
}

}  // namespace saltatlas::detail::murmurhash3

namespace saltatlas {

template <unsigned int seed = 123>
struct hash {
  template <typename T>
  inline std::size_t operator()(const T &key) const noexcept {
    uint64_t hash_val[2];
    detail::murmurhash3::MurmurHash3_x64_128(&key, sizeof(T), seed, &hash_val);
    return hash_val[0];
  }
};

}  // namespace saltatlas