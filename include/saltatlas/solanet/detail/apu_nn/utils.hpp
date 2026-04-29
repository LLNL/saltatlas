// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>

#if defined(__HIPCC__)
#define SALTATLAS_HD_HOST __host__
#define SALTATLAS_HD_DEVICE __device__
#define SALTATLAS_HD_HD __host__ __device__
#define SALTATLAS_HD_GLOBAL __global__
#define SALTATLAS_HD_SHARED __shared__
#define SALTATLAS_HD_FORCEINLINE __forceinline__
#else
// Plain C++ compilation: make them no-ops
#define SALTATLAS_HD_HOST
#define SALTATLAS_HD_DEVICE
#define SALTATLAS_HD_HD
#define SALTATLAS_HD_GLOBAL
#define SALTATLAS_HD_SHARED
#define SALTATLAS_HD_FORCEINLINE inline
#endif

#define SALTATLAS_HIP_CHECK(call)                                       \
  do {                                                                  \
    hipError_t err = (call);                                            \
    if (err != hipSuccess) {                                            \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err));                             \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace saltatlas::solanet::apu_nn {

SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_global_thread_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_global_thread_count() {
  return gridDim.x * blockDim.x;
}

template <int kWarpSize = 64>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_global_warp_id() {
  return get_global_thread_id() /
         kWarpSize;  // Assuming warp size of 64 for MI300A
}

template <int kWarpSize = 64>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_local_warp_id() {
  return threadIdx.x / kWarpSize;
}

template <int kWarpSize = 64>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_lane_id() {
  if constexpr (kWarpSize == 32) {
    return threadIdx.x & 31;
  } else if constexpr (kWarpSize == 64) {
    return threadIdx.x & 63;
  }
  return threadIdx.x % kWarpSize;  // Assuming warp size of 64 for MI300A
}

template <typename T>
SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE bool nearly_equal(
    const T a, const T b,
    const T eps = std::numeric_limits<T>::epsilon() * 10) {
  if constexpr (std::is_floating_point<T>::value) {
    return abs(a - b) < eps;
  }
  return a == b;
}

SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE void sync_warp() {
#ifdef __HIPCC__
  // TODO: make sure if this is actually needed for ROCM
  __threadfence_block();
  // __syncwarp();
  // __syncthreads();
  // https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/how-to/hip_cpp_language_extensions.html#synchronization-functions
#else
  // Host compilation path: no-op.
#endif
}

/// \brief Return and update state with a simple linear congruential generator
/// (LCG).
SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE uint64_t lcg64(uint64_t& state) {
  state = state * 6364136223846793005ULL + 1ULL;
  return state;
}

/// \brief Return a random number in [0, range) using LCG. Caller must ensure
/// range > 0.
SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE uint64_t
lcg_rand(uint64_t& state, const uint64_t range) {
  return (lcg64(state) >> 32) % range;
}

using rnd_state_type = rocrand_state_xorwow;

SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE void rnd_init(const int       tid,
                                                       const uint64_t  seed,
                                                       rnd_state_type& state) {
  rocrand_init(seed, tid, 0, &state);
}

SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE auto rnd_next(rnd_state_type& state) {
  return rocrand(&state);
}

SALTATLAS_HD_HD inline int u64_to_commas(uint64_t v, char* out) {
  // Max for uint64: "18,446,744,073,709,551,615" -> 26 chars + '\0' => 27
  // Caller must provide at least 27 bytes.
  char tmp[32];
  int  n = 0;

  // Special case: 0
  if (v == 0) {
    out[0] = '0';
    out[1] = '\0';
    return 1;
  }

  // Build reversed into tmp with commas every 3 digits
  int digit_count = 0;
  while (v > 0) {
    if (digit_count == 3) {
      tmp[n++]    = ',';
      digit_count = 0;
    }
    uint64_t q = v / 10;
    uint64_t r = v - q * 10;
    tmp[n++]   = char('0' + r);
    v          = q;
    digit_count++;
  }

  // Reverse into out
  for (int i = 0; i < n; i++) {
    out[i] = tmp[n - 1 - i];
  }
  out[n] = '\0';
  return n;
}

SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE size_t
align_up(const size_t size, const size_t alignment) {
  const size_t mask = alignment - 1;
  return (size + mask) & ~mask;
}

template <typename T>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE T shfl_up(T v, int delta) {
  // HIP has __shfl_up for int/float; for other types, specialize as needed.
  return __shfl_up(v, delta);
}

// Warp-exclusive scan for int (warp size 64 on MI300A)
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int warp_exclusive_scan_int_64(
    int x) {
  int lane = threadIdx.x & 63;
  int sum  = x;  // inclusive scan first
#pragma unroll
  for (int d = 1; d < 64; d <<= 1) {
    int y = __shfl_up(sum, d);
    if (lane >= d) sum += y;
  }
  return sum - x;  // exclusive
}

// Set the highest bit of value to 1
template <typename T>
SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE T set_msb(T value) {
  value |= (static_cast<T>(1) << (sizeof(T) * 8 - 1));
  return value;
}

// Get the value with the highest bit cleared
template <typename T>
SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE T clear_msb(const T value) {
  return value & ~(static_cast<T>(1) << (sizeof(T) * 8 - 1));
}

// Get the flag bit
template <typename T>
SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE bool get_msb(const T value) {
  return (value & (static_cast<T>(1) << (sizeof(T) * 8 - 1))) != 0;
}

SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE uint64_t
next_power_of_two(uint64_t v) {
  if (v == 0) return 1;
  v--;
  for (size_t i = 1; i < sizeof(uint64_t) * 8; i <<= 1) {
    v |= v >> i;
  }
  return v + 1;
};

}  // namespace saltatlas::solanet::apu_nn
