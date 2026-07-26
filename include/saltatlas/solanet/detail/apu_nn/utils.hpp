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

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <curand_kernel.h>
#else
#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
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

#if defined(__CUDACC__)
// CUDA backend: map the HIP runtime API used in apu_nn to the CUDA runtime so
// the same sources compile with nvcc. Kept to exactly the calls apu_nn uses.
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipMemcpyDefault cudaMemcpyDefault
#define hipMemset cudaMemset
#define hipGetLastError cudaGetLastError
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetDevice cudaGetDevice
#define hipSetDevice cudaSetDevice
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipDeviceProp_t cudaDeviceProp
#define hipMallocManaged cudaMallocManaged
#define hipMemAttachGlobal cudaMemAttachGlobal
#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...) \
  kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)
#endif

#define SALTATLAS_HIP_CHECK(call)                                       \
  do {                                                                  \
    hipError_t err = (call);                                            \
    if (err != hipSuccess) {                                            \
      std::fprintf(stderr, "GPU error %s:%d: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err));                             \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace saltatlas::solanet::apu_nn {

// Native SIMT width of the target backend: 64-lane wavefronts on AMD
// (MI300A), 32-lane warps on NVIDIA.
#if defined(__CUDACC__)
inline constexpr int k_native_warp_size = 32;
#else
inline constexpr int k_native_warp_size = 64;
#endif

SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_global_thread_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_global_thread_count() {
  return gridDim.x * blockDim.x;
}

template <int kWarpSize = k_native_warp_size>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_global_warp_id() {
  return get_global_thread_id() / kWarpSize;
}

template <int kWarpSize = k_native_warp_size>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_local_warp_id() {
  return threadIdx.x / kWarpSize;
}

template <int kWarpSize = k_native_warp_size>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int get_lane_id() {
  if constexpr (kWarpSize == 32) {
    return threadIdx.x & 31;
  } else if constexpr (kWarpSize == 64) {
    return threadIdx.x & 63;
  }
  return threadIdx.x % kWarpSize;
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
#if defined(__CUDA_ARCH__)
  __syncwarp();
#elif defined(__HIPCC__)
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

#if defined(__CUDACC__)
using rnd_state_type = curandStateXORWOW_t;

SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE void rnd_init(const int       tid,
                                                       const uint64_t  seed,
                                                       rnd_state_type& state) {
#if defined(__CUDA_ARCH__)
  curand_init(seed, tid, 0, &state);
#else
  (void)tid;
  (void)seed;
  (void)state;
#endif
}

SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE unsigned int rnd_next(
    rnd_state_type& state) {
#if defined(__CUDA_ARCH__)
  return curand(&state);
#else
  (void)state;
  return 0u;
#endif
}
#else
using rnd_state_type = rocrand_state_xorwow;

SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE void rnd_init(const int       tid,
                                                       const uint64_t  seed,
                                                       rnd_state_type& state) {
  rocrand_init(seed, tid, 0, &state);
}

SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE auto rnd_next(rnd_state_type& state) {
  return rocrand(&state);
}
#endif

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

// Note on CUDA masks: teams (sub-warp groups) in the same warp can diverge
// (e.g., different loop trip counts per team), so the full-warp constant mask
// would be illegal. __activemask() names exactly the lanes executing the
// intrinsic; exchanges stay within a team (width), whose lanes never diverge
// at these call sites. This matches HIP's implicit active-lane semantics.
template <typename T>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE T shfl_up(T v, int delta) {
#if defined(__CUDACC__)
  return __shfl_up_sync(__activemask(), v, delta);
#else
  // HIP has __shfl_up for int/float; for other types, specialize as needed.
  return __shfl_up(v, delta);
#endif
}

// Warp shuffle-down with an explicit width (sub-warp / team reductions).
// CUDA requires the *_sync variants; HIP keeps the classic intrinsics.
template <typename T>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE T
shfl_down(T v, int delta, int width = k_native_warp_size) {
#if defined(__CUDACC__)
  return __shfl_down_sync(__activemask(), v, delta, width);
#else
  return __shfl_down(v, delta, width);
#endif
}

// Warp broadcast from src_lane with an explicit width.
template <typename T>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE T
shfl_bcast(T v, int src_lane, int width = k_native_warp_size) {
#if defined(__CUDACC__)
  return __shfl_sync(__activemask(), v, src_lane, width);
#else
  return __shfl(v, src_lane, width);
#endif
}

// 64-bit atomic add usable on both backends. CUDA's atomicAdd has no
// size_t/unsigned long overload, only unsigned long long.
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE size_t
atomic_add_u64(size_t* address, size_t val) {
  static_assert(sizeof(size_t) == sizeof(unsigned long long),
                "size_t must be 64-bit");
  return static_cast<size_t>(
      atomicAdd(reinterpret_cast<unsigned long long*>(address),
                static_cast<unsigned long long>(val)));
}

// Warp-exclusive scan for int over the native warp width.
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int warp_exclusive_scan_int_64(
    int x) {
  int lane = threadIdx.x & (k_native_warp_size - 1);
  int sum  = x;  // inclusive scan first
#pragma unroll
  for (int d = 1; d < k_native_warp_size; d <<= 1) {
    int y = shfl_up(sum, d);
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
