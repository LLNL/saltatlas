// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#define SALTATLAS_SOLANET_APU_NND_SQL2
#ifdef SALTATLAS_SOLANET_APU_NND_SQL2
#ifndef NDEBUG
#warning "L2 distance function returns squared L2 distance."
#endif
#endif

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#if !defined(__CUDACC__)
#include <hip/hip_runtime.h>
#include <rocprim/block/block_reduce.hpp>
#endif

#include "saltatlas/solanet/detail/apu_nn/utils.hpp"

namespace saltatlas::solanet::apu_nn {

template <typename T>
using acc_type_t = std::conditional_t<std::is_same_v<T, double>, double, float>;

namespace {
template <typename T>
__host__ __device__ inline acc_type_t<T> l2_simd(const T* a, const T* b,
                                                 const size_t dims) {
  acc_type_t<T> sum = acc_type_t<T>(0);
  size_t        i   = 0;
  // Process 4 elements at a time
  for (; i + 4 <= dims; i += 4) {
    const acc_type_t<T> diff0 =
        static_cast<acc_type_t<T>>(a[i]) - static_cast<acc_type_t<T>>(b[i]);
    const acc_type_t<T> diff1 = static_cast<acc_type_t<T>>(a[i + 1]) -
                                static_cast<acc_type_t<T>>(b[i + 1]);
    const acc_type_t<T> diff2 = static_cast<acc_type_t<T>>(a[i + 2]) -
                                static_cast<acc_type_t<T>>(b[i + 2]);
    const acc_type_t<T> diff3 = static_cast<acc_type_t<T>>(a[i + 3]) -
                                static_cast<acc_type_t<T>>(b[i + 3]);
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }
  // Process remaining elements
  for (; i < dims; ++i) {
    const acc_type_t<T> diff =
        static_cast<acc_type_t<T>>(a[i]) - static_cast<acc_type_t<T>>(b[i]);
    sum += diff * diff;
  }
#ifdef SALTATLAS_SOLANET_APU_NND_SQL2
  return sum;
#else
  return static_cast<acc_type_t<T>>(::sqrt(sum));
#endif
}

template <typename T>
__host__ __device__ inline acc_type_t<T> alt_cosine_simd(const T* a, const T* b,
                                                         const size_t dims) {
  using acc_t = acc_type_t<T>;
  acc_t  n0   = acc_t(0);
  acc_t  n1   = acc_t(0);
  acc_t  dot  = acc_t(0);
  size_t i    = 0;

  for (; i + 4 <= dims; i += 4) {
    const acc_t a0 = static_cast<acc_t>(a[i]);
    const acc_t a1 = static_cast<acc_t>(a[i + 1]);
    const acc_t a2 = static_cast<acc_t>(a[i + 2]);
    const acc_t a3 = static_cast<acc_t>(a[i + 3]);
    const acc_t b0 = static_cast<acc_t>(b[i]);
    const acc_t b1 = static_cast<acc_t>(b[i + 1]);
    const acc_t b2 = static_cast<acc_t>(b[i + 2]);
    const acc_t b3 = static_cast<acc_t>(b[i + 3]);

    n0 += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
    n1 += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
    dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
  }

  for (; i < dims; ++i) {
    const acc_t va = static_cast<acc_t>(a[i]);
    const acc_t vb = static_cast<acc_t>(b[i]);
    n0 += va * va;
    n1 += vb * vb;
    dot += va * vb;
  }

  if (nearly_equal<acc_t>(n0, acc_t(0)) && nearly_equal<acc_t>(n1, acc_t(0)))
    return acc_t(0);
  else if (nearly_equal<acc_t>(n0, acc_t(0)) ||
           nearly_equal<acc_t>(n1, acc_t(0)))
    return std::numeric_limits<acc_t>::max() / acc_t(2);

  if (dot < acc_t(0) || nearly_equal<acc_t>(dot, acc_t(0))) {
    return std::numeric_limits<acc_t>::max() / acc_t(2);
  }

  const acc_t val = static_cast<acc_t>(::log2(::sqrt(n0 * n1) / dot));
  if (val < acc_t(0)) return acc_t(0);
  return val;
}

template <typename T>
__host__ __device__ inline acc_type_t<T> inner_product_simd(const T*     a,
                                                            const T*     b,
                                                            const size_t dims) {
  using acc_t = acc_type_t<T>;
  acc_t  dot  = acc_t(0);
  size_t i    = 0;
  for (; i + 4 <= dims; i += 4) {
    const acc_t a0 = static_cast<acc_t>(a[i]);
    const acc_t a1 = static_cast<acc_t>(a[i + 1]);
    const acc_t a2 = static_cast<acc_t>(a[i + 2]);
    const acc_t a3 = static_cast<acc_t>(a[i + 3]);
    const acc_t b0 = static_cast<acc_t>(b[i]);
    const acc_t b1 = static_cast<acc_t>(b[i + 1]);
    const acc_t b2 = static_cast<acc_t>(b[i + 2]);
    const acc_t b3 = static_cast<acc_t>(b[i + 3]);
    dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
  }
  for (; i < dims; ++i) {
    dot += static_cast<acc_t>(a[i]) * static_cast<acc_t>(b[i]);
  }
  // Keep IP distance as -dot so "smaller is better" everywhere in this
  // pipeline. Changing this to 1-dot or dot breaks cross-stage comparability.
  return -dot;
}
}  // namespace

template <typename T>
__host__ __device__ inline acc_type_t<T> l2(const T* a, const T* b,
                                            const size_t dims) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  return l2_simd(a, b, dims);
#else
  acc_type_t<T> sum = acc_type_t<T>(0);
  for (size_t i = 0; i < dims; ++i) {
    const acc_type_t<T> diff =
        static_cast<acc_type_t<T>>(a[i]) - static_cast<acc_type_t<T>>(b[i]);
    sum += diff * diff;
  }
  return static_cast<acc_type_t<T>>(::sqrt(sum));
#endif
}

template <typename T>
__host__ __device__ inline acc_type_t<T> alt_cosine(const T* a, const T* b,
                                                    const size_t dims) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  return alt_cosine_simd(a, b, dims);
#else
  using acc_t = acc_type_t<T>;
  // compute norms
  acc_t n0  = acc_t(0);
  acc_t n1  = acc_t(0);
  acc_t dot = acc_t(0);
  for (size_t i = 0; i < dims; ++i) {
    const acc_t va = static_cast<acc_t>(a[i]);
    const acc_t vb = static_cast<acc_t>(b[i]);
    n0 += va * va;
    n1 += vb * vb;
    dot += va * vb;
  }

  if (nearly_equal<acc_t>(n0, acc_t(0)) && nearly_equal<acc_t>(n1, acc_t(0)))
    return acc_t(0);
  else if (nearly_equal<acc_t>(n0, acc_t(0)) ||
           nearly_equal<acc_t>(n1, acc_t(0)))
    return std::numeric_limits<acc_t>::max() / acc_t(2);

  if (dot < acc_t(0) || nearly_equal<acc_t>(dot, acc_t(0))) {
    return std::numeric_limits<acc_t>::max() / acc_t(2);
  }

  const acc_t val = static_cast<acc_t>(::log2(::sqrt(n0 * n1) / dot));
  if (val < acc_t(0)) return acc_t(0);
  return val;
#endif
}

template <typename T>
__host__ __device__ inline acc_type_t<T> inner_product(const T* a, const T* b,
                                                       const size_t dims) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  return inner_product_simd(a, b, dims);
#else
  acc_type_t<T> dot = acc_type_t<T>(0);
  for (size_t i = 0; i < dims; ++i) {
    dot += static_cast<acc_type_t<T>>(a[i]) * static_cast<acc_type_t<T>>(b[i]);
  }
  // Keep the same min-close convention as the device path.
  return -dot;
#endif
}

// alt-cosine distance using multiple threads in a team parallel
// TEAM_SIZE: #of threads in a team
// All threads are in the same block, and team threads are contiguous.
// Warning : only the first lane in the team returns the correct distance, and
// other lanes may return intermediate values.
template <typename T, int TEAM_SIZE = 8>
__host__ __device__ inline acc_type_t<T> alt_cosine_team(const T* a, const T* b,
                                                         const size_t dims) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  static_assert(TEAM_SIZE > 0, "TEAM_SIZE must be positive.");
  static_assert((TEAM_SIZE & (TEAM_SIZE - 1)) == 0,
                "TEAM_SIZE must be a power of two.");
  using acc_t    = acc_type_t<T>;
  const int lane = threadIdx.x & (TEAM_SIZE - 1);
  acc_t     n0   = acc_t(0);
  acc_t     n1   = acc_t(0);
  acc_t     dot  = acc_t(0);
  for (size_t i = static_cast<size_t>(lane); i < dims; i += TEAM_SIZE) {
    const acc_t va = static_cast<acc_t>(a[i]);
    const acc_t vb = static_cast<acc_t>(b[i]);
    n0 += va * va;
    n1 += vb * vb;
    dot += va * vb;
  }
#pragma unroll
  for (int offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
    n0 += shfl_down(n0, offset, TEAM_SIZE);
    n1 += shfl_down(n1, offset, TEAM_SIZE);
    dot += shfl_down(dot, offset, TEAM_SIZE);
  }

  if (nearly_equal<acc_t>(n0, acc_t(0)) && nearly_equal<acc_t>(n1, acc_t(0)))
    return acc_t(0);
  else if (nearly_equal<acc_t>(n0, acc_t(0)) ||
           nearly_equal<acc_t>(n1, acc_t(0)))
    return std::numeric_limits<acc_t>::max() / acc_t(2);

  if (dot < acc_t(0) || nearly_equal<acc_t>(dot, acc_t(0))) {
    return std::numeric_limits<acc_t>::max() / acc_t(2);
  }

  const acc_t val = static_cast<acc_t>(::log2(::sqrt(n0 * n1) / dot));
  if (val < acc_t(0)) return acc_t(0);
  return val;
#else
  return alt_cosine(a, b, dims);
#endif
}

// squared-l2 distance using multiple threads in a team parallel
// TEAM_SIZE: #of threads in a team
// All threads are in the same block, and team threads are contiguous.
// Warning : only the first lane in the team returns the correct distance, and
// other lanes may return intermediate values.
template <typename T, int TEAM_SIZE = 8>
__host__ __device__ inline acc_type_t<T> l2_team(const T* a, const T* b,
                                                 const size_t dims) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  static_assert(TEAM_SIZE > 0, "TEAM_SIZE must be positive.");
  static_assert((TEAM_SIZE & (TEAM_SIZE - 1)) == 0,
                "TEAM_SIZE must be a power of two.");
  const int     lane = threadIdx.x & (TEAM_SIZE - 1);
  acc_type_t<T> sum  = acc_type_t<T>(0);
  for (size_t i = static_cast<size_t>(lane); i < dims; i += TEAM_SIZE) {
    const acc_type_t<T> diff =
        static_cast<acc_type_t<T>>(a[i]) - static_cast<acc_type_t<T>>(b[i]);
    sum += diff * diff;
  }
#pragma unroll
  for (int offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
    sum += shfl_down(sum, offset, TEAM_SIZE);
  }
#else
  acc_type_t<T> sum = acc_type_t<T>(0);
  for (size_t i = 0; i < dims; ++i) {
    const acc_type_t<T> diff =
        static_cast<acc_type_t<T>>(a[i]) - static_cast<acc_type_t<T>>(b[i]);
    sum += diff * diff;
  }
#endif

#ifdef SALTATLAS_SOLANET_APU_NND_SQL2
  return sum;
#else
  return std::sqrt(sum);
#endif
}

// inner-product distance using multiple threads in a team parallel
// TEAM_SIZE: #of threads in a team
// All threads are in the same block, and team threads are contiguous.
// Warning : only the first lane in the team returns the correct distance, and
// other lanes may return intermediate values.
template <typename T, int TEAM_SIZE = 8>
__host__ __device__ inline acc_type_t<T> inner_product_team(const T*     a,
                                                            const T*     b,
                                                            const size_t dims) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  static_assert(TEAM_SIZE > 0, "TEAM_SIZE must be positive.");
  static_assert((TEAM_SIZE & (TEAM_SIZE - 1)) == 0,
                "TEAM_SIZE must be a power of two.");
  using acc_t    = acc_type_t<T>;
  const int lane = threadIdx.x & (TEAM_SIZE - 1);
  acc_t     dot  = acc_t(0);
  for (size_t i = static_cast<size_t>(lane); i < dims; i += TEAM_SIZE) {
    dot += static_cast<acc_t>(a[i]) * static_cast<acc_t>(b[i]);
  }
#pragma unroll
  for (int offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
    dot += shfl_down(dot, offset, TEAM_SIZE);
  }
  // Keep the same min-close convention as other IP distance paths.
  return -dot;
#else
  return inner_product(a, b, dims);
#endif
}

}  // namespace saltatlas::solanet::apu_nn
