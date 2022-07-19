// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>

#ifdef ENABLE_AVX
#include <immintrin.h>  // AVX
#endif

namespace saltatlas::dndetail::blas {
template <typename value_type>
inline value_type inner_product(const std::size_t       size,
                                const value_type *const lhd,
                                const value_type *const rhd) {
  value_type x = 0;
  for (std::size_t i = 0; i < size; ++i) {
    x += lhd[i] * rhd[i];
  }
  return x;
}

#ifdef ENABLE_AVX
inline void avx_mac(const std::size_t length, const double scalar,
                    const double *const aligned_b, double *const aligned_c) {
  assert(length % 4 == 0);

  alignas(256) const double aligned_scalar = scalar;
  const __m256d             avx_scalar     = _mm256_set1_pd(aligned_scalar);
  for (std::size_t i = 0; i < length; i += 4) {
    __m256d avx_b = _mm256_load_pd(&aligned_b[i]);
    __m256d tmp   = _mm256_mul_pd(avx_scalar, avx_b);
    __m256d avx_c = _mm256_load_pd(&aligned_c[i]);
    avx_c         = _mm256_add_pd(tmp, avx_c);
    // avx_c = _mm256_fmadd_pd(avx_scalar, avx_b, avx_c); // compile error !?
    _mm256_store_pd(&aligned_c[i], avx_c);
  }
}
#endif

/// \brief multiply–accumulate operation
template <typename value_type>
inline void mac(const std::size_t size, const value_type a,
                const value_type *const in_vec, value_type *const out_vec) {
  for (std::size_t i = 0; i < size; ++i) {
    out_vec[i] += a * in_vec[i];
  }
}
}  // namespace saltatlas::dndetail::blas