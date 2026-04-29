// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <saltatlas/dnnd/detail/utilities/omp.hpp>

template <typename T>
inline void l2_normalize_points(T* data, std::size_t num_rows,
                                std::size_t num_cols) noexcept {
  // static_assert(std::is_floating_point_v<T>,
  //               "normalize_points requires a floating-point element type.");

  if (data == nullptr || num_rows == 0 || num_cols == 0) {
    return;
  }

  OMP_DIRECTIVE(parallel for)
  for (std::size_t row = 0; row < num_rows; ++row) {
    T* row_ptr = data + row * num_cols;

    T norm_sq = T(0);
    for (std::size_t col = 0; col < num_cols; ++col) {
      const T v = row_ptr[col];
      norm_sq += v * v;
    }

    if (norm_sq == T(0)) {
      continue;
    }

    const T inv_norm = T(1) / static_cast<T>(std::sqrt(norm_sq));
    for (std::size_t col = 0; col < num_cols; ++col) {
      row_ptr[col] *= inv_norm;
    }
  }
}
