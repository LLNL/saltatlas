// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <string_view>

#include <saltatlas/dnnd/detail/utilities/blas.hpp>

namespace saltatlas::distance {
template <typename T>
using metric_type = T(const std::size_t, const T *const, const T *const);

template <typename T>
inline auto invalid(const std::size_t, const T *const, const T *const) {
  assert(false);
  return T{};
}

template <typename T>
inline auto l2(const std::size_t len, const T *const f0, const T *const f1) {
  T d = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const auto x = (f0[i] - f1[i]);
    d += x * x;
  }
  return static_cast<T>(std::sqrt(d));
}

template <typename T>
inline auto cosine(const std::size_t len, const T *const f0,
                   const T *const f1) {
  const T n0 = std::sqrt(dndetail::blas::inner_product(len, f0, f0));
  const T n1 = std::sqrt(dndetail::blas::inner_product(len, f1, f1));
  if (n0 == 0 && n1 == 0)
    return static_cast<T>(0);
  else if (n0 == 0 || n1 == 0)
    return static_cast<T>(1);

  const T x = dndetail::blas::inner_product(len, f0, f1);
  return static_cast<T>(1.0 - x / (n0 * n1));
}

template <typename T>
inline auto jaccard_index(const std::size_t len, const T *const f0,
                          const T *const f1) {
  std::size_t num_non_zero = 0;
  std::size_t num_equal    = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const bool x_true = !!f0[i];
    const bool y_true = !!f1[i];
    num_non_zero += x_true | y_true;
    num_equal += x_true & y_true;
  }

  if (num_non_zero == 0)
    return T{0};
  else
    return static_cast<T>(num_non_zero - num_equal) /
           static_cast<T>(num_non_zero);
}

enum class metric_id : uint8_t { invalid, l2, cosine, jaccard };

inline metric_id convert_to_metric_id(const std::string_view &metric_name) {
  if (metric_name == "l2") {
    return metric_id::l2;
  } else if (metric_name == "cosine") {
    return metric_id::cosine;
  } else if (metric_name == "jaccard") {
    return metric_id::jaccard;
  }
  return metric_id::invalid;
}

template <typename T>
inline metric_type<T> &metric(const metric_id &id) {
  if (id == metric_id::l2) {
    return l2<T>;
  } else if (id == metric_id::cosine) {
    return cosine<T>;
  } else if (id == metric_id::jaccard) {
    return jaccard_index<T>;
  }
  return invalid<T>;
}

template <typename T>
inline metric_type<T> &metric(const std::string_view metric_name) {
  return metric<T>(convert_to_metric_id(metric_name));
}
}  // namespace saltatlas::distance