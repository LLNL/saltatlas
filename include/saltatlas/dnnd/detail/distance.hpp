// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <string_view>

#include <saltatlas/dnnd/detail/utilities/blas.hpp>

namespace saltatlas::dndetail::distance {
template <typename feature_element_type, typename distance_type>
using metric_type = distance_type(const std::size_t,
                                  const feature_element_type *const,
                                  const feature_element_type *const);

template <typename feature_element_type, typename distance_type>
inline distance_type invalid(const std::size_t,
                             const feature_element_type *const,
                             const feature_element_type *const) {
  assert(false);
  return distance_type{};
}

template <typename feature_element_type, typename distance_type>
inline distance_type l2(const std::size_t                 len,
                        const feature_element_type *const f0,
                        const feature_element_type *const f1) {
  distance_type d = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const auto x = (f0[i] - f1[i]);
    d += x * x;
  }
  return static_cast<distance_type>(std::sqrt(d));
}

template <typename feature_element_type, typename distance_type>
inline distance_type cosine(const std::size_t                 len,
                            const feature_element_type *const f0,
                            const feature_element_type *const f1) {
  const distance_type n0 =
      std::sqrt(dndetail::blas::inner_product(len, f0, f0));
  const distance_type n1 =
      std::sqrt(dndetail::blas::inner_product(len, f1, f1));
  if (n0 == 0 && n1 == 0)
    return static_cast<distance_type>(0);
  else if (n0 == 0 || n1 == 0)
    return static_cast<distance_type>(1);

  const distance_type x = dndetail::blas::inner_product(len, f0, f1);
  return static_cast<distance_type>(1.0 - x / (n0 * n1));
}

template <typename feature_element_type, typename distance_type>
inline distance_type jaccard_index(const std::size_t                 len,
                                   const feature_element_type *const f0,
                                   const feature_element_type *const f1) {
  std::size_t num_non_zero = 0;
  std::size_t num_equal    = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const bool x_true = !!f0[i];
    const bool y_true = !!f1[i];
    num_non_zero += x_true | y_true;
    num_equal += x_true & y_true;
  }

  if (num_non_zero == 0)
    return distance_type{0};
  else
    return static_cast<distance_type>(num_non_zero - num_equal) /
           static_cast<distance_type>(num_non_zero);
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

template <typename feature_element_type, typename distance_type>
inline metric_type<feature_element_type, distance_type> &metric(
    const metric_id &id) {
  if (id == metric_id::l2) {
    return l2<feature_element_type, distance_type>;
  } else if (id == metric_id::cosine) {
    return cosine<feature_element_type, distance_type>;
  } else if (id == metric_id::jaccard) {
    return jaccard_index<feature_element_type, distance_type>;
  }
  return invalid<feature_element_type, distance_type>;
}

template <typename feature_element_type, typename distance_type>
inline metric_type<feature_element_type, distance_type> &metric(
    const std::string_view metric_name) {
  return metric<feature_element_type, distance_type>(
      convert_to_metric_id(metric_name));
}
}  // namespace saltatlas::dndetail::distance