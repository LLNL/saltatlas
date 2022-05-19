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
template <typename feature_vector_type>
using metric_type = std::remove_cv_t<typename feature_vector_type::value_type>(
    const feature_vector_type &, const feature_vector_type &);

template <typename feature_vector_type>
inline auto invalid(const feature_vector_type &, const feature_vector_type &) {
  using element_type = typename feature_vector_type::value_type;
  assert(false);
  return element_type{};
}

template <typename feature_vector_type>
inline auto l2(const feature_vector_type &f0, const feature_vector_type &f1) {
  using element_type = typename feature_vector_type::value_type;
  assert(f0.size() == f1.size());
  const std::size_t len = f0.size();
  element_type      d   = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const auto x = (f0[i] - f1[i]);
    d += x * x;
  }
  return static_cast<element_type>(std::sqrt(d));
}

template <typename feature_vector_type>
inline auto cosine(const feature_vector_type &f0,
                   const feature_vector_type &f1) {
  assert(f0.size() == f1.size());
  using element_type = typename feature_vector_type::value_type;

  const element_type n0 =
      std::sqrt(dndetail::blas::inner_product(f0.size(), f0.data(), f0.data()));
  const element_type n1 =
      std::sqrt(dndetail::blas::inner_product(f1.size(), f1.data(), f1.data()));
  if (n0 == 0 && n1 == 0)
    return static_cast<element_type>(0);
  else if (n0 == 0 || n1 == 0)
    return static_cast<element_type>(1);

  const element_type x =
      dndetail::blas::inner_product(f0.size(), f0.data(), f1.data());
  return static_cast<element_type>(1.0 - x / (n0 * n1));
}

template <typename feature_vector_type>
inline auto jaccard_index(const feature_vector_type &f0,
                          const feature_vector_type &f1) {
  using element_type = typename feature_vector_type::value_type;
  assert(f0.size() == f1.size());
  std::size_t num_non_zero = 0;
  std::size_t num_equal    = 0;
  for (std::size_t i = 0; i < f0.size(); ++i) {
    const bool x_true = !!f0[i];
    const bool y_true = !!f1[i];
    num_non_zero += x_true | y_true;
    num_equal += x_true & y_true;
  }

  if (num_non_zero == 0)
    return element_type{0};
  else
    return static_cast<element_type>(num_non_zero - num_equal) /
           static_cast<element_type>(num_non_zero);
}

template <typename feature_vector_type>
inline const metric_type<feature_vector_type> &metric(
    const std::string_view &metric_name) {
  if (metric_name == "l2")
    return l2<feature_vector_type>;
  else if (metric_name == "cosine")
    return cosine<feature_vector_type>;
  else if (metric_name == "jaccard")
    return jaccard_index<feature_vector_type>;

  return invalid<feature_vector_type>;
}
}  // namespace saltatlas::distance