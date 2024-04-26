// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <string_view>

#include <saltatlas/dnnd/detail/utilities/blas.hpp>
#include <saltatlas/dnnd/detail/utilities/float.hpp>

namespace saltatlas::distance {

using saltatlas::dndetail::nearly_equal;

enum class id : uint8_t {
  invalid,
  l1,
  l2,
  sql2,
  cosine,
  altcosine,
  jaccard,
  altjaccard,
  levenshtein,
  custom  // User-defined distance function
};

template <typename point_type, typename distance_type>
using distance_function_type =
    std::function<distance_type(const point_type &, const point_type &)>;

template <typename point_type, typename distance_type>
inline distance_type invalid(const point_type &, const point_type &) {
  throw std::runtime_error("Invalid distance function.");
  return distance_type{};
}

template <typename point_type, typename distance_type>
inline distance_type l1(const point_type &p0, const point_type &p1) {
  assert(p0.size() == p1.size());
  distance_type d = 0;
  for (std::size_t i = 0; i < p0.size(); ++i) {
    const auto x = std::abs(p0[i] - p1[i]);
    d += x;
  }
  return static_cast<distance_type>(std::sqrt(d));
}

template <typename point_type, typename distance_type>
inline distance_type l2(const point_type &p0, const point_type &p1) {
  assert(p0.size() == p1.size());
  distance_type d = 0;
  for (std::size_t i = 0; i < p0.size(); ++i) {
    const auto x = (p0[i] - p1[i]);
    d += x * x;
  }
  return static_cast<distance_type>(std::sqrt(d));
}

/// \brief Squared Euclidean distance, which omits the final square root in the
/// calculation of l2 norm.
template <typename point_type, typename distance_type>
inline distance_type sql2(const point_type &p0, const point_type &p1) {
  assert(p0.size() == p1.size());
  distance_type d = 0;
  for (std::size_t i = 0; i < p0.size(); ++i) {
    const auto x = (p0[i] - p1[i]);
    d += x * x;
  }
  return d;
}

template <typename point_type, typename distance_type>
inline distance_type cosine(const point_type &p0, const point_type &p1) {
  assert(p0.size() == p1.size());
  const distance_type n0 =
      dndetail::blas::inner_product(p0.size(), p0.data(), p0.data());
  const distance_type n1 =
      dndetail::blas::inner_product(p1.size(), p1.data(), p1.data());
  if (nearly_equal(n0, distance_type(0)) && nearly_equal(n1, distance_type(0)))
    return static_cast<distance_type>(0);
  else if (nearly_equal(n0, distance_type(0)) ||
           nearly_equal(n1, distance_type(0)))
    return static_cast<distance_type>(1);

  const distance_type x =
      dndetail::blas::inner_product(p0.size(), p0.data(), p1.data());
  return static_cast<distance_type>(1.0 - x / std::sqrt(n0 * n1));
}

/// \brief Alternative cosine distance. The original model is from PyNNDescent.
/// This function returns the same relative distance orders as the normal
/// cosine similarity.
template <typename point_type, typename distance_type>
inline distance_type alt_cosine(const point_type &p0, const point_type &p1) {
  assert(p0.size() == p1.size());
  const distance_type n0 =
      dndetail::blas::inner_product(p0.size(), p0.data(), p0.data());
  const distance_type n1 =
      dndetail::blas::inner_product(p1.size(), p1.data(), p1.data());
  if (nearly_equal(n0, distance_type(0)) &&
      nearly_equal(n1, distance_type(0))) {
    return static_cast<distance_type>(0);
  } else if (nearly_equal(n0, distance_type(0)) ||
             nearly_equal(n1, distance_type(0))) {
    // Does not return the max value to prevent overflow on the caller side.
    return std::numeric_limits<distance_type>::max() / distance_type(2);
  }

  const distance_type x =
      dndetail::blas::inner_product(p0.size(), p0.data(), p1.data());
  if (x < 0 || nearly_equal(x, distance_type(0))) {
    return std::numeric_limits<distance_type>::max() / distance_type(2);
  }

  return static_cast<distance_type>(std::log2(std::sqrt(n0 * n1) / x));
}

template <typename point_type, typename distance_type>
inline distance_type jaccard_index(const point_type &p0, const point_type &p1) {
  assert(p0.size() == p1.size());
  std::size_t num_non_zero = 0;
  std::size_t num_equal    = 0;
  for (std::size_t i = 0; i < p0.size(); ++i) {
    const bool x_true = !!p0[i];
    const bool y_true = !!p1[i];
    num_non_zero += x_true | y_true;
    num_equal += x_true & y_true;
  }

  if (num_non_zero == 0)
    return distance_type{0};
  else
    return static_cast<distance_type>(num_non_zero - num_equal) /
           static_cast<distance_type>(num_non_zero);
}

/// \brief Alternative Jaccard index. The original model is from PyNNDescent.
/// This function returns the same relative distance orders as the normal
/// Jaccard index.
template <typename point_type, typename distance_type>
inline distance_type alt_jaccard_index(const point_type &p0,
                                       const point_type &p1) {
  assert(p0.size() == p1.size());
  std::size_t num_non_zero = 0;
  std::size_t num_equal    = 0;
  for (std::size_t i = 0; i < p0.size(); ++i) {
    const bool x_true = !!p0[i];
    const bool y_true = !!p1[i];
    num_non_zero += x_true | y_true;
    num_equal += x_true & y_true;
  }

  if (num_non_zero == 0)
    return distance_type{0};
  else
    return static_cast<distance_type>(
        -std::log2(distance_type(num_equal) / distance_type(num_non_zero)));
}

template <typename point_type, typename distance_type>
inline distance_type levenshtein(const point_type &p0, const point_type &p1) {
  const auto m = p0.size();
  const auto n = p1.size();
  if (m == 0) {
    return n;
  }
  if (n == 0) {
    return m;
  }

  // Row of matrix for dynamic programming approach
  std::vector<size_t> dist_row(m + 1);
  for (size_t i = 0; i < m + 1; ++i) {
    dist_row[i] = i;
  }

  for (size_t i = 1; i < n + 1; ++i) {
    size_t diag = i - 1;
    size_t next_diag;
    dist_row[0] = i;
    for (size_t j = 1; j < m + 1; ++j) {
      next_diag              = dist_row[j];
      bool substitution_cost = (p0[j - 1] != p1[i - 1]);

      dist_row[j] =
          std::min(1 + dist_row[j],
                   std::min(1 + dist_row[j - 1], substitution_cost + diag));
      diag = next_diag;
    }
  }

  return static_cast<distance_type>(dist_row[m]);
}

inline distance::id convert_to_distance_id(
    const std::string_view &distance_name) {
  if (distance_name == "l1") {
    return distance::id::l1;
  } else if (distance_name == "l2") {
    return distance::id::l2;
  } else if (distance_name == "sql2") {
    return distance::id::sql2;
  } else if (distance_name == "cosine") {
    return distance::id::cosine;
  } else if (distance_name == "altcosine") {
    return distance::id::altcosine;
  } else if (distance_name == "jaccard") {
    return distance::id::jaccard;
  } else if (distance_name == "altjaccard") {
    return distance::id::altjaccard;
  } else if (distance_name == "levenshtein") {
    return distance::id::levenshtein;
  }
  return distance::id::invalid;
}

inline std::string convert_to_distance_name(const distance::id &id) {
  switch (id) {
    case distance::id::l1:
      return "l1";
    case distance::id::l2:
      return "l2";
    case distance::id::sql2:
      return "sql2";
    case distance::id::cosine:
      return "cosine";
    case distance::id::altcosine:
      return "altcosine";
    case distance::id::jaccard:
      return "jaccard";
    case distance::id::altjaccard:
      return "altjaccard";
    case distance::id::levenshtein:
      return "levenshtein";
    case distance::id::custom:
      return "custom";
    default:
      return "invalid";
  }
}

template <typename point_type, typename distance_type>
inline distance_function_type<point_type, distance_type> distance_function(
    const distance::id &id) {
  if (id == distance::id::l1) {
    return l1<point_type, distance_type>;
  } else if (id == distance::id::l2) {
    return l2<point_type, distance_type>;
  } else if (id == distance::id::sql2) {
    return sql2<point_type, distance_type>;
  } else if (id == distance::id::cosine) {
    return cosine<point_type, distance_type>;
  } else if (id == distance::id::altcosine) {
    return alt_cosine<point_type, distance_type>;
  } else if (id == distance::id::jaccard) {
    return jaccard_index<point_type, distance_type>;
  } else if (id == distance::id::altjaccard) {
    return alt_jaccard_index<point_type, distance_type>;
  } else if (id == distance::id::levenshtein) {
    return levenshtein<point_type, distance_type>;
  }
  return invalid<point_type, distance_type>;
}

template <typename point_type, typename distance_type>
inline distance_function_type<point_type, distance_type> distance_function(
    const std::string_view distance_name) {
  return distance_function<point_type, distance_type>(
      convert_to_distance_id(distance_name));
}
}  // namespace saltatlas::distance
