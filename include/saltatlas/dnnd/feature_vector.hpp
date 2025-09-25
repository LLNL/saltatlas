// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <ostream>

#include <boost/container/vector.hpp>
#include <ygm/utility/boost_vector.hpp>

#if __has_include(<metall/metall.hpp>)
#include <metall/metall.hpp>
#endif

#include "saltatlas/dnnd/detail/utilities/allocator.hpp"

namespace saltatlas {

/// \brief Feature vector type.
template <typename Element, typename Allocator = std::allocator<Element>>
using feature_vector = boost::container::vector<Element, Allocator>;

#if __has_include(<metall/metall.hpp>)
/// \brief Feature vector type with persistent memory support.
template <typename Element,
          typename Allocator = metall::manager::fallback_allocator<Element>>
using pm_feature_vector = boost::container::vector<Element, Allocator>;
#endif

// to_string for feature_vector and pm_feature_vector
template <typename Element, typename Allocator>
inline std::string to_string(
    const boost::container::vector<Element, Allocator> &v) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    oss << v[i];
    if (i + 1 < v.size()) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}
}  // namespace saltatlas
