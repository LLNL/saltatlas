// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <cmath>

namespace saltatlas::dndetail {

template<typename T>
static constexpr double k_epsilon = std::numeric_limits<T>::epsilon();

template<typename T>
bool nearly_equal(const T a, const T b) {
  return (std::fabs(a - b) < k_epsilon<T>);
}

}  // namespace saltatlas::dndetail