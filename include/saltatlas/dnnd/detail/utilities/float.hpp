// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <limits>

namespace saltatlas::dndetail {

template <typename T>
bool nearly_equal(const T a, const T b,
                  const double eps = std::numeric_limits<T>::epsilon()) {
  return (std::fabs(a - b) < eps);
}

}  // namespace saltatlas::dndetail