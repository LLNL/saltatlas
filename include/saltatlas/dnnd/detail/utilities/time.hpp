// Copyright 2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>

namespace saltatlas::dndetail {

inline std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}

inline double elapsed_time_sec(
    const std::chrono::high_resolution_clock::time_point &start) {
  const auto duration_time = std::chrono::high_resolution_clock::now() - start;
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(
                 duration_time)
                 .count()) /
         1e6;
}

}  // namespace saltatlas::dndetail