// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>

#include "saltatlas/dnnd/detail/utilities/time.hpp"

namespace saltatlas::solanet {

using saltatlas::dndetail::launch_timer;

inline double get_elapsed_sec(
    const std::chrono::high_resolution_clock::time_point& start) {
  return saltatlas::dndetail::elapsed_time_sec(start);
}

}  // namespace saltatlas::solanet
