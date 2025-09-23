// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef SALTATLAS_USE_NUMA
#ifndef __APPLE__
#include <numa.h>
#else
#warning "Does not use NUMA"
#endif
#endif

namespace saltatlas::dndetail::numa {

bool available() noexcept {
#ifdef SALTATLAS_USE_NUMA
  return ::numa_available() != -1;
#else
  return false;
#endif
}

int get_num_avail_nodes() noexcept {
#ifdef SALTATLAS_USE_NUMA
  return ::numa_max_node() + 1;
#else
  return 1;
#endif
}

}  // namespace saltatlas::dndetail::numa