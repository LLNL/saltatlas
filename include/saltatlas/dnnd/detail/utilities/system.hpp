// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#if __has_include(<metall/detail/memory.hpp>)
#include <metall/detail/memory.hpp>
#else
#error "Metall is required"
#endif

namespace saltatlas::dndetail {

using metall::mtlldetail::get_free_ram_size;
using metall::mtlldetail::get_page_cache_size;
using metall::mtlldetail::get_total_ram_size;
using metall::mtlldetail::get_used_ram_size;
using metall::mtlldetail::read_meminfo;

/// \brief Returns the size of the 'MemAvailable' ram size from /proc/meminfo
/// \return On success, returns the 'MemAvailable' ram size. On error, returns
/// -1.
inline ssize_t get_available_ram_size() {
  const ssize_t mem_available = read_meminfo("MemAvailable:");
  if (mem_available == -1) {
    return -1;  // something wrong;
  }
  return static_cast<ssize_t>(mem_available);
}

}  // namespace saltatlas::dndetail