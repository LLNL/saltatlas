// Copyright 2020-2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>

#include <saltatlas/common/detail/data_reader_kernel.hpp>

namespace saltatlas {

template <typename point_type>
inline bool read_query(const std::filesystem::path &query_file_path,
                       std::vector<point_type>     &queries) {
  return detail::read_query_kernel<point_type>(query_file_path, queries);
}

template <typename id_type, typename distance_type>
inline bool read_neighbors(
    const std::filesystem::path &file_path,
    std::vector<std::vector<detail::neighbor<id_type, distance_type>>> &store) {
  return detail::read_neighbors_kernel(file_path, store);
}

}  // namespace saltatlas