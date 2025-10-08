// Copyright 2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \file data_reader_kernel.hpp
/// Contains internal functions for data reading.
///
/// The main purpose of this file is to reuse the functions in MPI and non-MPI
/// (shared-memory) implementations.
/// This file must not depend on MPI or YGM.

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "saltatlas/common/detail/neighbor.hpp"
#include "saltatlas/common/detail/utilities/general.hpp"
#include "saltatlas/common/detail/utilities/string_cast.hpp"
#include "saltatlas/common/point_store.hpp"

namespace saltatlas::detail {

/// \brief Parse a feature vector from a string.
/// Each element is separated by whitespace.
/// \tparam feature_element_t Type of each element in the feature vector.
/// \param input Input string.
/// \return Parsed feature vector.
template <typename feature_element_t>
inline std::vector<feature_element_t> parse_feature_vector(
    const std::string &input) {
  return str_split<feature_element_t>(input);
};

/// \brief Parse a feature vector from a string.
/// Each element is separated by 'delimiter'.
/// \tparam feature_element_t Type of each element in the feature vector.
/// \param input Input string.
/// \param delimiter Delimiter character.
/// \return Parsed feature vector.
template <typename feature_element_t>
inline std::vector<feature_element_t> parse_feature_vector(
    const std::string &input, const char delimiter) {
  return str_split<feature_element_t>(input, delimiter);
};

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by whitespace.
/// \warning
/// This function uses static variables internally. Each process must call this
/// function only once at a time.
template <typename id_type, typename feature_element_t>
inline std::pair<id_type, std::vector<feature_element_t>>
parse_feature_vector_with_id(const std::string &input) {
  std::string       buf;
  std::stringstream ss(input);

  // Extract ID (first token)
  ss >> buf;
  id_type id;
  id = str_cast<id_type>(buf);

  // Extract point (remaining tokens)
  std::string point_str = ss.str().substr(ss.tellg());
  auto        fv        = parse_feature_vector<feature_element_t>(point_str);
  return std::make_pair(std::move(id), std::move(fv));
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by 'delimiter'.
template <typename id_type, typename feature_element_t>
inline std::pair<id_type, std::vector<feature_element_t>>
parse_feature_vector_with_id(const std::string &input, const char delimiter) {
  std::stringstream ss(input);
  std::string       buf;
  std::getline(ss, buf, delimiter);

  id_type id = str_cast<id_type>(buf);

  std::string point_str = ss.str().substr(ss.tellg());
  auto fv = parse_feature_vector<feature_element_t>(point_str, delimiter);
  return std::make_pair(std::move(id), std::move(fv));
}

/// \brief Read neighbors from a file.
/// A neighbor file is a text file and consists of two blocks:
/// ID block and distance block.
/// In ID block, each line is a list of IDs of neighbors of a point.
/// In distance block, each line is a list of distances of neighbors of a point.
/// i-th point's neighbors are stored in i-th line and distances are stored in
/// (i+N)-th line in the file, where N is the number of points in the file.
/// The number of lines in ID block and distance block must be the same.
/// The number of IDs in each line must be the same.
/// The number of distances in each line must be the same.
/// \tparam id_type ID type.
/// \tparam distance_type Distance type.
/// \param file_path Path to a neighbor file.
/// \param store Neighbor table instance.
template <typename id_type, typename distance_type>
inline bool read_neighbors_kernel(
    const std::filesystem::path                                &file_path,
    std::vector<std::vector<neighbor<id_type, distance_type>>> &store) {
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open: " << file_path << std::endl;
    return false;
  }

  std::size_t num_entries = 0;
  {
    std::size_t cnt_lines = 0;
    for (std::string buf; std::getline(ifs, buf);) {
      ++cnt_lines;
    }
    if (!ifs.eof() && (ifs.bad() || ifs.fail())) {
      std::cerr << "Failed reading data from " << file_path << std::endl;
      return false;
    }
    ifs.clear();
    ifs.seekg(0);

    if (cnt_lines % 2 != 0) {
      std::cerr << "#of lines in the file is not an even number: " << file_path
                << std::endl;
      return false;
    }
    num_entries = cnt_lines / 2;
  }
  store.reserve(num_entries);

  // Count #of neighbors per entry (line) by reading the first line.
  std::size_t num_neighbors_per_entry = 0;
  {
    std::string buf;
    std::getline(ifs, buf);
    if (!ifs.eof() && (ifs.bad() || ifs.fail())) {
      std::cerr << "Failed reading data from " << file_path << std::endl;
      return false;
    }
    ifs.clear();
    ifs.seekg(0);

    num_neighbors_per_entry = detail::str_split<id_type>(buf).size();
  }

  // Reads neighbor IDs.
  for (std::string buf; std::getline(ifs, buf);) {
    const auto ids = detail::str_split<id_type>(buf);
    std::vector<detail::neighbor<id_type, distance_type>> neighbors;
    for (const auto id : ids) {
      neighbors.emplace_back(id, distance_type{});
    }
    if (neighbors.size() != num_neighbors_per_entry) {
      std::cerr << "#of neighbors per line are not the same" << std::endl;
      return false;
    }
    store.push_back(std::move(neighbors));
    if (store.size() == num_entries) break;
  }
  if (store.size() != num_entries || ifs.bad() || ifs.fail()) {
    std::cerr << "Failed reading data from " << file_path << std::endl;
    return false;
  }

  // Reads distances.
  std::size_t line_no = 0;
  for (std::string buf; std::getline(ifs, buf);) {
    const auto distances = detail::str_split<distance_type>(buf);
    if (distances.size() != num_neighbors_per_entry) {
      std::cerr << "#of neighbors per line are not the same" << std::endl;
      return false;
    }

    std::size_t k = 0;
    for (const auto d : distances) {
      store[line_no][k++].distance = d;
    }

    ++line_no;
  }
  if (line_no != num_entries || (!ifs.eof() && (ifs.bad() || ifs.fail()))) {
    std::cerr << "Failed reading data from " << file_path << std::endl;
    return false;
  }
  return true;
}

/// \brief Reads a file that contain queries.
/// Each line is the feature vector of a query point.
/// Can read the white space separated format (without ID).
/// \tparam point_t Point type.
/// \param query_file_path Path to a query file.
/// \param queries Buffer to store read queries.
template <typename point_t>
inline bool read_query_kernel(
    const std::filesystem::path                &query_file_path,
    std::function<point_t(const std::string &)> parser,
    std::vector<point_t>                       &queries) {
  if (query_file_path.empty()) return true;

  std::ifstream ifs(query_file_path);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << query_file_path << std::endl;
    return false;
  }

  for (std::string line; std::getline(ifs, line);) {
    queries.push_back(parser(line));
  }
  return true;
}

/// \brief read_query function for reading feature vectors.
template <typename point_t>
inline bool read_query_kernel(const std::filesystem::path &query_file_path,
                              std::vector<point_t>        &queries) {
  return read_query_kernel<point_t>(
      query_file_path,
      [](const std::string &line) {
        auto data = detail::str_split<typename point_t::value_type>(line);
        return point_t(data.begin(), data.end());
      },
      queries);
}

}  // namespace saltatlas::detail