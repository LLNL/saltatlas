// Copyright 2020-2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "saltatlas/common/detail/data_reader_kernel.hpp"
#include "saltatlas/dnnd/detail/utilities/omp.hpp"
#include "saltatlas/shm_knng_query/dense_point_store.hpp"

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

namespace smqdetail {

inline std::size_t count_lines_in_file(const std::filesystem::path &file_path) {
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open: " << file_path << std::endl;
    return 0;
  }

  // Count #of lines.
  std::size_t cnt_lines = 0;
  for (std::string buf; std::getline(ifs, buf);) {
    ++cnt_lines;
  }
  if (!ifs.eof() && (ifs.bad() || ifs.fail())) {
    std::cerr << "Failed reading data from " << file_path << std::endl;
    return 0;
  }
  return cnt_lines;
}

inline std::vector<std::size_t> count_lines_in_files(
    const std::vector<std::filesystem::path> &point_file_paths) {
  std::vector<std::size_t> line_counts(point_file_paths.size(), 0);
  OMP_DIRECTIVE (parallel for)
  for (std::size_t i = 0; i < point_file_paths.size(); ++i) {
    line_counts[i] = count_lines_in_file(point_file_paths[i]);
  }
  return line_counts;
}  // namespace smqdetail

template <typename eid_t>
inline eid_t make_generated_external_id(const std::size_t      value,
                                        const std::string_view format) {
  if constexpr (std::is_convertible_v<std::size_t, eid_t> ||
                std::is_constructible_v<eid_t, std::size_t>) {
    return static_cast<eid_t>(value);
  } else {
    std::cerr << "Format '" << format
              << "' requires an explicit external ID when the external ID "
                 "type is not numeric. Use an '*-id' format instead."
              << std::endl;
    std::abort();
  }
}

template <typename iid_t, typename eid_t, typename e2i_id_map_type>
inline iid_t make_internal_id(const eid_t                         &eid,
                              const std::optional<e2i_id_map_type> &e2i_id_table,
                              const std::string_view                format) {
  if (e2i_id_table) {
    return e2i_id_table->at(eid);
  }

  if constexpr (std::is_convertible_v<eid_t, iid_t> ||
                std::is_constructible_v<iid_t, eid_t>) {
    return static_cast<iid_t>(eid);
  } else {
    std::cerr << "Format '" << format
              << "' requires an external-to-internal ID map when the "
                 "external ID type is not directly convertible to the "
                 "internal ID type."
              << std::endl;
    std::abort();
  }
}

template <typename id_t, typename fe_t>
inline std::size_t get_dims(const std::filesystem::path &point_file_path,
                            const std::string_view       format) {
  std::ifstream ifs(point_file_path);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open: " << point_file_path << std::endl;
    return 0;
  }

  std::string buf;
  if (std::getline(ifs, buf)) {
    if (format == "wsv" || format == "tsv") {
      return saltatlas::detail::parse_feature_vector<fe_t>(buf).size();
    } else if (format == "wsv-id" || format == "tsv-id") {
      const auto ret =
          saltatlas::detail::parse_feature_vector_with_id<id_t, fe_t>(buf);
      return ret.second.size();
    } else if (format == "csv") {
      return saltatlas::detail::parse_feature_vector<fe_t>(buf, ',').size();
    } else if (format == "csv-id") {
      const auto ret =
          saltatlas::detail::parse_feature_vector_with_id<id_t, fe_t>(buf, ',');
      return ret.second.size();
    } else {
      std::cerr << "Unsupported format: " << format << std::endl;
      return 0;
    }
  }

  std::cerr << "No points found in " << point_file_path << std::endl;
  return 0;
}

template <typename eid_t, typename iid_t, typename fe_t,
          typename e2i_id_map_type, typename alloc_t>
inline void load_points_kernel(
    const std::vector<std::filesystem::path> &point_file_paths,
    const std::function<std::pair<eid_t, std::vector<fe_t>>(
        const std::size_t, const std::size_t, const std::string &)>
                                            &line_parser,
    dense_point_store<iid_t, fe_t, alloc_t> &point_store,
    std::optional<e2i_id_map_type>           e2i_id_table = std::nullopt) {
  OMP_DIRECTIVE (parallel for)
  for (std::size_t i = 0; i < point_file_paths.size(); ++i) {
    const auto   &point_file_path = point_file_paths[i];
    std::ifstream ifs(point_file_path);
    if (!ifs) {
      std::cerr << "Cannot open " << point_file_path << std::endl;
      std::abort();
    }

    std::string line_buf;
    std::size_t line_no = 0;
    while (std::getline(ifs, line_buf)) {
      const auto [eid, points] = line_parser(i, line_no, line_buf);
      if (points.size() != point_store.num_dimensions()) {
        std::cerr << "Unexpected #of dimensions." << std::endl;
        std::cerr << "read dimensions: " << points.size() << std::endl;
        std::cerr << line_buf << std::endl;
        std::abort();
      }
      const auto iid =
          make_internal_id<iid_t>(eid, e2i_id_table, "point loading");
      point_store.assign(iid, points);
      ++line_no;
    }
  }
}

template <typename eid_t, typename iid_t, typename fe_t,
          typename e2i_id_map_type, typename alloc_t>
inline dense_point_store<iid_t, fe_t, alloc_t> load_points(
    const std::vector<std::filesystem::path> &point_file_paths,
    const std::string                        &format,
    std::optional<e2i_id_map_type>            e2i_id_table = std::nullopt,
    const alloc_t                            &alloc        = alloc_t()) {
  if (point_file_paths.empty()) {
    std::cerr << "No point files are given." << std::endl;
    return dense_point_store<iid_t, fe_t, alloc_t>(0, 0);
  }

  // Count #of points
  const auto line_counts = smqdetail::count_lines_in_files(point_file_paths);
  const auto n_points =
      std::accumulate(line_counts.begin(), line_counts.end(), 0ull);
  if (n_points == 0) {
    std::cerr << "No points found in the given files." << std::endl;
    return dense_point_store<iid_t, fe_t, alloc_t>(0, 0, alloc);
  }

  // Count #of dimensions
  const std::size_t dims = get_dims<eid_t, fe_t>(point_file_paths[0], format);

  dense_point_store<iid_t, fe_t, alloc_t> point_store(n_points, dims, alloc);

  std::vector<std::size_t> id_offsets(point_file_paths.size(), 0);
  for (std::size_t i = 1; i < point_file_paths.size(); ++i) {
    id_offsets[i] = id_offsets[i - 1] + line_counts[i - 1];
  }

  if (format == "wsv" || format == "tsv") {
    load_points_kernel<eid_t, iid_t, fe_t, e2i_id_map_type, alloc_t>(
        point_file_paths,
        [&format, &id_offsets](const std::size_t  file_no,
                               const std::size_t  line_no,
                               const std::string &line) {
          const eid_t id = make_generated_external_id<eid_t>(
              line_no + id_offsets.at(file_no), format);
          return std::make_pair(
              id, saltatlas::detail::parse_feature_vector<fe_t>(line));
        },
        point_store);
  } else if (format == "wsv-id" || format == "tsv-id") {
    load_points_kernel<eid_t, iid_t, fe_t, e2i_id_map_type, alloc_t>(
        point_file_paths,
        [&format](const std::size_t /*file_no*/, const std::size_t /*line_no*/,
                  const std::string &line) {
          return saltatlas::detail::parse_feature_vector_with_id<eid_t, fe_t>(
              line);
        },
        point_store, e2i_id_table);
  } else if (format == "csv") {
    load_points_kernel<eid_t, iid_t, fe_t, e2i_id_map_type, alloc_t>(
        point_file_paths,
        [&format, &id_offsets](const std::size_t  file_no,
                               const std::size_t  line_no,
                               const std::string &line) {
          const eid_t id = make_generated_external_id<eid_t>(
              line_no + id_offsets.at(file_no), format);
          return std::make_pair(
              id, saltatlas::detail::parse_feature_vector<fe_t>(line, ','));
        },
        point_store);
  } else if (format == "csv-id") {
    load_points_kernel<eid_t, iid_t, fe_t, e2i_id_map_type, alloc_t>(
        point_file_paths,
        [&format](const std::size_t /*file_no*/, const std::size_t /*line_no*/,
                  const std::string &line) {
          return saltatlas::detail::parse_feature_vector_with_id<eid_t, fe_t>(
              line, ',');
        },
        point_store, e2i_id_table);
  } else {
    std::cerr << "Unsupported format: " << format << std::endl;
    std::abort();
  }

  return point_store;
}
}  // namespace smqdetail

/// \brief Loads points from files.
/// Supported formats are:
/// wsv, tsv, wsv-id, tsv-id, csv or csv-id
/// External ID can be string or numeric type depending on the template
/// parameter.
template <typename eid_t, typename iid_t, typename fe_t,
          typename e2i_id_map_type, typename alloc_t = std::allocator<fe_t>>
inline dense_point_store<iid_t, fe_t, alloc_t> load_points(
    const std::vector<std::filesystem::path> &point_file_paths,
    const std::string                        &format,
    std::optional<e2i_id_map_type>            e2i_id_table = std::nullopt,
    const alloc_t                            &alloc        = alloc_t()) {
  return smqdetail::load_points<eid_t, iid_t, fe_t, e2i_id_map_type, alloc_t>(
      point_file_paths, format, e2i_id_table, alloc);
}
}  // namespace saltatlas