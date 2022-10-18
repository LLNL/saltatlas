// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/point_store.hpp>
#include <saltatlas/dnnd/detail/utilities/general.hpp>
#include <saltatlas/dnnd/detail/utilities/string_cast.hpp>
#include <saltatlas/dnnd/detail/utilities/ygm.hpp>

namespace saltatlas::dndetail {

/// \brief Read points (feature vectors) using multiple processes.
/// Point IDs are equal to the corresponding line numbers,
/// assuming that the input files were concatenated as a single file with the
/// order in list 'sorted_file_names'.
/// \param sorted_file_names This list must be sorted correctly. All points ID
/// in i-th file are less than the IDs in (i+k)-th files, where i >= 0 and k
/// >= 1.
template <typename id_t, typename T, typename pstore_alloc,
          typename parser_func>
void read_points_helper(
    const std::vector<std::string> &sorted_file_names, parser_func parser,
    point_store<id_t, T, pstore_alloc>       &local_point_store,
    const std::function<int(const id_t &id)> &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  // Counts #of points each process to read.
  std::size_t count_points = 0;
  const auto  range =
      partial_range(sorted_file_names.size(), comm.rank(), comm.size());
  for (std::size_t i = range.first; i < range.second; ++i) {
    const auto   &file_name = sorted_file_names[i];
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open " << file_name << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::string buf;
    while (std::getline(ifs, buf)) {
      ++count_points;
    }
  }
  comm.cf_barrier();

  std::size_t id_offset = 0;
  {
    ygm::ygm_ptr<std::size_t> ptr_id_offset(&id_offset);
    comm.cf_barrier();
    for (int r = comm.rank() + 1; r < comm.size(); ++r) {
      auto adder = [](auto, const std::size_t n,
                      const ygm::ygm_ptr<std::size_t> &ptr_offset) {
        *ptr_offset += n;
      };
      comm.async(r, adder, count_points, ptr_id_offset);
    }
    comm.barrier();
  }

  if (comm.rank() == comm.size() - 1 &&
      id_offset > std::numeric_limits<id_t>::max()) {
    comm.cerr() << "Too many points in the file(s)" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  ygm::ygm_ptr<point_store<id_t, T, pstore_alloc>> ptr_point_store(
      &local_point_store);
  local_point_store.clear();
  comm.cf_barrier();

  // Reads points
  count_points = 0;
  for (std::size_t i = range.first; i < range.second; ++i) {
    const auto &file_name = sorted_file_names[i];
    if (verbose) std::cout << "Open " << file_name << std::endl;
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open " << file_name << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Parse a vector and send it to the destination.
    std::string line_buf;
    while (std::getline(ifs, line_buf)) {
      const auto id = count_points + id_offset;

      std::vector<T> feature;
      const auto     ret = parser(line_buf, feature);
      if (!ret || feature.empty()) {
        std::cerr << "Invalid line " << line_buf << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      // Send to the corresponding process
      comm.async(
          point_partitioner(id),
          [](auto, const id_t id, const auto &feature, auto ptr_point_store) {
            ptr_point_store->set(id, feature.begin(), feature.end());
          },
          id, feature, ptr_point_store);

      ++count_points;
    }
  }
  comm.barrier();
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by 'delimiter'.
template <typename id_t, typename T, typename pstore_alloc,
          typename parser_func>
void read_points_with_id_helper(
    const std::vector<std::string> &file_names, parser_func parser,
    point_store<id_t, T, pstore_alloc>       &local_point_store,
    const std::function<int(const id_t &id)> &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto range = partial_range(file_names.size(), comm.rank(), comm.size());
  local_point_store.clear();
  static auto &ref_point_store = local_point_store;
  comm.cf_barrier();

  for (std::size_t file_no = range.first; file_no < range.second; ++file_no) {
    const auto &file_name = file_names[file_no];
    if (verbose) std::cout << "Open " << file_name << std::endl;
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open " << file_name << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::string line_buf;
    while (std::getline(ifs, line_buf)) {
      id_t           id{};
      std::vector<T> feature;
      const auto     ret = parser(line_buf, id, feature);
      if (!ret || feature.empty()) {
        std::cerr << "Invalid line " << line_buf << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      // Send to the corresponding rank
      auto receiver = [](auto, const id_t id, const auto &sent_feature) {
        if (ref_point_store.contains(id)) {
          std::cerr << "Duplicate ID " << id << std::endl;
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        auto &feature = ref_point_store.feature_vector(id);
        feature.insert(feature.begin(), sent_feature.begin(),
                       sent_feature.end());
      };
      comm.async(point_partitioner(id), receiver, id, feature);
    }
  }
  comm.barrier();
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by 'delimiter'.
template <typename id_t, typename T, typename pstore_alloc>
void read_points_with_id(
    const std::vector<std::string> &file_names, const char delimiter,
    point_store<id_t, T, pstore_alloc>       &local_point_store,
    const std::function<int(const id_t &id)> &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [delimiter](const std::string &input, id_t &id,
                                  std::vector<T> &feature) {
    std::string buf;
    bool        first = true;
    feature.clear();
    for (std::stringstream ss(input); std::getline(ss, buf, delimiter);) {
      if (first)
        id = str_cast<id_t>(buf);
      else
        feature.push_back(str_cast<T>(buf));
      first = false;
    }
    return true;
  };

  read_points_with_id_helper(file_names, parser, local_point_store,
                             point_partitioner, comm, verbose);
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by whitespace.
/// \warning
/// This function uses static variables internally. Each process must call this
/// function only once at a time.
template <typename id_type, typename T, typename pstore_alloc>
void read_points_with_id(
    const std::vector<std::string>              &file_names,
    point_store<id_type, T, pstore_alloc>       &local_point_store,
    const std::function<int(const id_type &id)> &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [](const std::string &input, id_type &id,
                         std::vector<T> &feature) {
    std::string buf;
    bool        first = true;
    feature.clear();
    for (std::stringstream ss(input); ss >> buf;) {
      if (first)
        id = str_cast<id_type>(buf);
      else
        feature.push_back(str_cast<T>(buf));
      first = false;
    }
    return true;
  };

  read_points_with_id_helper(file_names, parser, local_point_store,
                             point_partitioner, comm, verbose);
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files do not contain ID and each column is separated by
/// 'delimiter'.
template <typename id_type, typename T, typename pstore_alloc>
void read_points(const std::vector<std::string>              &file_names,
                 const char                                   delimiter,
                 point_store<id_type, T, pstore_alloc>       &local_point_store,
                 const std::function<int(const id_type &id)> &point_partitioner,
                 ygm::comm &comm, const bool verbose) {
  const auto parser = [delimiter](const std::string &input,
                                  std::vector<T>    &feature) {
    std::string buf;
    feature.clear();
    for (std::stringstream ss(input); std::getline(ss, buf, delimiter);) {
      feature.push_back(str_cast<T>(buf));
    }
    return true;
  };

  read_points_helper(file_names, parser, local_point_store, point_partitioner,
                     comm, verbose);
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files do not contain ID and each column is separated by
/// whitespace.
template <typename id_t, typename T, typename pstore_alloc>
void read_points(const std::vector<std::string>           &file_names,
                 point_store<id_t, T, pstore_alloc>       &local_point_store,
                 const std::function<int(const id_t &id)> &point_partitioner,
                 ygm::comm &comm, const bool verbose) {
  const auto parser = [](const std::string &input, std::vector<T> &feature) {
    std::string buf;
    feature.clear();
    for (std::stringstream ss(input); ss >> buf;) {
      feature.push_back(str_cast<T>(buf));
    }
    return true;
  };

  read_points_helper(file_names, parser, local_point_store, point_partitioner,
                     comm, verbose);
}

}  // namespace saltatlas::dndetail

namespace saltatlas {
template <typename id_type, typename feature_element_type,
          typename point_store_allocator_type>
inline void read_points(
    const std::vector<std::string> &point_file_names,
    const std::string_view &format, const bool verbose,
    const std::function<int(const id_type &id)>       &point_partitioner,
    dndetail::point_store<id_type, feature_element_type,
                          point_store_allocator_type> &local_point_store,
    ygm::comm                                         &comm) {
  if (format == "wsv") {
    if (verbose)
      comm.cout0() << "Read WSV format (whitespace separated, no ID) files"
                   << std::endl;
    dndetail::read_points(point_file_names, local_point_store,
                          point_partitioner, comm, verbose);
  } else if (format == "wsv-id") {
    if (verbose)
      comm.cout0() << "Read WSV-ID (whitespace separated with ID) format files"
                   << std::endl;
    dndetail::read_points_with_id(point_file_names, local_point_store,
                                  point_partitioner, comm, verbose);
  } else if (format == "csv") {
    if (verbose)
      comm.cout0() << "Read CSV format (without ID) files" << std::endl;
    dndetail::read_points(point_file_names, ',', local_point_store,
                          point_partitioner, comm, verbose);
  } else if (format == "csv-id") {
    if (verbose) comm.cout0() << "Read CSV-ID format files" << std::endl;
    dndetail::read_points_with_id(point_file_names, ',', local_point_store,
                                  point_partitioner, comm, verbose);
  } else {
    comm.cerr0() << "Invalid reader mode" << std::endl;
  }
}

namespace {
using saltatlas::dndetail::neighbor;

template <typename id_t, typename dist_t>
using neighbors_tbl = std::vector<std::vector<neighbor<id_t, dist_t>>>;
}  // namespace

/// \brief
/// \tparam id_type
/// \tparam distance_type
/// \param file_path
/// \param store
template <typename id_type, typename distance_type>
inline void read_neighbors(const std::string_view                &file_path,
                           neighbors_tbl<id_type, distance_type> &store) {
  std::ifstream ifs(file_path.data());
  if (!ifs.is_open()) {
    std::cerr << "Failed to open: " << file_path << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  std::size_t num_entries = 0;
  {
    std::size_t cnt_lines = 0;
    for (std::string buf; std::getline(ifs, buf);) {
      ++cnt_lines;
    }
    if (!ifs.eof() && (ifs.bad() || ifs.fail())) {
      std::cerr << "Failed reading data from " << file_path << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    ifs.clear();
    ifs.seekg(0);

    if (cnt_lines % 2 != 0) {
      std::cerr << "#of lines in the file is not an even number: " << file_path
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    num_entries = cnt_lines / 2;
  }
  store.reserve(num_entries);

  std::size_t num_neighbors_per_entry = 0;
  {
    std::string buf;
    std::getline(ifs, buf);
    std::stringstream ss(buf);
    id_type           id;
    while (ss >> id) {
      ++num_neighbors_per_entry;
    }
    if (!ifs.eof() && (ifs.bad() || ifs.fail())) {
      std::cerr << "Failed reading data from " << file_path << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    ifs.clear();
    ifs.seekg(0);
  }

  // Reads neighbor IDs.
  for (std::string buf; std::getline(ifs, buf);) {
    std::vector<dndetail::neighbor<id_type, distance_type>> neighbors;
    std::stringstream                                       ss(buf);
    id_type                                                 id;
    while (ss >> id) {
      neighbors.emplace_back(id, distance_type{});
    }
    if (neighbors.size() != num_neighbors_per_entry) {
      std::cerr << "#of neighbors per line are not the same" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    store.push_back(std::move(neighbors));
    if (store.size() == num_entries) break;
  }
  if (store.size() != num_entries || ifs.bad() || ifs.fail()) {
    std::cerr << "Failed reading data from " << file_path << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Reads distances.
  std::size_t i = 0;
  for (std::string buf; std::getline(ifs, buf);) {
    std::stringstream ss(buf);
    distance_type     d;
    std::size_t       k = 0;
    while (ss >> d) {
      store[i][k++].distance = d;
    }
    if (k != num_neighbors_per_entry) {
      std::cerr << "#of neighbors per line are not the same" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    ++i;
  }
  if (i != num_entries ||
      (!ifs.eof() && (ifs.bad() || ifs.fail()))) {
    std::cerr << "Failed reading data from " << file_path << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
}

/// \tparam id_type
/// \tparam distance_type
/// \param file_path
/// \param store
/// \param comm
template <typename id_type, typename distance_type>
inline void read_neighbors(const std::string_view                &file_path,
                           neighbors_tbl<id_type, distance_type> &store,
                           ygm::comm                             &comm) {
  neighbors_tbl<id_type, distance_type> global_store;
  if (comm.rank0()) {
    read_neighbors(file_path, global_store);
  }
  dndetail::distribute_elements_by_block(global_store, store, comm);
}

/// \brief Reads a file that contain queries.
/// Each line is the feature vector of a query point.
/// Can read the white space separated format (without ID).
/// \tparam feature_element_type Feature element type.
/// \param query_file_path Path to a query file.
/// \param queries Buffer to store read queries.
template <typename feature_element_type>
inline void read_query(
    const std::string_view                         &query_file_path,
    std::vector<std::vector<feature_element_type>> &queries) {
  if (query_file_path.empty()) return;

  std::ifstream ifs(query_file_path.data());
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << query_file_path << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  std::string buf;
  while (std::getline(ifs, buf)) {
    std::stringstream    ss(buf);
    feature_element_type v;
    queries.resize(queries.size() + 1);
    while (ss >> v) {
      queries.back().push_back(v);
    }
  }
}

/// \brief
/// \tparam feature_element_type
/// \param query_file_path
/// \param queries
/// \param comm
template <typename feature_element_type>
inline void read_query(const std::string_view &query_file_path,
                       std::vector<std::vector<feature_element_type>> &queries,
                       ygm::comm                                      &comm) {
  std::vector<std::vector<feature_element_type>> global_store;
  if (comm.rank0()) {
    read_query(query_file_path, global_store);
  }
  dndetail::distribute_elements_by_block(global_store, queries, comm);
}

}  // namespace saltatlas
