// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/utilities/general.hpp>
#include <saltatlas/dnnd/detail/utilities/string_cast.hpp>
#include <saltatlas/dnnd/detail/utilities/ygm.hpp>
#include "saltatlas/point_store.hpp"

namespace saltatlas::dndetail {

/// \brief Read points (feature vectors) using multiple processes.
/// Point IDs are equal to the corresponding line numbers,
/// assuming that the input files were concatenated as a single file with the
/// order in list 'sorted_file_names'.
/// \param sorted_file_names This list must be sorted correctly. All points ID
/// in i-th file are less than the IDs in (i+k)-th files, where i >= 0 and k
/// >= 1.
template <typename id_t, typename point_t, typename H, typename E,
          typename pstore_alloc, typename parser_func>
void read_points_helper(
    const std::vector<std::filesystem::path>       &sorted_file_names,
    parser_func                                     parser,
    point_store<id_t, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_t &id)>       &point_partitioner,
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

  ygm::ygm_ptr<point_store<id_t, point_t, H, E, pstore_alloc>> ptr_point_store(
      &local_point_store);
  local_point_store.reset();
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

      point_t    point;
      const auto ret = parser(line_buf, point);
      if (!ret || point.empty()) {
        std::cerr << "Invalid line " << line_buf << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      // Send to the corresponding process
      comm.async(
          point_partitioner(id),
          [](auto, const id_t id, const auto &point, auto ptr_point_store) {
            auto &p = (*ptr_point_store)[id];
            p.clear();
            p.insert(p.begin(), point.begin(), point.end());
          },
          id, point, ptr_point_store);

      ++count_points;
    }
  }
  comm.barrier();
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by 'delimiter'.
template <typename id_t, typename point_t, typename H, typename E,
          typename pstore_alloc, typename parser_func>
void read_points_with_id_helper(
    const std::vector<std::filesystem::path> &file_names, parser_func parser,
    point_store<id_t, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_t &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto range = partial_range(file_names.size(), comm.rank(), comm.size());
  local_point_store.reset();
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
      id_t       id{};
      point_t    point;
      const auto ret = parser(line_buf, id, point);
      if (!ret) {
        std::cerr << "Invalid line " << line_buf << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      // Send to the corresponding rank
      auto receiver = [](auto, const id_t id, const auto &sent_point) {
        if (ref_point_store.contains(id)) {
          std::cerr << "Duplicate ID " << id << std::endl;
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        ref_point_store[id] = sent_point;
      };
      comm.async(point_partitioner(id), receiver, id, point);
    }
  }
  comm.barrier();
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by 'delimiter'.
template <typename id_t, typename point_t, typename H, typename E,
          typename pstore_alloc>
void read_points_with_id(
    const std::vector<std::filesystem::path> &file_names, const char delimiter,
    point_store<id_t, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_t &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [delimiter](const std::string &input, id_t &id,
                                  point_t &point) {
    std::string buf;
    bool        first = true;
    point.clear();
    for (std::stringstream ss(input); std::getline(ss, buf, delimiter);) {
      if (first) {
        id = str_cast<id_t>(buf);
      } else {
        point.push_back(str_cast<typename point_t::value_type>(buf));
      }
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
template <typename id_type, typename point_t, typename H, typename E,
          typename pstore_alloc>
void read_points_with_id(
    const std::vector<std::filesystem::path>          &file_names,
    point_store<id_type, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_type &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [](const std::string &input, id_type &id,
                         point_t &point) {
    std::string buf;
    bool        first = true;
    point.clear();
    for (std::stringstream ss(input); ss >> buf;) {
      if (first) {
        id = str_cast<id_type>(buf);
      } else {
        point.push_back(str_cast<typename point_t::value_type>(buf));
      }
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
template <typename id_type, typename point_t, typename H, typename E,
          typename pstore_alloc>
void read_points(
    const std::vector<std::filesystem::path> &file_names, const char delimiter,
    point_store<id_type, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_type &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [delimiter](const std::string &input, point_t &point) {
    const auto buf = str_split<typename point_t::value_type>(input, delimiter);
    point.clear();
    point.insert(point.begin(), buf.begin(), buf.end());
    return true;
  };

  read_points_helper(file_names, parser, local_point_store, point_partitioner,
                     comm, verbose);
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files do not contain ID and each column is separated by
/// whitespace.
template <typename id_type, typename point_t, typename H, typename E,
          typename pstore_alloc>
void read_points(
    const std::vector<std::filesystem::path>          &file_names,
    point_store<id_type, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_t &id)>          &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [](const std::string &input, point_t &point) {
    const auto buf = str_split<typename point_t::value_type>(input);
    point.clear();
    point.insert(point.begin(), buf.begin(), buf.end());
    return true;
  };

  read_points_helper(file_names, parser, local_point_store, point_partitioner,
                     comm, verbose);
}
}  // namespace saltatlas::dndetail

namespace saltatlas {

/// \brief Read points (feature vectors) using multiple processes.
template <typename id_type, typename point_t, typename H, typename E,
          typename PA>
inline void read_points(
    const std::vector<std::filesystem::path> &point_file_names,
    const std::filesystem::path &format, const bool verbose,
    const std::function<int(const id_type &id)> &point_partitioner,
    point_store<id_type, point_t, H, E, PA>     &local_point_store,
    ygm::comm                                   &comm) {
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
  } else if (format == "str") {
    if (verbose) comm.cout0() << "Read string format files" << std::endl;
    dndetail::read_points(point_file_names, local_point_store,
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
inline void read_neighbors(const std::filesystem::path           &file_path,
                           neighbors_tbl<id_type, distance_type> &store) {
  std::ifstream ifs(file_path);
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

  // Count #of neighbors per entry (line) by reading the first line.
  std::size_t num_neighbors_per_entry = 0;
  {
    std::string buf;
    std::getline(ifs, buf);
    if (!ifs.eof() && (ifs.bad() || ifs.fail())) {
      std::cerr << "Failed reading data from " << file_path << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    ifs.clear();
    ifs.seekg(0);

    num_neighbors_per_entry = dndetail::str_split<id_type>(buf).size();
  }

  // Reads neighbor IDs.
  for (std::string buf; std::getline(ifs, buf);) {
    const auto ids = dndetail::str_split<id_type>(buf);
    std::vector<dndetail::neighbor<id_type, distance_type>> neighbors;
    for (const auto id : ids) {
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
  std::size_t line_no = 0;
  for (std::string buf; std::getline(ifs, buf);) {
    const auto distances = dndetail::str_split<distance_type>(buf);
    if (distances.size() != num_neighbors_per_entry) {
      std::cerr << "#of neighbors per line are not the same" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::size_t k = 0;
    for (const auto d : distances) {
      store[line_no][k++].distance = d;
    }

    ++line_no;
  }
  if (line_no != num_entries || (!ifs.eof() && (ifs.bad() || ifs.fail()))) {
    std::cerr << "Failed reading data from " << file_path << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
}

/// \brief Reads a neighbor file and distributes them.
template <typename id_type, typename distance_type>
inline void read_neighbors(const std::filesystem::path           &file_path,
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
/// \tparam point_t Point type.
/// \param query_file_path Path to a query file.
/// \param queries Buffer to store read queries.
template <typename point_t>
inline void read_query(const std::filesystem::path &query_file_path,
                       std::function<point_t(const std::string &)> parser,
                       std::vector<point_t>                       &queries) {
  if (query_file_path.empty()) return;

  std::ifstream ifs(query_file_path);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << query_file_path << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  for (std::string line; std::getline(ifs, line);) {
    queries.push_back(parser(line));
  }
}

/// \brief read_query function for reading feature vectors.
template <typename point_t>
inline void read_query(const std::filesystem::path &query_file_path,
                       std::vector<point_t>        &queries) {
  read_query<point_t>(
      query_file_path,
      [](const std::string &line) {
        auto data = dndetail::str_split<typename point_t::value_type>(line);
        return point_t(data.begin(), data.end());
      },
      queries);
}

/// \brief
/// \tparam point_t
/// \param query_file_path
/// \param queries
/// \param comm
template <typename point_t>
inline void read_query(const std::filesystem::path &query_file_path,
                       std::vector<point_t> &queries, ygm::comm &comm) {
  std::vector<point_t> global_store;
  if (comm.rank0()) {
    read_query(query_file_path, global_store);
  }
  dndetail::distribute_elements_by_block(global_store, queries, comm);
}

}  // namespace saltatlas
