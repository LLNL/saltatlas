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
#include <ygm/detail/collective.hpp>

#include "saltatlas/common/detail/data_reader_kernel.hpp"
#include "saltatlas/common/detail/neighbor.hpp"
#include "saltatlas/common/detail/utilities/hash.hpp"
#include "saltatlas/common/detail/utilities/ygm.hpp"
#include "saltatlas/common/point_store.hpp"

namespace saltatlas::detail {

/// \brief Read points (feature vectors) using multiple processes.
/// Point IDs are equal to the corresponding line numbers,
/// assuming that the input files were concatenated as a single file with the
/// order in list 'sorted_file_names'.
/// \param sorted_file_names This list must be sorted correctly. All points ID
/// in i-th file are less than the IDs in (i+k)-th files, where i >= 0 and k
/// >= 1.
template <typename id_t, typename point_t, typename H, typename E,
          typename pstore_alloc, typename parser_func>
inline void read_points_helper(
    const std::vector<std::filesystem::path>       &sorted_file_names,
    parser_func                                     parser,
    point_store<id_t, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_t &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto worker = [num_ranks = comm.size()](const std::size_t i) {
    return hash<>{}(i) % num_ranks;
  };

  // Counts #of points each file contains
  std::vector<std::size_t> file_num_points(sorted_file_names.size(), 0);
  for (std::size_t i = 0; i < sorted_file_names.size(); ++i) {
    if (worker(i) != comm.rank()) continue;
    const auto   &file_name = sorted_file_names[i];
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      comm.cerr() << "Failed to open " << file_name << std::endl;
      MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
    }
    std::string buf;
    std::size_t count_points = 0;
    while (std::getline(ifs, buf)) {
      ++count_points;
    }
    file_num_points[i] = count_points;
  }
  comm.cf_barrier();

  // Broadcasts #of points in each file
  for (std::size_t i = 0; i < sorted_file_names.size(); ++i) {
    file_num_points[i] = comm.mpi_bcast(file_num_points[i], worker(i));
  }
  comm.cf_barrier();

  // Calculates ID offset for each file
  std::vector<std::size_t> id_offsets(sorted_file_names.size(), 0);
  for (std::size_t i = 1; i < sorted_file_names.size(); ++i) {
    id_offsets[i] = id_offsets[i - 1] + file_num_points[i - 1];
  }

  // Sanity check
  const std::size_t total_num_points =
      id_offsets.back() + file_num_points.back();
  if (verbose) {
    comm.cout0() << "#of total points: " << total_num_points << std::endl;
  }
  if constexpr (std::is_integral_v<id_t>) {
    if (std::numeric_limits<id_t>::max() <= total_num_points) {
      comm.cerr0() << "Too small ID type: " << typeid(id_t).name()
                   << " to hold " << total_num_points << std::endl;
      MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
    }
  }

  ygm::ygm_ptr<point_store<id_t, point_t, H, E, pstore_alloc>> ptr_point_store(
      &local_point_store);
  comm.cf_barrier();

  // Reads points
  std::size_t count_points     = 0;
  std::size_t unreadable_lines = 0;
  for (std::size_t i = 0; i < sorted_file_names.size(); ++i) {
    if (worker(i) != comm.rank()) continue;
    const auto &file_name = sorted_file_names[i];
    if (verbose) std::cout << "Open " << file_name << std::endl;
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open " << file_name << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Parse a vector and send it to the destination.
    std::string line_buf;
    id_t        id = id_offsets[i];
    while (std::getline(ifs, line_buf)) {
      point_t point;
      bool    parse_success;
      try {
        parse_success = parser(line_buf, point);
      } catch (...) {
        parse_success = false;
      }

      if (not parse_success) {
        ++unreadable_lines;
        if (verbose) {
          std::cerr << "Unable to read line " << id - id_offsets[i] + 1
                    << " in " << file_name << ": " << line_buf << std::endl;
        }

        ++id;
        continue;
      }

      // Send to the corresponding process
      comm.async(
          point_partitioner(id),
          [](auto, const id_t id, const auto &point, auto ptr_point_store) {
            if (ptr_point_store->contains(id)) {
              std::cerr << "Duplicate ID " << id << std::endl;
              MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            auto &p = (*ptr_point_store)[id];
            p       = point;
          },
          id, point, ptr_point_store);

      ++count_points;
      ++id;
    }
  }
  comm.barrier();
  std::size_t total_unreadable_lines = ygm::sum(unreadable_lines, comm);
  if (total_unreadable_lines != 0) {
    comm.cerr0() << "Found " << total_unreadable_lines << " unreadable lines"
                 << std::endl;
  }
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by 'delimiter'.
template <typename id_t, typename point_t, typename H, typename E,
          typename pstore_alloc, typename parser_func>
inline void read_points_with_id_helper(
    const std::vector<std::filesystem::path> &file_names, parser_func parser,
    point_store<id_t, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_t &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto assigned = [&comm](const std::size_t i) -> bool {
    return (hash<>{}(i) % comm.size()) == comm.rank();
  };
  static auto &ref_point_store = local_point_store;
  ref_point_store              = local_point_store;
  comm.cf_barrier();

  std::size_t unreadable_lines = 0;
  for (std::size_t file_no = 0; file_no < file_names.size(); ++file_no) {
    if (!assigned(file_no)) continue;

    const auto &file_name = file_names[file_no];
    if (verbose) std::cout << "Open " << file_name << std::endl;
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open " << file_name << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::string line_buf;
    std::size_t line_num{0};
    while (std::getline(ifs, line_buf)) {
      ++line_num;

      id_t    id{};
      point_t point;
      bool    parse_success;
      try {
        parse_success = parser(line_buf, id, point);
      } catch (...) {
        parse_success = false;
      }

      if (not parse_success) {
        ++unreadable_lines;
        if (verbose) {
          std::cerr << "Unable to read line " << line_num << " in " << file_name
                    << ": " << line_buf << std::endl;
        }
        continue;
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
  std::size_t total_unreadable_lines = ygm::sum(unreadable_lines, comm);
  if (total_unreadable_lines != 0) {
    comm.cerr0() << "Found " << total_unreadable_lines << " unreadable lines"
                 << std::endl;
  }
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by 'delimiter'.
template <typename id_t, typename point_t, typename H, typename E,
          typename pstore_alloc>
inline void read_points_with_id(
    const std::vector<std::filesystem::path> &file_names, const char delimiter,
    point_store<id_t, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_t &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [delimiter](const std::string &input, id_t &id,
                                  point_t &point) {
    const auto [pid, elems] =
        parse_feature_vector_with_id<id_t, typename point_t::value_type>(
            input, delimiter);
    id = pid;
    point.clear();
    point.insert(point.begin(), elems.begin(), elems.end());
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
inline void read_points_with_id(
    const std::vector<std::filesystem::path>          &file_names,
    point_store<id_type, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_type &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [](const std::string &input, id_type &id,
                         point_t &point) {
    const auto [pid, elems] =
        parse_feature_vector_with_id<id_type, typename point_t::value_type>(
            input);
    id = pid;
    point.clear();
    point.insert(point.begin(), elems.begin(), elems.end());
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
inline void read_points(
    const std::vector<std::filesystem::path> &file_names, const char delimiter,
    point_store<id_type, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_type &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [delimiter](const std::string &input, point_t &point) {
    const auto elems =
        parse_feature_vector<typename point_t::value_type>(input, delimiter);
    point.clear();
    point.insert(point.begin(), elems.begin(), elems.end());
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
inline void read_points(
    const std::vector<std::filesystem::path>          &file_names,
    point_store<id_type, point_t, H, E, pstore_alloc> &local_point_store,
    const std::function<int(const id_type &id)>       &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  const auto parser = [](const std::string &input, point_t &point) {
    const auto elems =
        parse_feature_vector<typename point_t::value_type>(input);
    point.clear();
    point.insert(point.begin(), elems.begin(), elems.end());
    return true;
  };

  read_points_helper(file_names, parser, local_point_store, point_partitioner,
                     comm, verbose);
}
}  // namespace saltatlas::detail

namespace saltatlas {

/// \brief Read points (feature vectors) using multiple processes.
/// Supported values for `format` are:
/// - `wsv`, `tsv`: whitespace-separated points without explicit IDs.
///   IDs are generated from global line order and therefore require integral
///   `id_type`.
/// - `wsv-id`, `tsv-id`: whitespace-separated records where the first field is
///   an explicit ID. External ID can be string or numeric type depending on the
///   template parameter.
/// - `csv`: comma-separated points without explicit IDs. Requires integral
///   `id_type`.
/// - `csv-id`: comma-separated records where the first field is an explicit ID.
///   External ID can be string or numeric type depending on the template
///   parameter.
/// - `str`: whitespace-separated string points without explicit IDs. Requires
///   `point_t::value_type == char` and integral `id_type`.
/// - `str-id`: string points with explicit IDs. External ID can be string or
/// numeric type depending on the template parameter.
///
/// Parse or type-constraint failures are reported to `comm.cerr0()`. Input
/// records that pass parsing are inserted into `local_point_store` on the rank
/// returned by `point_partitioner(id)`.
///
/// \param point_file_names Input files. For formats without explicit IDs, files
/// should be ordered by global ID order.
/// \param format Input record format selector listed above.
/// \param verbose Enables informational logs on rank 0 and per-rank file-open
/// logs in helpers.
/// \param point_partitioner Maps point IDs to destination rank IDs.
/// \param local_point_store Local distributed storage for owned points.
/// \param comm YGM communicator used for distributed reads and point exchange.
template <typename id_type, typename point_t, typename H, typename E,
          typename PA>
inline void read_points(
    const std::vector<std::filesystem::path> &point_file_names,
    const std::string &format, const bool verbose,
    const std::function<int(const id_type &id)> &point_partitioner,
    point_store<id_type, point_t, H, E, PA>     &local_point_store,
    ygm::comm                                   &comm) {
  if (format == "wsv" || format == "tsv") {
    if (verbose)
      comm.cout0() << "Read WSV/TSV (whitespace separated, no ID) format files"
                   << std::endl;
    if constexpr (std::is_integral_v<id_type>) {
      detail::read_points(point_file_names, local_point_store,
                          point_partitioner, comm, verbose);
    } else {
      comm.cerr0() << "ID type must be an integral type for WSV/TSV format"
                   << std::endl;
    }
  } else if (format == "wsv-id" || format == "tsv-id") {
    if (verbose)
      comm.cout0()
          << "Read WSV-ID/TSV-ID (whitespace separated with ID) format files"
          << std::endl;
    detail::read_points_with_id(point_file_names, local_point_store,
                                point_partitioner, comm, verbose);
  } else if (format == "csv") {
    if (verbose)
      comm.cout0() << "Read CSV format (without ID) files" << std::endl;
    if constexpr (std::is_integral_v<id_type>) {
      detail::read_points(point_file_names, ',', local_point_store,
                          point_partitioner, comm, verbose);
    } else {
      comm.cerr0() << "ID type must be an integral type for CSV format"
                   << std::endl;
    }
  } else if (format == "csv-id") {
    if (verbose) comm.cout0() << "Read CSV-ID format files" << std::endl;
    detail::read_points_with_id(point_file_names, ',', local_point_store,
                                point_partitioner, comm, verbose);
  } else if (format == "str") {
    if (verbose) comm.cout0() << "Read string format files" << std::endl;
    if (!std::is_same_v<typename point_t::value_type, char>) {
      comm.cerr0() << "Point type must be a vector of char" << std::endl;
    } else {
      if constexpr (std::is_integral_v<id_type>) {
        detail::read_points(point_file_names, local_point_store,
                            point_partitioner, comm, verbose);
      } else {
        comm.cerr0() << "ID type must be an integral type for STR format"
                     << std::endl;
      }
    }
  } else if (format == "str-id") {
    if (verbose)
      comm.cout0() << "Read string format files with IDs" << std::endl;
    if (!std::is_same_v<typename point_t::value_type, char>) {
      comm.cerr0() << "Point type must be a vector of char" << std::endl;
    } else {
      detail::read_points_with_id(point_file_names, local_point_store,
                                  point_partitioner, comm, verbose);
    }
  } else {
    comm.cerr0() << "Unsupported point file format: " << format << std::endl;
  }
}

/// \brief Reads a neighbor file and distributes them.
template <typename id_type, typename distance_type>
inline void read_neighbors(
    const std::filesystem::path &file_path,
    std::vector<std::vector<detail::neighbor<id_type, distance_type>>> &store,
    ygm::comm                                                          &comm) {
  std::vector<std::vector<detail::neighbor<id_type, distance_type>>>
      global_store;
  if (comm.rank0()) {
    if (!detail::read_neighbors_kernel(file_path, global_store)) {
      MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
    }
  }
  detail::distribute_elements_by_block(global_store, store, comm);
}

/// \brief
/// \tparam point_type
/// \param query_file_path
/// \param queries
/// \param comm
template <typename point_type>
inline void read_query(const std::filesystem::path &query_file_path,
                       std::vector<point_type> &queries, ygm::comm &comm) {
  std::vector<point_type> global_store;
  if (comm.rank0()) {
    if (!detail::read_query_kernel(query_file_path, global_store)) {
      MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
    }
  }
  detail::distribute_elements_by_block(global_store, queries, comm);
}

}  // namespace saltatlas
