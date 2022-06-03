// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ygm/comm.hpp"

#include "saltatlas/dnnd/detail/point_store.hpp"
#include "saltatlas/dnnd/detail/utilities/general.hpp"
#include "saltatlas/dnnd/detail/utilities/string_cast.hpp"

namespace saltatlas::dndetail {

/// \brief Read points (feature vectors) using multiple processes.
/// Point IDs are equal to the corresponding line numbers,
/// assuming that the input files were concatenated as a single file with the
/// order in list 'sorted_file_names'. \param sorted_file_names This list must
/// be sorted correctly. All points ID in i-th file are less than the IDs in
/// (i+k)-th files, where i >= 0 and k >= 1.
template <typename id_type, typename feature_element_type,
          typename point_store_allocator_type>
void read_points_wsv(const std::vector<std::string>          &sorted_file_names,
                     point_store<id_type, feature_element_type,
                                 point_store_allocator_type> &local_point_store,
                     ygm::comm                               &comm) {
  using point_store_type =
      point_store<id_type, feature_element_type, point_store_allocator_type>;
  using feature_vec_type = typename point_store_type::feature_vector_type;

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

  /// TODO: use YGM's all-to-all function
  std::vector<std::size_t> num_points(comm.size());
  static auto             &ref_num_points = num_points;
  comm.cf_barrier();

  for (int r = 0; r < comm.size(); ++r) {
    if (r == comm.rank()) {
      for (int d = 0; d < comm.size(); ++d) {
        auto receiver = []([[maybe_unused]] auto pcomm,
                           const id_type source_rank, const std::size_t n) {
          ref_num_points[source_rank] = n;
        };
        comm.async(d, receiver, comm.rank(), count_points);
      }
    }
  }
  comm.barrier();

  std::vector<std::size_t> id_offsets(comm.size(), 0);
  for (std::size_t i = 1; i < id_offsets.size(); ++i) {
    id_offsets[i] = id_offsets[i - 1] + num_points[i - 1];
  }
  id_type id = id_offsets[comm.rank()];

  local_point_store.clear();
  static auto &ref_point_store = local_point_store;
  comm.cf_barrier();

  for (std::size_t i = range.first; i < range.second; ++i) {
    const auto   &file_name = sorted_file_names[i];
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open " << file_name << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::string buf;
    while (std::getline(ifs, buf)) {
      std::stringstream    ss(buf);
      feature_element_type p;
      feature_vec_type     feature;
      while (ss >> p) {
        feature.push_back(p);
      }

      auto receiver = []([[maybe_unused]] auto pcomm, const id_type id,
                         const auto &feature) {
        ref_point_store.feature_vector(id) = feature;
      };
      comm.async(id % comm.size(), receiver, id, feature);
      ++id;
    }
  }
  comm.barrier();
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by 'delimiter'.
/// \param file_names A list of file names.
template <typename id_type, typename feature_element_type,
          typename point_store_allocator_type, typename converter_function>
void read_points_with_id(
    const std::vector<std::string> &file_names,
    const converter_function       &converter,
    point_store<id_type, feature_element_type, point_store_allocator_type>
              &local_point_store,
    ygm::comm &comm) {
  using point_store_type =
      point_store<id_type, feature_element_type, point_store_allocator_type>;
  using feature_vec_type = typename point_store_type::feature_vector_type;

  const auto range = partial_range(file_names.size(), comm.rank(), comm.size());
  local_point_store.clear();
  static auto &ref_point_store = local_point_store;
  comm.cf_barrier();

  for (std::size_t file_no = range.first; file_no < range.second; ++file_no) {
    const auto &file_name = file_names[file_no];
    std::cout << "Open " << file_name << std::endl;
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open " << file_name << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::string line_buf;
    while (std::getline(ifs, line_buf)) {
      id_type          id{};
      feature_vec_type feature{};
      const auto       ret = converter(line_buf, &id, &feature);
      if (!ret || feature.empty()) {
        std::cerr << "Invalid line " << line_buf << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      // Send to the corresponding rank
      auto receiver = [](auto, const id_type id, const auto &feature) {
        if (ref_point_store.contains(id)) {
          std::cerr << "Duplicate ID " << id << std::endl;
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        ref_point_store.feature_vector(id) = feature;
      };
      comm.async(id % comm.size(), receiver, id, feature);
    }
  }
  comm.barrier();
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by 'delimiter'.
/// \param file_names A list of file names.
template <typename id_type, typename feature_element_type,
          typename point_store_allocator_type>
void read_points_with_id_and_delimiter(
    const std::vector<std::string> &file_names, const char delimiter,
    point_store<id_type, feature_element_type, point_store_allocator_type>
              &local_point_store,
    ygm::comm &comm) {
  using point_store_type =
      point_store<id_type, feature_element_type, point_store_allocator_type>;
  using feature_vec_type = typename point_store_type::feature_vector_type;

  static const auto converter = [delimiter](const std::string &input,
                                            id_type           *id,
                                            feature_vec_type  *feature) {
    std::vector<std::string> result;
    std::string              buf;
    bool                     first = true;
    feature->clear();
    for (std::stringstream ss(input); std::getline(ss, buf, delimiter);) {
      if (first)
        *id = str_cast<id_type>(buf);
      else
        feature->push_back(str_cast<feature_element_type>(buf));
      first = false;
    }
    return true;
  };

  read_points_with_id(file_names, converter, local_point_store, comm);
}

/// \brief Read points (feature vectors) using multiple processes.
/// The input files contains ID at the first column,
/// and each column is separated by whitespace.
/// \param file_names A list of file names.
/// \warning
/// This function uses static variables internally. Each process must call this
/// function only once at a time.
template <typename id_type, typename feature_element_type,
          typename point_store_allocator_type>
void read_points_whitespace_separated_with_id(
    const std::vector<std::string> &file_names,
    point_store<id_type, feature_element_type, point_store_allocator_type>
              &local_point_store,
    ygm::comm &comm) {
  using point_store_type =
      point_store<id_type, feature_element_type, point_store_allocator_type>;
  using feature_vec_type = typename point_store_type::feature_vector_type;

  static const auto converter = [](const std::string &input, id_type *id,
                                   feature_vec_type *feature) {
    std::vector<std::string> result;
    std::string              buf;
    bool                     first = true;
    feature->clear();
    for (std::stringstream ss(input); ss >> buf;) {
      if (first)
        *id = str_cast<id_type>(buf);
      else
        feature->push_back(str_cast<feature_element_type>(buf));
      first = false;
    }
    return true;
  };

  read_points_with_id(file_names, converter, local_point_store, comm);
}

}  // namespace saltatlas::dndetail

namespace saltatlas {
template <typename id_type, typename feature_element_type,
          typename point_store_allocator_type>
void read_points(
    const std::vector<std::string> &point_file_names, const std::string &format,
    const bool                                         verbose,
    dndetail::point_store<id_type, feature_element_type,
                          point_store_allocator_type> &local_point_store,
    ygm::comm                                         &comm) {
  if (format == "wsv") {
    if (verbose)
      comm.cout0() << "Read WSV format (whitespace separated, no ID) files"
                   << std::endl;
    dndetail::read_points_wsv(point_file_names, local_point_store, comm);
  } else if (format == "csv-id") {
    if (verbose) comm.cout0() << "Read CSV-ID format files" << std::endl;
    dndetail::read_points_with_id_and_delimiter(point_file_names, ',',
                                                local_point_store, comm);
  } else if (format == "wsv-id") {
    if (verbose)
      comm.cout0() << "Read WSV-ID (whitespace separated with ID) format files"
                   << std::endl;
    dndetail::read_points_whitespace_separated_with_id(point_file_names,
                                                       local_point_store, comm);
  } else {
    comm.cerr0() << "Invalid reader mode" << std::endl;
  }
}
}  // namespace saltatlas
