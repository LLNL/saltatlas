// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/dnnd.hpp>
#include <saltatlas/dnnd/dnnd_pm.hpp>

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;

using dnnd_type = saltatlas::dnnd<id_type, feature_element_type, distance_type>;
using neighbor_type = typename dnnd_type::neighbor_type;

using dnnd_pm_type =
    saltatlas::dnnd_pm<id_type, feature_element_type, distance_type>;
using pm_neighbor_type = typename dnnd_pm_type::neighbor_type;

static constexpr std::size_t k_ygm_buff_size = 256 * 1024 * 1024;

/// \brief Reads a file that contain queries.
/// Each line is the feature vector of a query point.
/// Can read the white space separated format (without ID).
template <typename dnnd_type>
inline void read_query(
    const std::string                          &query_file,
    typename dnnd_type::query_point_store_type &query_points) {
  if (query_file.empty()) return;

  std::ifstream ifs(query_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << query_file << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  typename dnnd_type::id_type id = 0;
  std::string                 buf;
  while (std::getline(ifs, buf)) {
    std::stringstream                                     ss(buf);
    typename dnnd_type::feature_element_type              p;
    std::vector<typename dnnd_type::feature_element_type> feature;
    while (ss >> p) {
      feature.push_back(p);
    }
    query_points.feature_vector(id).insert(
        query_points.feature_vector(id).begin(), feature.begin(),
        feature.end());
    ++id;
  }
}

/// \brief Gather query results to the root rank.
template <typename neighbor_type, typename query_result_store_type>
inline std::vector<std::vector<neighbor_type>> gather_query_result(
    const query_result_store_type &local_result, ygm::comm &comm) {
  const std::size_t num_total_queries =
      comm.all_reduce_sum(local_result.size());

  static std::vector<std::vector<neighbor_type>> global_result;
  if (comm.rank0()) {
    global_result.resize(num_total_queries);
  }
  comm.cf_barrier();

  for (const auto &item : local_result) {
    const auto                 query_no = item.first;
    std::vector<neighbor_type> neighbors(item.second.begin(),
                                         item.second.end());
    if (neighbors.empty()) {
      std::cerr << query_no << "-th query result is empty (before sending)."
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    comm.async(
        0,
        [](const std::size_t                 query_no,
           const std::vector<neighbor_type> &neighbors) {
          global_result[query_no] = neighbors;
        },
        query_no, neighbors);
  }
  comm.barrier();

  // Sanity check
  if (comm.rank0()) {
    for (std::size_t i = 0; i < global_result.size(); ++i) {
      if (global_result[i].empty()) {
        std::cerr << i << "-th query result is empty (after gather)."
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
    }
  }

  return global_result;
}

/// \brief Read a file that contain a list of nearest neighbor IDs.
/// Each line is the nearest neighbor IDs of a point.
/// Can read the white space separated format (no source point IDs).
template <typename id_type>
inline std::vector<std::vector<id_type>> read_neighbor_ids(
    const std::string &neighbors_file) {
  std::ifstream ifs(neighbors_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << neighbors_file << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  std::vector<std::vector<id_type>> neighbor_ids;
  std::string                       buf;
  while (std::getline(ifs, buf)) {
    std::stringstream    ss(buf);
    id_type              id;
    std::vector<id_type> list;
    while (ss >> id) {
      list.push_back(id);
    }
    neighbor_ids.push_back(list);
  }

  return neighbor_ids;
}

/// \brief Calculate and show accuracy
template <typename id_type, typename neighbor_type>
inline void show_accuracy(
    const std::vector<std::vector<id_type>>       &ground_truth,
    const std::vector<std::vector<neighbor_type>> &test_result) {
  if (ground_truth.size() != test_result.size()) {
    std::cerr
        << "#of numbers of ground truth and test result neighbors are different"
        << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  std::vector<double> accuracies;
  for (std::size_t i = 0; i < test_result.size(); ++i) {
    if (test_result[i].empty()) {
      std::cerr << "The " << i << "-th query result is empty" << std::endl;
      return;
    }

    if (ground_truth[i].empty()) {
      std::cerr << "The " << i << "-th ground truth is empty" << std::endl;
      return;
    }

    std::unordered_set<id_type> true_set;
    for (const auto &n : ground_truth[i]) true_set.insert(n);

    std::size_t num_corrects = 0;
    for (const auto &n : test_result[i]) {
      num_corrects += true_set.count(n.id);
    }

    accuracies.push_back((double)num_corrects / (double)test_result[i].size() *
                         100.0);
  }

  std::sort(accuracies.begin(), accuracies.end());

  std::cout << "Min accuracy\t" << accuracies.front() << std::endl;
  std::cout << "Mean accuracy\t"
            << std::accumulate(accuracies.begin(), accuracies.end(), 0.0) /
                   accuracies.size()
            << std::endl;
  std::cout << "Max accuracy\t" << accuracies.back() << std::endl;
}

/// \brief The root process dumps query results.
template <typename neighbor_type>
inline void dump_query_result(
    const std::vector<std::vector<neighbor_type>> &result,
    const std::string &out_file_name, ygm::comm &comm) {
  if (comm.rank0() && !out_file_name.empty()) {
    comm.cout0() << "Dump result to files with prefix " << out_file_name
                 << std::endl;
    std::ofstream ofs_neighbors;
    std::ofstream ofs_distances;
    ofs_neighbors.open(out_file_name + "-neighbors.txt");
    ofs_distances.open(out_file_name + "-distances.txt");
    if (!ofs_distances.is_open() || !ofs_neighbors.is_open()) {
      comm.cerr0() << "Failed to create search result file(s)" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (const auto &neighbors : result) {
      for (const auto &n : neighbors) {
        ofs_neighbors << n.id << "\t";
        ofs_distances << n.distance << "\t";
      }
      ofs_neighbors << "\n";
      ofs_distances << "\n";
    }
  }
}