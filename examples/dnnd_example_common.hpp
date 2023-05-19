// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/data_reader.hpp>
#include <saltatlas/dnnd/dhnsw_index_reader.hpp>
#include <saltatlas/dnnd/dnnd.hpp>
#include <saltatlas/dnnd/dnnd_pm.hpp>
#include <saltatlas/dnnd/utility.hpp>

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;

using dnnd_type = saltatlas::dnnd<id_type, feature_element_type, distance_type>;
using neighbor_type = typename dnnd_type::neighbor_type;

using dnnd_pm_type =
    saltatlas::dnnd_pm<id_type, feature_element_type, distance_type>;
using pm_neighbor_type = typename dnnd_pm_type::neighbor_type;

/// Returns the name of the given primitive type in string.
/// Returns the name of the given type in string.
template <typename T>
std::string get_type_name() {
  if (std::is_same_v<T, int8_t>) {
    return "int8_t";
  } else if (std::is_same_v<T, uint8_t>) {
    return "uint8_t";
  } else if (std::is_same_v<T, int32_t>) {
    return "int32_t";
  } else if (std::is_same_v<T, uint32_t>) {
    return "uint32_t";
  } else if (std::is_same_v<T, int64_t>) {
    return "int64_t";
  } else if (std::is_same_v<T, uint64_t>) {
    return "uint64_t";
  } else if (std::is_same_v<T, float>) {
    return "float";
  } else if (std::is_same_v<T, double>) {
    return "double";
  } else {
    return "unknown";
  }
}

void show_config(ygm::comm& comm) {
  comm.cout0() << "ID type: " << get_type_name<id_type>() << std::endl;
  comm.cout0() << "Feature element type: "
               << get_type_name<feature_element_type>() << std::endl;
  comm.cout0() << "Distance type: " << get_type_name<distance_type>()
               << std::endl;
  comm.welcome();
}

inline void show_query_recall_score_helper(
    const std::string_view score_name, const std::vector<double>& local_scores,
    ygm::comm& comm) {
  const auto local_min =
      (local_scores.empty())
          ? std::numeric_limits<double>::max()
          : *std::min_element(local_scores.begin(), local_scores.end());

  const auto local_max =
      (local_scores.empty())
          ? std::numeric_limits<double>::min()
          : *std::max_element(local_scores.begin(), local_scores.end());

  const double local_sum =
      (local_scores.empty()) ? double(0.0)
                             : std::accumulate(local_scores.begin(),
                                               local_scores.end(), double(0.0));

  const auto num_scores = comm.all_reduce_sum(local_scores.size());

  comm.cout0() << score_name << " recall scores (min mean max):\t"
               << comm.all_reduce_min(local_min) << "\t"
               << comm.all_reduce_sum(local_sum) / num_scores << "\t"
               << comm.all_reduce_max(local_max) << std::endl;
  comm.cf_barrier();
}

template <typename neighbor_store_type>
inline void show_query_recall_score(
    const neighbor_store_type& test_result,
    const std::string_view& ground_truth_file_path, ygm::comm& comm) {
  neighbor_store_type ground_truth;
  saltatlas::read_neighbors(ground_truth_file_path, ground_truth, comm);

  std::vector<double> local_sores;
  if (!test_result.empty()) {
    local_sores = saltatlas::utility::get_recall_scores(
        test_result, ground_truth, test_result[0].size());
  }
  show_query_recall_score_helper("Exact", local_sores, comm);
}

template <typename neighbor_store_type>
inline void show_query_recall_score_with_only_distance(
    const neighbor_store_type& test_result,
    const std::string_view& ground_truth_file_path, ygm::comm& comm,
    const double epsilon = 1e-6) {
  neighbor_store_type ground_truth;
  saltatlas::read_neighbors(ground_truth_file_path, ground_truth, comm);

  std::vector<double> local_sores;
  if (!test_result.empty()) {
    local_sores = saltatlas::utility::get_recall_scores_with_only_distance(
        test_result, ground_truth, test_result[0].size(), epsilon);
  }
  show_query_recall_score_helper("Distance-only", local_sores, comm);
}

template <typename neighbor_store_type>
inline void show_query_recall_score_with_distance_ties(
    const neighbor_store_type& test_result,
    const std::string_view& ground_truth_file_path, ygm::comm& comm,
    const double epsilon = 1e-6) {
  neighbor_store_type ground_truth;
  saltatlas::read_neighbors(ground_truth_file_path, ground_truth, comm);

  std::vector<double> local_sores;
  if (!test_result.empty()) {
    local_sores = saltatlas::utility::get_recall_scores_with_distance_ties(
        test_result, ground_truth, test_result[0].size(), epsilon);
  }
  show_query_recall_score_helper("Tied-distance", local_sores, comm);
}