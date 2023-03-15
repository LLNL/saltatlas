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
#include <string_view>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/data_reader.hpp>
#include <saltatlas/dnnd/dnnd.hpp>
#include <saltatlas/dnnd/dnnd_pm.hpp>
#include <saltatlas/dnnd/utility.hpp>
#include <saltatlas/dnnd/dhnsw_index_reader.hpp>

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;

using dnnd_type = saltatlas::dnnd<id_type, feature_element_type, distance_type>;
using neighbor_type = typename dnnd_type::neighbor_type;

using dnnd_pm_type =
    saltatlas::dnnd_pm<id_type, feature_element_type, distance_type>;
using pm_neighbor_type = typename dnnd_pm_type::neighbor_type;

static constexpr std::size_t k_ygm_buff_size = 256 * 1024 * 1024;

template <typename neighbor_store_type>
inline void show_query_recall_score(
    const neighbor_store_type& test_result,
    const std::string_view& ground_truth_file_path, ygm::comm& comm) {
  neighbor_store_type ground_truth;
  saltatlas::read_neighbors(ground_truth_file_path, ground_truth, comm);

  const auto local_sores = saltatlas::utility::get_recall_scores(
      test_result, ground_truth, test_result[0].size());

  const auto local_min =
      *std::min_element(local_sores.begin(), local_sores.end());
  const auto local_max =
      *std::max_element(local_sores.begin(), local_sores.end());
  const auto local_sum =
      std::accumulate(local_sores.begin(), local_sores.end(), 0.0);

  const auto global_sum = comm.all_reduce_sum(local_sum);
  const auto num_scores = comm.all_reduce_sum(local_sores.size());

  comm.cout0() << "Min exact recall score\t" << comm.all_reduce_min(local_min)
               << "\nMean exact recall score\t" << global_sum / num_scores
               << "\nMax exact recall score\t" << comm.all_reduce_max(local_max)
               << std::endl;
  comm.cf_barrier();
}

template <typename neighbor_store_type>
inline void show_query_recall_score_with_only_distance(
    const neighbor_store_type& test_result,
    const std::string_view& ground_truth_file_path, ygm::comm& comm) {
  neighbor_store_type ground_truth;
  saltatlas::read_neighbors(ground_truth_file_path, ground_truth, comm);

  const auto local_sores =
      saltatlas::utility::get_recall_scores_with_only_distance(
          test_result, ground_truth, test_result[0].size());

  const auto local_min =
      *std::min_element(local_sores.begin(), local_sores.end());
  const auto local_max =
      *std::max_element(local_sores.begin(), local_sores.end());
  const auto local_sum =
      std::accumulate(local_sores.begin(), local_sores.end(), 0.0);

  const auto global_sum = comm.all_reduce_sum(local_sum);
  const auto num_scores = comm.all_reduce_sum(local_sores.size());

  comm.cout0() << "Min distance-only recall score\t"
               << comm.all_reduce_min(local_min)
               << "\nMean distance-only recall score\t"
               << global_sum / num_scores
               << "\nMax distance-only recall score\t"
               << comm.all_reduce_max(local_max) << std::endl;
  comm.cf_barrier();
}

template <typename neighbor_store_type>
inline void show_query_recall_score_with_distance_ties(
    const neighbor_store_type& test_result,
    const std::string_view& ground_truth_file_path, ygm::comm& comm) {
  neighbor_store_type ground_truth;
  saltatlas::read_neighbors(ground_truth_file_path, ground_truth, comm);

  const auto local_sores =
      saltatlas::utility::get_recall_scores_with_distance_ties(
          test_result, ground_truth, test_result[0].size());

  const auto local_min =
      *std::min_element(local_sores.begin(), local_sores.end());
  const auto local_max =
      *std::max_element(local_sores.begin(), local_sores.end());
  const auto local_sum =
      std::accumulate(local_sores.begin(), local_sores.end(), 0.0);

  const auto global_sum = comm.all_reduce_sum(local_sum);
  const auto num_scores = comm.all_reduce_sum(local_sores.size());

  comm.cout0() << "Min tied-distance recall score\t"
               << comm.all_reduce_min(local_min)
               << "\nMean tied-distance recall score\t"
               << global_sum / num_scores
               << "\nMax tied-distance recall score\t"
               << comm.all_reduce_max(local_max) << std::endl;
  comm.cf_barrier();
}