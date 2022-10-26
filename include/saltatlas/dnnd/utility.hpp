// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/detail/neighbor.hpp>

namespace saltatlas::utility {

/// \brief Calculate recall scores.
/// \tparam T Element type.
/// \param test_result Test result set.
/// \param ground_truth Ground truth set.
/// \param k If give, calculates recall@k.
/// \return Returns recall scores.
template <typename T>
inline std::vector<double> get_recall_scores(
    const std::vector<std::vector<T>> &test_result,
    const std::vector<std::vector<T>> &ground_truth,
    const std::size_t k = std::numeric_limits<std::size_t>::max()) {
  if (ground_truth.size() != test_result.size()) {
    std::cerr << "#of ground truth and test result neighbors are different: "
              << test_result.size() << " != " << ground_truth.size()
              << std::endl;
    return {};
  }

  std::vector<double> scores;
  for (std::size_t i = 0; i < test_result.size(); ++i) {
    if (test_result[i].empty()) {
      std::cerr << i << "-th test result is empty" << std::endl;
      return {};
    }

    if (ground_truth[i].empty()) {
      std::cerr << i << "-th ground truth is empty" << std::endl;
      return {};
    }

    std::unordered_set<T> true_set;
    for (std::size_t n = 0; n < std::min(ground_truth[i].size(), k); ++n) {
      true_set.insert(ground_truth[i][n]);
    }

    std::size_t num_corrects = 0;
    for (std::size_t n = 0; n < std::min(test_result[i].size(), k); ++n) {
      num_corrects += true_set.count(test_result[i][n]);
    }

    scores.push_back((double)num_corrects / (double)test_result[i].size() *
                     100.0);
  }
  return scores;
}

namespace {
using saltatlas::dndetail::neighbor;

template <typename id_t, typename dist_t>
using neighbors_tbl = std::vector<std::vector<neighbor<id_t, dist_t>>>;
}  // namespace

template <typename id_t, typename dist_t>
inline std::vector<double> get_recall_scores_flexible(
    const neighbors_tbl<id_t, dist_t> &test_result,
    const neighbors_tbl<id_t, dist_t> &ground_truth,
    const std::size_t k = std::numeric_limits<std::size_t>::max()) {
  if (ground_truth.size() != test_result.size()) {
    std::cerr << "#of ground truth and test result neighbors are different: "
              << test_result.size() << " != " << ground_truth.size()
              << std::endl;
    return {};
  }

  std::vector<double> scores;
  for (std::size_t i = 0; i < test_result.size(); ++i) {
    if (test_result[i].empty()) {
      std::cerr << i << "-th test result is empty" << std::endl;
      return {};
    }

    if (ground_truth[i].empty()) {
      std::cerr << i << "-th ground truth is empty" << std::endl;
      return {};
    }

    std::unordered_set<id_t> true_set;
    for (const auto &n : ground_truth[i]) true_set.insert(n.id);

    dist_t max_distance = std::numeric_limits<dist_t>::min();
    for (std::size_t n = 0; n < std::min(ground_truth[i].size(), k); ++n)
      max_distance = std::max(ground_truth[i][n].distance, max_distance);

    std::size_t num_corrects = 0;
    for (std::size_t n = 0; n < std::min(ground_truth[i].size(), k); ++n) {
      if (true_set.count(test_result[i][n].id)) {
        ++num_corrects;
      } else if (test_result[i][n].distance <= max_distance) {
        ++num_corrects;
      }
    }

    scores.push_back((double)num_corrects / (double)test_result[i].size() *
                     100.0);
  }
  return scores;
}

/// \brief Gather neighbors to the specified rank.
/// \tparam id_t ID type.
/// \tparam dist_t Distance type.
/// \param local_results Neighbors in the local.
/// \param root_results Gathered neighbors.
/// \param comm YGM communicator.
/// \param root_rank Root rank ID.
/// \return Returns gathered query results on the root rank.
/// Results are sorted in the ascent order of the MPI ranks.
/// The original orders of the results remain the same.
template <typename id_t, typename dist_t>
inline void gather_neighbors(const neighbors_tbl<id_t, dist_t> &local_results,
                             neighbors_tbl<id_t, dist_t>       &root_results,
                             ygm::comm &comm, const int root_rank = 0) {
  using nb_tbl_t = neighbors_tbl<id_t, dist_t>;

  const std::size_t num_queries = comm.all_reduce_sum(local_results.size());
  ygm::ygm_ptr<nb_tbl_t> ptr_root_results(&root_results);
  comm.cf_barrier();

  for (int r = 0; r < comm.size(); ++r) {
    if (r == comm.rank()) {
      comm.async(
          root_rank,
          [](ygm::ygm_ptr<nb_tbl_t> ptr_root_results,
             const nb_tbl_t        &local_results) {
            ptr_root_results->insert(ptr_root_results->end(),
                                     local_results.begin(),
                                     local_results.end());
          },
          ptr_root_results, local_results);
    }
    comm.barrier();
  }
}

/// \brief Dumps neighbors to a file
/// \tparam id_t ID type.
/// \tparam dist_t Distance type.
/// \param table Neighbors to dump.
/// \param dump_file_path Out file path.
template <typename id_t, typename dist_t>
inline void dump_neighbors(const neighbors_tbl<id_t, dist_t> &table,
                           const std::string_view            &dump_file_path) {
  std::ofstream ofs(dump_file_path.data());
  if (!ofs.is_open()) {
    std::cerr << "Failed to create search table file(s)" << std::endl;
    return;
  }
  for (const auto &neighbors : table) {
    for (std::size_t k = 0; k < neighbors.size(); ++k) {
      if (k > 0) ofs << "\t";
      ofs << neighbors[k].id;
    }
    ofs << "\n";
  }

  for (const auto &neighbors : table) {
    for (std::size_t k = 0; k < neighbors.size(); ++k) {
      if (k > 0) ofs << "\t";
      ofs << neighbors[k].distance;
    }
    ofs << "\n";
  }
}

template <typename id_t, typename dist_t>
inline void gather_and_dump_neighbors(const neighbors_tbl<id_t, dist_t> &table,
                                      const std::string_view &dump_file_path,
                                      ygm::comm &comm, const int root = 0) {
  neighbors_tbl<id_t, dist_t> root_table;
  saltatlas::utility::gather_neighbors(table, root_table, comm);

  if (comm.rank() == root) {
    saltatlas::utility::dump_neighbors(root_table, dump_file_path);
  }
  comm.cf_barrier();
}
}  // namespace saltatlas::utility
