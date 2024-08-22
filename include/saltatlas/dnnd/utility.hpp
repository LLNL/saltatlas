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
#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/dnnd/detail/utilities/float.hpp>

namespace saltatlas::utility {

using saltatlas::dndetail::find_file_paths;

namespace {
using saltatlas::dndetail::neighbor;

template <typename id_t, typename dist_t>
using neighbors_tbl = std::vector<std::vector<neighbor<id_t, dist_t>>>;
}  // namespace

/// \brief Calculate exact recall@k scores.
/// Test result IDs must exist in ground truth.
/// Distance values are ignored.
/// \tparam T Element type.
/// \param test_result Test result set.
/// \param ground_truth Ground truth set.
/// \param k Calculates recall@k.
/// \return Returns recall scores.
template <typename id_t, typename dist_t>
inline std::vector<double> get_recall_scores(
    const neighbors_tbl<id_t, dist_t> &test_result,
    const neighbors_tbl<id_t, dist_t> &ground_truth, const std::size_t k) {
  if (ground_truth.size() != test_result.size()) {
    std::cerr << "#of ground truth and test result entries are different: "
              << test_result.size() << " != " << ground_truth.size()
              << std::endl;
    return {};
  }

  std::vector<double> scores;
  for (std::size_t i = 0; i < test_result.size(); ++i) {
    if (test_result[i].size() < k) {
      std::cerr << "#of elements in " << i << "-th test result ("
                << test_result[i].size() << ") < k (" << k << ")" << std::endl;
      return {};
    }
    if (ground_truth[i].size() < k) {
      std::cerr << "#of elements in " << i << "-th ground truth ("
                << ground_truth[i].size() << ") < k (" << k << ")" << std::endl;
      return {};
    }

    auto sorted_test = test_result[i];
    std::sort(sorted_test.begin(), sorted_test.end());

    auto sorted_gt = ground_truth[i];
    std::sort(sorted_gt.begin(), sorted_gt.end());

    std::unordered_set<id_t> true_id_set;
    for (std::size_t n = 0; n < k; ++n) {
      true_id_set.insert(sorted_gt[n].id);
    }

    std::size_t num_corrects = 0;
    for (std::size_t n = 0; n < k; ++n) {
      num_corrects += true_id_set.count(sorted_test[n].id);
    }

    scores.push_back((double)num_corrects / (double)k * 100.0);
  }
  return scores;
}

/// \brief Calculate recall@k scores, considering only distances.
/// \tparam T Element type.
/// \param test_result Test result set.
/// \param ground_truth Ground truth set.
/// \param k Calculates recall@k.
/// \param epsilon Tolerance for distance comparison.
/// \return Returns recall scores.
template <typename id_t, typename dist_t>
inline std::vector<double> get_recall_scores_with_only_distance(
    const neighbors_tbl<id_t, dist_t> &test_result,
    const neighbors_tbl<id_t, dist_t> &ground_truth, const std::size_t k,
    const double epsilon = 1e-6) {
  if (ground_truth.size() != test_result.size()) {
    std::cerr << "#of ground truth and test result entries are different: "
              << test_result.size() << " != " << ground_truth.size()
              << std::endl;
    return {};
  }

  std::vector<double> scores;
  for (std::size_t i = 0; i < test_result.size(); ++i) {
    if (test_result[i].size() < k) {
      std::cerr << "#of elements in " << i << "-th test result ("
                << test_result[i].size() << ") < k (" << k << ")" << std::endl;
      return {};
    }
    if (ground_truth[i].size() < k) {
      std::cerr << "#of elements in " << i << "-th ground truth ("
                << ground_truth[i].size() << ") < k (" << k << ")" << std::endl;
      return {};
    }

    auto sorted_test = test_result[i];
    std::sort(sorted_test.begin(), sorted_test.end());

    auto sorted_gt = ground_truth[i];
    std::sort(sorted_gt.begin(), sorted_gt.end());

    const auto  max_distance = sorted_gt[k - 1].distance;
    std::size_t num_corrects = 0;
    for (std::size_t n = 0; n < k; ++n) {
      num_corrects += (sorted_test[n].distance < max_distance ||
                       dndetail::nearly_equal(sorted_test[n].distance,
                                              max_distance, epsilon));
    }

    scores.push_back((double)num_corrects / (double)k * 100.0);
  }
  return scores;
}

/// \brief Calculate recall@k scores, accepting distance ties.
/// More than k ground truth neighbors are used in the recall calculation,
/// if their distances are tied with k-th ground truth neighbor.
/// \tparam T Element type.
/// \param test_result Test result set.
/// \param ground_truth Ground truth set.
/// \param k Calculates recall@k.
/// \param epsilon Tolerance for distance comparison.
/// \return Returns recall scores.
template <typename id_t, typename dist_t>
inline std::vector<double> get_recall_scores_with_distance_ties(
    const neighbors_tbl<id_t, dist_t> &test_result,
    const neighbors_tbl<id_t, dist_t> &ground_truth, const std::size_t k,
    const double epsilon = 1e-6) {
  if (ground_truth.size() != test_result.size()) {
    std::cerr << "#of ground truth and test result entries are different: "
              << test_result.size() << " != " << ground_truth.size()
              << std::endl;
    return {};
  }

  std::vector<double> scores;
  for (std::size_t i = 0; i < test_result.size(); ++i) {
    if (test_result[i].size() < k) {
      std::cerr << "#of elements in " << i << "-th test result ("
                << test_result[i].size() << ") < k (" << k << ")" << std::endl;
      return {};
    }
    if (ground_truth[i].size() < k) {
      std::cerr << "#of elements in " << i << "-th ground truth ("
                << ground_truth[i].size() << ") < k (" << k << ")" << std::endl;
      return {};
    }

    auto sorted_test = test_result[i];
    std::sort(sorted_test.begin(), sorted_test.end());

    auto sorted_gt = ground_truth[i];
    std::sort(sorted_gt.begin(), sorted_gt.end());

    std::unordered_set<id_t> true_id_set;
    const auto               max_distance = sorted_gt[k - 1].distance;
    for (std::size_t n = 0; n < sorted_gt.size(); ++n) {
      if (n >= k &&
          !dndetail::nearly_equal(sorted_gt[n].distance, max_distance, epsilon))
        break;
      true_id_set.insert(sorted_gt[n].id);
    }

    std::size_t num_corrects = 0;
    for (std::size_t n = 0; n < k; ++n) {
      num_corrects += true_id_set.count(sorted_test[n].id);
    }

    scores.push_back((double)num_corrects / (double)k * 100.0);
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
                           const std::filesystem::path       &dump_file_path) {
  std::ofstream ofs(dump_file_path);
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
inline void gather_and_dump_neighbors(
    const neighbors_tbl<id_t, dist_t> &table,
    const std::filesystem::path &dump_file_path, ygm::comm &comm,
    const int root = 0) {
  neighbors_tbl<id_t, dist_t> root_table;
  saltatlas::utility::gather_neighbors(table, root_table, comm);

  if (comm.rank() == root) {
    saltatlas::utility::dump_neighbors(root_table, dump_file_path);
  }
  comm.cf_barrier();
}
}  // namespace saltatlas::utility
