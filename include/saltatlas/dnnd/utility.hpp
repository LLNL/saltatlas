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

#if __has_include(<ygm/comm.hpp>) && __has_include(<ygm/detail/collective.hpp>)
#define SALTATLAS_UTILITY_INCLUDED_YGM
#include <ygm/comm.hpp>
#include <ygm/detail/collective.hpp>
#endif

#include <saltatlas/common/detail/neighbor.hpp>
#include <saltatlas/common/detail/utilities/float.hpp>
#include <saltatlas/dnnd/detail/utilities/file.hpp>

namespace saltatlas::utility {

using saltatlas::dndetail::find_file_paths;

namespace {
using saltatlas::detail::neighbor;

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
                       detail::nearly_equal(sorted_test[n].distance,
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
          !detail::nearly_equal(sorted_gt[n].distance, max_distance, epsilon))
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

#ifdef SALTATLAS_UTILITY_INCLUDED_YGM

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

  const std::size_t      num_queries = ygm::sum(local_results.size(), comm);
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

/// \brief Gather a collection of queries on the specified rank.
/// \tparam point_t Point type.
/// \param local_queries Queries from the local rank.
/// \param root_queries Location to gather queries on the root rank.
/// \param comm YGM communicator.
/// \param root_rank ID of root rank
/// \return Returns gathered query points on the root rank.
/// Results are sorted in ascending order of MPI ranks.
/// The original order within a rank remains unchanged.
template <typename point_t>
inline void gather_queries(const std::vector<point_t> &local_queries,
                           std::vector<point_t> &root_queries, ygm::comm &comm,
                           const int root_rank = 0) {
  using query_vec_t = std::vector<point_t>;

  const std::size_t         num_queries = ygm::sum(local_queries.size(), comm);
  ygm::ygm_ptr<query_vec_t> ptr_root_queries(&root_queries);
  comm.cf_barrier();

  for (int r = 0; r < comm.size(); ++r) {
    if (r == comm.rank()) {
      comm.async(
          root_rank,
          [](ygm::ygm_ptr<query_vec_t> ptr_root_queries,
             const query_vec_t        &local_queries) {
            ptr_root_queries->insert(ptr_root_queries->end(),
                                     local_queries.begin(),
                                     local_queries.end());
          },
          ptr_root_queries, local_queries);
    }
    comm.barrier();
  }
}

/// \brief Gather a collection of neighbor features on the specified rank.
/// \tparam point_t Point type.
/// \param local_ngbr_features Queries from the local rank.
/// \param root_ngbr_features Location to gather queries on the root rank.
/// \param comm YGM communicator.
/// \param root_rank ID of root rank
/// \return Returns gathered neighbor features on the root rank.
/// Results are sorted in ascending order of MPI ranks.
/// The original order within a rank remains unchanged.
template <typename point_t>
inline void gather_neighbor_features(
    const std::vector<std::vector<point_t>> &local_ngbr_features,
    std::vector<std::vector<point_t>> &root_ngbr_features, ygm::comm &comm,
    const int root_rank = 0) {
  using ngbr_feats_t = std::vector<std::vector<point_t>>;

  const std::size_t num_queries = ygm::sum(local_ngbr_features.size(), comm);
  ygm::ygm_ptr<ngbr_feats_t> ptr_root_ngbr_features(&root_ngbr_features);
  comm.cf_barrier();

  for (int r = 0; r < comm.size(); ++r) {
    if (r == comm.rank()) {
      comm.async(
          root_rank,
          [](ygm::ygm_ptr<ngbr_feats_t> ptr_root_ngbr_features,
             const ngbr_feats_t        &local_ngbr_features) {
            ptr_root_ngbr_features->insert(ptr_root_ngbr_features->end(),
                                           local_ngbr_features.begin(),
                                           local_ngbr_features.end());
          },
          ptr_root_ngbr_features, local_ngbr_features);
    }
    comm.barrier();
  }
}
#endif  // SALTATLAS_UTILITY_INCLUDED_YGM

/// \brief Dumps neighbors to a file.
/// There are two blocks in the dumped file.
/// Assume that there are n neighbors for each query.
/// the first n-lines are IDs of neighbors, and the next n-lines are distances.
/// 0-th line is for the neighbor IDs of the first entry in the table.
/// n-th line is for the neighbor distances of the 0-th entry in the table.
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

/// \brief Dumps neighbors and their features to a file.
/// Each line contains a query point followed by pairs of neighbor features and
/// distances.
/// \tparam id_t ID type.
/// \tparam dist_t Distance type.
/// \tparam point_t Point type.
/// \param queries Search query points.
/// \param query_results Neighbor indices and distances.
/// \param ngbr_features Features for neighbors returned from search.
/// \param dump_file_path Path of output file.
template <typename id_t, typename dist_t, typename point_t>
void dump_neighbors_with_features(
    const std::vector<point_t>              &queries,
    const neighbors_tbl<id_t, dist_t>       &query_results,
    const std::vector<std::vector<point_t>> &ngbr_features,
    const std::filesystem::path             &dump_file_path) {
  YGM_ASSERT_RELEASE(queries.size() == query_results.size());
  YGM_ASSERT_RELEASE(queries.size() == ngbr_features.size());

  std::ofstream ofs(dump_file_path);
  if (!ofs.is_open()) {
    std::cerr << "Failed to create search table file(s)" << std::endl;
    return;
  }

  for (size_t i = 0; i < queries.size(); ++i) {
    ofs << queries[i];

    std::cout << query_results[i].size() << "\t" << ngbr_features[i].size()
              << "\t" << ngbr_features.size() << std::endl;
    YGM_ASSERT_RELEASE(query_results[i].size() == ngbr_features[i].size());
    for (size_t j = 0; j < query_results[i].size(); ++j) {
      // TODO: this will work for strings, but not vectors or other unprintable
      // data
      ofs << "\t" << ngbr_features[i][j] << "\t"
          << query_results[i][j].distance;
    }
    ofs << "\n";
  }
}

#ifdef SALTATLAS_UTILITY_INCLUDED_YGM
/// \brief Gather and dump neighbors to a file in the root rank.
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

/// \brief Gather and dump neighbors with their features to a file in the root
/// rank.
template <typename point_t, typename id_t, typename dist_t>
inline void gather_and_dump_neighbors_with_features(
    const std::vector<point_t>              &queries,
    const neighbors_tbl<id_t, dist_t>       &table,
    const std::vector<std::vector<point_t>> &ngbr_vecs,
    const std::filesystem::path &dump_file_path, ygm::comm &comm,
    const int root = 0) {
  neighbors_tbl<id_t, dist_t> root_table;
  saltatlas::utility::gather_neighbors(table, root_table, comm, root);

  std::vector<point_t> root_queries;
  saltatlas::utility::gather_queries(queries, root_queries, comm, root);

  std::vector<std::vector<point_t>> root_ngbr_vecs;
  saltatlas::utility::gather_neighbor_features(ngbr_vecs, root_ngbr_vecs, comm,
                                               root);

  if (comm.rank() == root) {
    saltatlas::utility::dump_neighbors_with_features(
        root_queries, root_table, root_ngbr_vecs, dump_file_path);
  }
  comm.cf_barrier();
}
#endif  // SALTATLAS_UTILITY_INCLUDED_YGM

}  // namespace saltatlas::utility
