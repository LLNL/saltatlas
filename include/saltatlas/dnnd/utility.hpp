// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>

#if __has_include(<ygm/comm.hpp>)
#define SALTATLAS_UTILITY_INCLUDED_YGM
#include <cereal/cereal.hpp>
#include <ygm/comm.hpp>
#include <ygm/detail/collective.hpp>
#include <ygm/detail/ygm_cereal_archive.hpp>
#include <ygm/utility/boost_json.hpp>
#endif

#if __has_include(<metall/metall.hpp>)
#include <metall/metall.hpp>
#endif

#include <saltatlas/common/detail/neighbor.hpp>
#include <saltatlas/common/detail/utilities/float.hpp>
#include <saltatlas/dnnd/detail/utilities/file.hpp>

namespace saltatlas {
#if __has_include(<metall/metall.hpp>)
namespace {
template <typename T>
using metall_fallback_allocator = metall::manager::fallback_allocator<T>;
}  // namespace
// Use a Metall-compatible string type so IDs can live in persistent storage.
using pm_str_id_type =
    boost::container::basic_string<char, std::char_traits<char>,
                                   metall_fallback_allocator<char>>;
#else
using pm_str_id_type =
    boost::container::basic_string<char, std::char_traits<char>>;
#endif
}  // namespace saltatlas

#ifdef SALTATLAS_UTILITY_INCLUDED_YGM
// Support cereal for pm_str_id_type.
namespace cereal {
template <typename Archive>
void CEREAL_SAVE_FUNCTION_NAME(Archive                         &archive,
                               const saltatlas::pm_str_id_type &str) {
  // Length (#of chars in the string)
  archive(cereal::make_size_tag(static_cast<std::size_t>(str.size())));

  // String data
  archive(cereal::binary_data(str.data(), str.size() * sizeof(char)));
}

template <typename Archive>
void CEREAL_LOAD_FUNCTION_NAME(Archive                   &archive,
                               saltatlas::pm_str_id_type &str) {
  std::size_t size;
  archive(cereal::make_size_tag(size));

  str.resize(size);
  archive(
      cereal::binary_data(const_cast<char *>(str.data()), size * sizeof(char)));
}
}  // namespace cereal
#endif

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

  // const std::size_t      num_queries = ygm::sum(local_results.size(), comm);
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

  // const std::size_t num_queries = ygm::sum(local_ngbr_features.size(), comm);
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

// Add comma separators to a number for better readability.
inline std::string add_comma_separators(const size_t num) {
  std::string num_str         = std::to_string(num);
  int         insert_position = static_cast<int>(num_str.length()) - 3;
  while (insert_position > 0) {
    num_str.insert(insert_position, ",");
    insert_position -= 3;
  }
  return num_str;
}

inline std::string add_comma_separators(const double num) {
  std::string num_str         = std::to_string(num);
  int         insert_position = static_cast<int>(num_str.find('.')) - 3;
  while (insert_position > 0) {
    num_str.insert(insert_position, ",");
    insert_position -= 3;
  }
  return num_str;
}

/// \brief Dump a KNNG into a file.
/// \tparam sparse_knng_type e.g., std::unordered_map<id_type,
/// std::vector<neighbor<id_type, distance_type>>>
/// \param knng_out_path File path to dump the KNNG file.
/// \param knng KNNG to dump.
/// \param dump_distance Whether to dump distances or not.
/// \details Here is the format of the dumped KNNG file:
///
template <typename sparse_knng_type>
inline void dump_knng(const std::filesystem::path &knng_out_path,
                      const sparse_knng_type      &knng,
                      const bool                   dump_distance = false) {
  std::ofstream ofs(knng_out_path);

  if (!ofs.is_open()) {
    std::cerr << "Failed to create kNNG file" << std::endl;
    return;
  }

  for (const auto &elem : knng) {
    ofs << elem.first;
    for (const auto &neighbor : elem.second) {
      ofs << " " << neighbor.id;
    }
    ofs << "\n";

    if (!dump_distance) continue;
    ofs << "0.0";  // dummy
    for (const auto &neighbor : elem.second) {
      ofs << " " << neighbor.distance;
    }
    ofs << "\n";
  }
  ofs.close();
}

namespace detail {
template <typename id_type, typename distance_type>
inline std::tuple<bool, id_type, std::vector<neighbor<id_type, distance_type>>>
read_neighbor_lines(const bool read_distance, std::ifstream &ifs) {
  std::vector<neighbor<id_type, distance_type>> neighbors;

  std::string line;
  if (!std::getline(ifs, line)) {
    return {false, id_type(), neighbors};
  }
  std::istringstream iss(line);
  id_type            point_id;
  iss >> point_id;
  if (iss.fail()) {
    std::cerr << "Failed to read point ID from KNNG file." << std::endl;
    std::abort();
  }

  std::vector<id_type> neighbor_ids;
  id_type              nid;
  while (iss >> nid) {
    neighbor_ids.push_back(nid);
  }

  std::vector<distance_type> dists;
  if (read_distance) {
    if (!std::getline(ifs, line)) {
      std::cerr << "Failed to read distance line from KNNG file." << std::endl;
      std::abort();
    }
    std::istringstream diss(line);
    distance_type      dummy;
    diss >> dummy;  // skip dummy
    distance_type dist;
    while (diss >> dist) {
      dists.push_back(dist);
    }
    if (neighbor_ids.size() != dists.size()) {
      std::cerr << "#of neighbor IDs and #of distances do not match."
                << std::endl;
      std::abort();
    }
  }

  for (size_t i = 0; i < neighbor_ids.size(); ++i) {
    if (read_distance)
      neighbors.emplace_back(neighbor_ids[i], dists[i]);
    else
      neighbors.emplace_back(
          neighbor_ids[i], 0);  // distance is set to 0 when not read from file
  }

  return {true, point_id, std::move(neighbors)};
}

}  // namespace detail

/// \brief Load a kNNG file and distribute them.
/// Read neighbors are sorted by distance in ascending order, if the KNNG file
/// contains distances.
/// \tparam sparse_knng_type e.g., std::unordered_map<id_type,
/// std::vector<neighbor<id_type, distance_type>>>
/// \param knng_path Path to a KNNG file. Expected format is the same as the
/// output of dump_knng().
/// \param knng KNNG to load.
/// \param has_distance Whether the KNNG file contains distances or not.
/// \param max_k Maximum number of neighbors to load for each point. If the
/// number of neighbors in the file exceeds max_k, only the first max_k
/// neighbors are loaded after sorting by distance.
template <typename sparse_knng_type>
inline void load_knng(const std::filesystem::path &knng_path,
                      sparse_knng_type &knng, const bool has_distance = false,
                      const size_t max_k = std::numeric_limits<size_t>::max()) {
  using id_type = typename std::decay_t<sparse_knng_type>::key_type;
  using neighbor_type =
      typename std::decay_t<sparse_knng_type>::mapped_type::value_type;
  using dist_type = typename neighbor_type::distance_type;

  std::ifstream ifs(knng_path);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open kNNG file: " << knng_path << std::endl;
    return;
  }

  while (true) {
    auto [success, point_id, neighbors] =
        detail::read_neighbor_lines<id_type, dist_type>(has_distance, ifs);
    if (!success) {
      break;
    }
    if (knng.count(point_id) > 0) {
      std::cerr << "Duplicate ID found: " << point_id << std::endl;
      std::abort();
    }
    if (has_distance) {
      std::sort(neighbors.begin(), neighbors.end());
    }
    if (neighbors.size() > max_k) {
      neighbors.resize(max_k);
    }
    knng[point_id] = std::move(neighbors);
  }
  ifs.close();
}

#ifdef SALTATLAS_UTILITY_INCLUDED_YGM
/// \brief load_knng() with YGM communication to distribute the loaded KNNG.
template <typename sparse_knng_type, typename partitioner_type>
inline void load_knng(const std::filesystem::path &knng_path,
                      sparse_knng_type &knng, ygm::comm &comm,
                      const partitioner_type &partitioner,
                      const bool              has_distance = false,
                      const size_t max_k = std::numeric_limits<size_t>::max()) {
  ygm::ygm_ptr<sparse_knng_type> ptr_knng(&knng);
  comm.cf_barrier();

  auto file_paths = saltatlas::dndetail::find_file_paths(knng_path);
  for (int i = 0; i < file_paths.size(); ++i) {
    if (comm.rank() != i % comm.size()) {
      continue;
    }

    std::ifstream ifs(file_paths[i]);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open kNNG file: " << file_paths[i] << std::endl;
      return;
    }

    using id_type = typename std::decay_t<decltype(knng)>::key_type;
    using neighbor_type =
        typename std::decay_t<decltype(knng)>::mapped_type::value_type;
    using dist_type = typename neighbor_type::distance_type;

    while (true) {
      auto [success, point_id, neighbors] =
          detail::read_neighbor_lines<id_type, dist_type>(has_distance, ifs);
      if (!success) {
        break;
      }

      if (has_distance) {
        std::sort(neighbors.begin(), neighbors.end());
      }

      if (neighbors.size() > max_k) {
        neighbors.resize(max_k);
      }

      const auto target_rank = partitioner(point_id);
      comm.async(
          target_rank,
          [](ygm::ygm_ptr<sparse_knng_type> ptr_knng, id_type pid,
             std::vector<neighbor_type> neighbors) {
            if (ptr_knng->count(pid) > 0) {
              std::cerr << "Duplicate ID found: " << pid << std::endl;
              std::abort();
            }
            (*ptr_knng)[pid].clear();
            (*ptr_knng)[pid].insert((*ptr_knng)[pid].end(), neighbors.begin(),
                                    neighbors.end());
          },
          ptr_knng, point_id, std::move(neighbors));
    }
  }
  comm.barrier();
}

template <typename sparse_knng_type, typename partitioner_type>
inline void make_knng_undirected(
    const sparse_knng_type &in_knng, ygm::comm &comm,
    const partitioner_type &partitioner,
    const size_t            max_degree = std::numeric_limits<size_t>::max(),
    const bool              verbose    = false) {
  using id_type = typename std::decay_t<sparse_knng_type>::key_type;
  using neighbor_type =
      typename std::decay_t<sparse_knng_type>::mapped_type::value_type;

  if (verbose) {
    comm.cout0() << "Making KNNG undirected..." << std::endl;
  }

  sparse_knng_type               out_knng;
  ygm::ygm_ptr<sparse_knng_type> ptr_out_knng(&out_knng);
  comm.cf_barrier();

  // Make the KNNG undirected by adding reverse edges.
  for (const auto &[id, neighbors] : in_knng) {
    for (const auto &neighbor : neighbors) {
      const auto target_rank = partitioner(neighbor.id);
      comm.async(
          target_rank,
          [](ygm::ygm_ptr<sparse_knng_type> ptr_out_knng, id_type src_id,
             neighbor_type neighbor) {
            auto &neighbors = (*ptr_out_knng)[neighbor.id];
            if (std::none_of(
                    neighbors.begin(), neighbors.end(),
                    [&](const neighbor_type &n) { return n.id == src_id; })) {
              neighbors.emplace_back(src_id, neighbor.distance);
            }
          },
          ptr_out_knng, id, neighbor);
    }
  }
  comm.barrier();

  if (verbose) {
    comm.cout0() << "KNNG made undirected. Now sorting neighbors and keeping "
                    "only the closest "
                 << max_degree << " neighbors for each node..." << std::endl;
  }

  // Sort neighbors by distance and keep only the closest max_degree neighbors.
  for (auto &[id, neighbors] : out_knng) {
    std::sort(neighbors.begin(), neighbors.end());
    if (neighbors.size() > max_degree) {
      neighbors.resize(max_degree);
    }
  }
  comm.cf_barrier();

  if (verbose) {
    comm.cout0() << "KNNG is now undirected." << std::endl;
  }
}
#endif
}  // namespace saltatlas::utility
