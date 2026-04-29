// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/common/data_reader.hpp>
#include <saltatlas/common/detail/neighbor.hpp>
#include <saltatlas/common/detail/neighbor_cereal.hpp>
#include <saltatlas/dnnd/utility.hpp>

using id_type       = uint32_t;
using distance_type = float;
using knng_type     = std::unordered_map<
        id_type, std::vector<saltatlas::detail::neighbor<id_type, distance_type>>>;

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  std::string test_knng_path;
  std::string gt_knng_path;
  bool        test_contain_distances = false;
  bool        gt_contain_distances   = false;
  int         k                      = 0;
  {
    const auto show_usage = [&]() {
      comm.cerr0() << "Usage: " << argv[0]
                   << " -k <k> -g <test-knng-path> -G <gt-knng-path> [-d] [-D]"
                   << std::endl;
    };

    int opt_char = 0;
    while ((opt_char = ::getopt(argc, argv, "g:G:k:dDh")) != -1) {
      switch (opt_char) {
        case 'g':
          test_knng_path = optarg;
          break;
        case 'G':
          gt_knng_path = optarg;
          break;
        case 'd':
          test_contain_distances = true;
          break;
        case 'D':
          gt_contain_distances = true;
          break;
        case 'k':
          k = std::stoi(optarg);
          break;
        case 'h':
          show_usage();
          return 0;
        default:
          show_usage();
          return 1;
      }
    }

    if (k <= 0 || test_knng_path.empty() || gt_knng_path.empty()) {
      show_usage();
      return 1;
    }
  }

  auto load_knng = [&comm](const std::string& knng_path, knng_type& knng,
                           bool has_distance) {
    auto file_paths = saltatlas::dndetail::find_file_paths(knng_path);
    for (int i = 0; i < file_paths.size(); ++i) {
      if (comm.rank() == i % comm.size()) {
        comm.cout() << "Loading " << file_paths[i] << std::endl;
        saltatlas::utility::load_knng(file_paths[i], knng, has_distance);
      }
    }
    comm.barrier();
  };
  auto distribute_knng =
      [&comm](const knng_type&                             input_knng,
              const std::function<int(const id_type& id)>& partitioner,
              knng_type&                                   out_knng) {
        static knng_type* ref_out_knng = nullptr;
        ref_out_knng                   = &out_knng;
        comm.cf_barrier();

        for (const auto& [id, neighbors] : input_knng) {
          comm.async(
              partitioner(id),
              [](id_type id, auto neighbors) {
                if (ref_out_knng->count(id) > 0) {
                  std::cerr << "Duplicate ID found: " << id << std::endl;
                  std::abort();
                }
                (*ref_out_knng)[id] = std::move(neighbors);
              },
              id, neighbors);
        }
        comm.barrier();
      };
  auto partitioner = [&comm](const id_type& id) { return id % comm.size(); };

  comm.cout0() << "Loading ground truth KNNG" << std::endl;
  knng_type gt_knng;
  {
    knng_type tmp_knng;
    load_knng(gt_knng_path, tmp_knng, gt_contain_distances);
    distribute_knng(tmp_knng, partitioner, gt_knng);
  }
  comm.cout0() << "Finished loading ground truth KNNG" << std::endl;
  comm.cout0() << "Number of points in ground truth KNNG: "
               << ygm::sum(gt_knng.size(), comm) << std::endl;

  comm.cout0() << "Loading test KNNG" << std::endl;
  knng_type test_knng;
  {
    knng_type tmp_knng;
    load_knng(test_knng_path, tmp_knng, test_contain_distances);
    distribute_knng(tmp_knng, partitioner, test_knng);
  }
  comm.cout0() << "Finished loading test KNNG" << std::endl;
  comm.cout0() << "Number of points in test KNNG: "
               << ygm::sum(test_knng.size(), comm) << std::endl;

  // Check both have the same set of IDs
  {
    std::unordered_set<id_type> gt_ids;
    for (const auto& [id, _] : gt_knng) {
      gt_ids.insert(id);
    }
    for (const auto& [id, _] : test_knng) {
      if (!gt_ids.count(id)) {
        std::cerr << "ID " << id << " exists in test KNNG but not in ground "
                  << "truth KNNG" << std::endl;
        std::abort();
      }
    }

    std::unordered_set<id_type> test_ids;
    for (const auto& [id, _] : test_knng) {
      test_ids.insert(id);
    }
    for (const auto& [id, _] : gt_knng) {
      if (!test_ids.count(id)) {
        std::cerr << "ID " << id << " exists in ground truth KNNG but not in "
                  << "test KNNG" << std::endl;
        std::abort();
      }
    }
  }

  std::vector<std::vector<saltatlas::detail::neighbor<id_type, distance_type>>>
      gt_neighbors;
  gt_neighbors.reserve(gt_knng.size());
  std::unordered_map<id_type, size_t> id_to_index;
  for (auto& [id, neighbors] : gt_knng) {
    gt_neighbors.push_back(std::move(neighbors));
    id_to_index[id] = gt_neighbors.size() - 1;
  }

  std::vector<std::vector<saltatlas::detail::neighbor<id_type, distance_type>>>
      test_neighbors(test_knng.size());
  for (auto& [id, neighbors] : test_knng) {
    test_neighbors.at(id_to_index.at(id)) = std::move(neighbors);
  }

  auto show_query_recall_score_helper =
      [&comm](const std::string_view     score_name,
              const std::vector<double>& local_scores) {
        const auto local_min =
            (local_scores.empty())
                ? std::numeric_limits<double>::max()
                : *std::min_element(local_scores.begin(), local_scores.end());

        const auto local_max =
            (local_scores.empty())
                ? std::numeric_limits<double>::min()
                : *std::max_element(local_scores.begin(), local_scores.end());

        const double local_sum =
            (local_scores.empty())
                ? double(0.0)
                : std::accumulate(local_scores.begin(), local_scores.end(),
                                  double(0.0));

        const auto num_scores = ygm::sum(local_scores.size(), comm);

        comm.cout0() << score_name << " recall scores (min mean max):\t"
                     << ygm::min(local_min, comm) << "\t"
                     << ygm::sum(local_sum, comm) / num_scores << "\t"
                     << ygm::max(local_max, comm) << std::endl;
        comm.cf_barrier();
      };

  {
    const auto recall_scores =
        saltatlas::utility::get_recall_scores(test_neighbors, gt_neighbors, k);
    show_query_recall_score_helper("Exact", recall_scores);
  }

  if (gt_contain_distances && test_contain_distances) {
    const auto recall_scores_dist_tie =
        saltatlas::utility::get_recall_scores_with_distance_ties(
            test_neighbors, gt_neighbors, k);
    show_query_recall_score_helper("ID-based with distance ties",
                                   recall_scores_dist_tie);

    const auto recall_scores_dist_only =
        saltatlas::utility::get_recall_scores_with_only_distance(
            test_neighbors, gt_neighbors, k);
    show_query_recall_score_helper("Distance-only", recall_scores_dist_only);
    comm.cf_barrier();

    // Show results with significant differences in recall scores between
    // two methods for debugging.
    if (comm.rank() == 0) {
      int cnt = 0;
      for (size_t i = 0; i < recall_scores_dist_tie.size(); ++i) {
        if (std::abs(recall_scores_dist_tie[i] - recall_scores_dist_only[i]) >
            0.1) {
          std::cerr
              << "Significant difference in recall scores for ID-based with "
              << "distance ties and distance-only for ID " << i << ", "
              << recall_scores_dist_tie[i] << " vs "
              << recall_scores_dist_only[i] << std::endl;
          std::cerr << "GT-ID\tGT-Distance\tTest-ID\tTest-Distance"
                    << std::endl;
          for (size_t n = 0; n < k; ++n) {
            const auto& gt_neighbor   = gt_neighbors[i][n];
            const auto& test_neighbor = test_neighbors[i][n];
            std::cerr << gt_neighbor.id << "\t" << gt_neighbor.distance << "\t"
                      << test_neighbor.id << "\t" << test_neighbor.distance
                      << std::endl;
          }
          ++cnt;
        }
        if (cnt >= 10) {
          break;
        }
      }
    }
  }

  return 0;
}
