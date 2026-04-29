// Compare two KNNGs (stored in files) by computing the recall@k of neighbors in
// one KNNG with respect to the other KNNG.
// Recall score is computed based on neighbor distances.
// This is an MPI program (use YGM)

// Copyright 2020-2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Must run srun with '-mblock' option, which is the default one.
// Do not use '--mpibind=off' option.

#include <stddef.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>

#include <saltatlas/dnnd/detail/utilities/omp.hpp>
#include <saltatlas/dnnd/utility.hpp>

using id_type   = uint64_t;
using dist_type = float;

std::vector<dist_type> load_kth_distance(
    const std::filesystem::path &gt_knng_path, const int k) {
  std::vector<std::filesystem::path> file_paths =
      saltatlas::utility::find_file_paths(gt_knng_path);

  std::vector<dist_type> gt_kth_distances;

  id_type max_pid = 0;
  OMP_DIRECTIVE(parallel) {
    boost::unordered_flat_map<id_type, dist_type> local_gt_kth_distance_table;
    id_type                                       local_max_pid = 0;
    OMP_DIRECTIVE(for)
    for (int fi = 0; fi < file_paths.size(); ++fi) {
      std::cout << "Processing " << file_paths[fi] << std::endl;

      std::ifstream ifs(file_paths[fi]);
      if (!ifs) {
        std::cerr << "Failed to open the file: " << file_paths[fi] << std::endl;
        std::abort();
      }

      std::string line;
      bool        id_line = true;
      id_type     pid     = 0;
      while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        if (id_line) {
          ss >> pid;
          if (ss.fail()) {
            std::cerr << "Failed to read the source id: " << line << std::endl;
            std::abort();
          }
          local_max_pid = std::max(local_max_pid, pid);
        } else {
          dist_type distance;
          // Skip the first k-1 distances, including the dummy distance at the
          // beginning of the line.
          for (int i = 0; i < k + 1; ++i) {
            ss >> distance;
            if (ss.fail()) {
              std::cerr << "Failed to read the distance: " << line << std::endl;
              std::abort();
            }
          }
          local_gt_kth_distance_table[pid] = distance;
        }
        id_line = !id_line;
      }
    }

    // find_max_pid
    OMP_DIRECTIVE(critical) { max_pid = std::max(max_pid, local_max_pid); }
    OMP_DIRECTIVE(barrier)

    OMP_DIRECTIVE(single) {
      std::cout << "max_pid: " << max_pid << std::endl;
      gt_kth_distances.resize(max_pid + 1);
    }
    for (const auto &[pid, distance] : local_gt_kth_distance_table) {
      gt_kth_distances[pid] = distance;
    }
  }

  return gt_kth_distances;
}

inline bool parse_options(int argc, char **argv, int &k,
                          std::filesystem::path &knng_path,
                          std::filesystem::path &gt_knng_path) {
  k            = 0;
  knng_path    = std::filesystem::path();
  gt_knng_path = std::filesystem::path();

  int opt_char;
  while ((opt_char = ::getopt(argc, argv, "k:i:g:")) != -1) {
    switch (opt_char) {
      case 'k':
        k = std::atoi(optarg);
        break;
      case 'i':
        knng_path = std::filesystem::path(optarg);
        break;
      case 'g':
        gt_knng_path = std::filesystem::path(optarg);
        break;
      default:
        return false;
    }
  }

  if (knng_path.empty() && optind < argc) {
    knng_path = std::filesystem::path(argv[optind++]);
  }
  if (gt_knng_path.empty() && optind < argc) {
    gt_knng_path = std::filesystem::path(argv[optind++]);
  }

  return k > 0 && !knng_path.empty() && !gt_knng_path.empty();
}

int main(int argc, char *argv[]) {
  int                   k = 0;
  std::filesystem::path knng_path;
  std::filesystem::path gt_knng_path;
  if (!parse_options(argc, argv, k, knng_path, gt_knng_path)) {
    std::cerr << "Usage: " << argv[0]
              << " -k <k> -i <knng-path> -g <gt-knng-path>\n";
    return 1;
  }

  const auto gt_kth_distances = load_kth_distance(gt_knng_path, k);

  boost::unordered_flat_map<id_type, double> recall_scores;
  for (id_type pid = 0; pid < gt_kth_distances.size(); ++pid) {
    recall_scores[pid] = 0.0;
  }

  const auto test_knng_files = saltatlas::utility::find_file_paths(knng_path);
  OMP_DIRECTIVE (parallel for)
  for (const auto &test_knng_file : test_knng_files) {
    std::ifstream ifs(test_knng_file);
    if (!ifs) {
      std::cerr << "Failed to open the file: " << test_knng_file << std::endl;
      std::abort();
    }

    std::string line;
    bool        id_line = true;
    id_type     pid     = 0;
    while (std::getline(ifs, line)) {
      std::stringstream ss(line);
      if (id_line) {
        ss >> pid;
        if (ss.fail()) {
          std::cerr << "Failed to read the source id: " << line << std::endl;
          std::abort();
        }
      } else {
        dist_type distance;
        size_t    num_corrects = 0;
        for (int i = 0; i < k + 1; ++i) {
          ss >> distance;
          if (ss.fail()) {
            std::cerr << "Failed to read the distance: " << line << std::endl;
            std::abort();
          }
          if (i == 0)
            continue;  // skip the dummy distance at the beginning of the line

          if (distance <= gt_kth_distances.at(pid)) {
            ++num_corrects;
          }
        }
        const double recall = (double)num_corrects / (double)k;
        recall_scores[pid]  = recall;
      }
      id_line = !id_line;
    }
    std::cout << "Finished processing " << test_knng_file << std::endl;
  }

  // Print min, max, average recall scores.
  double min_recall = 1.0;
  double max_recall = 0.0;
  double sum_recall = 0.0;
  for (const auto &[_, recall] : recall_scores) {
    min_recall = std::min(min_recall, recall);
    max_recall = std::max(max_recall, recall);
    sum_recall += recall;
  }
  const double avg_recall = sum_recall / (double)recall_scores.size();
  std::cout << "Recall@k: min = " << min_recall * 100.0
            << "%, max = " << max_recall * 100.0
            << "%, avg = " << avg_recall * 100.0 << "%" << std::endl;

  return 0;
}
