// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/dnnd/detail/utilities/time.hpp>
#include <saltatlas/dnnd/utility.hpp>

#include <saltatlas/shm_knng_query/csr_knng.hpp>
#include <saltatlas/shm_knng_query/packed_point_store.hpp>
#include <saltatlas/shm_knng_query/query.hpp>
#include <saltatlas/shm_knng_query/shm_data_reader.hpp>

using id_type = uint32_t;
#ifdef SALTATLAS_FEATURE_ELEMENT_TYPE
using fe_type = SALTATLAS_FEATURE_ELEMENT_TYPE;
#else
using fe_type = float;
#endif
using distance_type =
    std::conditional_t<std::is_same_v<fe_type, double>, double, float>;

using point_store_type = saltatlas::packed_point_store<id_type, fe_type>;

using knng_type = saltatlas::csr_knng<id_type, std::size_t>;

using nn_query_kernel =
    saltatlas::knn_parallel_query_kernel<point_store_type, knng_type, id_type,
                                         distance_type, fe_type>;

struct option {
  nn_query_kernel::option query_option;
  std::filesystem::path   point_files_dir;
  std::string             point_file_format;
  std::filesystem::path   index_files_dir;
  std::string             distance_metric;
  std::filesystem::path   query_file_path;
  std::filesystem::path   ground_truth_file_path;
  std::filesystem::path   query_result_file_path;

  /// Show the option values
  void show() const {
    std::cout << "Option:" << std::endl;
    std::cout << "point_files_dir: " << point_files_dir << std::endl;
    std::cout << "point_file_format: " << point_file_format << std::endl;
    std::cout << "index_files_dir: " << index_files_dir << std::endl;
    std::cout << "distance_metric: " << distance_metric << std::endl;
    std::cout << "query_file_path: " << query_file_path << std::endl;
    std::cout << "ground_truth_file_path: " << ground_truth_file_path
              << std::endl;
    std::cout << "query_result_file_path: " << query_result_file_path
              << std::endl;
    std::cout << "query_option.k: " << query_option.k << std::endl;
    std::cout << "query_option.rnd_seed: " << query_option.rnd_seed
              << std::endl;
    std::cout << "query_option.verbose: " << query_option.verbose << std::endl;
  }
};

// parse CLI arguments
bool parse_options(int argc, char* argv[], option& opt) {
  int c;
  while ((c = getopt(argc, argv, "i:k:p:f:q:n:g:o:v")) != -1) {
    switch (c) {
      case 'i':
        opt.point_files_dir = optarg;
        break;

      case 'p':
        opt.point_file_format = optarg;
        break;

      case 'k':
        opt.index_files_dir = optarg;
        break;

      case 'f':
        opt.distance_metric = optarg;
        break;

      case 'q':
        opt.query_file_path = optarg;
        break;

      case 'n':
        opt.query_option.k = std::stoi(optarg);
        break;

      case 'g':
        opt.ground_truth_file_path = optarg;
        break;

      case 'o':
        opt.query_result_file_path = optarg;
        break;

      case 'v':
        opt.query_option.verbose = true;
        break;

      default:
        std::cerr << "Invalid option" << std::endl;
        return false;
    }
  }

  return true;
}

void show_usage(char* argv[]) {
  std::cout << "Usage: " << argv[0]
            << " -i <point files directory path (required)> -p <point file "
               "format (required)> -k <k-NN index files directory path> -f "
               "<distance metric> -n <#of neighbors to find  (required)> -q "
               "<query file path (required)> -g <ground truth file path> -o "
               "<query result file path> [-v]"
            << std::endl;
}

int main(int argc, char* argv[]) {
  option opt;
  if (!parse_options(argc, argv, opt)) {
    show_usage(argv);
    return 1;
  }
  opt.show();
  std::cout << std::endl;

  std::cout << "\nLoad point" << std::endl;
  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(opt.point_files_dir);
  const auto points = point_store_type(point_file_paths, opt.point_file_format);

  std::cout << "\nLoad k-nn index" << std::endl;
  const auto index_file_paths =
      saltatlas::dndetail::find_file_paths(opt.index_files_dir);
  const auto knng = knng_type(index_file_paths);

  std::cout << "\nRead queries" << std::endl;
  std::vector<std::vector<fe_type>> queries;
  saltatlas::read_query(opt.query_file_path, queries);
  std::cout << "Number of queries: " << queries.size() << std::endl;

  std::cout << "\nStart query" << std::endl;
  std::vector<distance_type> epsilons{0.0,  0.01,  0.02, 0.03,  0.04, 0.05,
                                      0.06, 0.07,  0.08, 0.09,  0.1,  0.125,
                                      0.15, 0.175, 0.2,  0.225, 0.25, 0.275,
                                      0.3,  0.325, 0.35, 0.375, 0.4};
  for (const auto epsilon : epsilons) {
    opt.query_option.epsilon = epsilon;
    nn_query_kernel kernel(opt.query_option, points,
                           opt.distance_metric.c_str(), knng);
    std::cout << "\nepsilon: " << epsilon << std::endl;

    std::vector<std::vector<nn_query_kernel::neighbor_type>> results;
    double                                                   query_sec = 0.0;
    {
      const auto start_time = saltatlas::dndetail::get_time();
      results               = kernel.query(queries);
      query_sec             = saltatlas::dndetail::elapsed_time_sec(start_time);
    }
    std::cout << "Total query time (s): " << query_sec << std::endl;
    std::cout << "Throughput (qps): " << queries.size() / query_sec
              << std::endl;
    std::cout << "Mean query latency (ms): "
              << query_sec * 1000.0 / queries.size() << std::endl;

    std::vector<std::vector<nn_query_kernel::neighbor_type>> ground_truth;
    saltatlas::read_neighbors(opt.ground_truth_file_path, ground_truth);

    std::cout << "\nRecall scores" << std::endl;
    {
      const auto scores = saltatlas::utility::get_recall_scores(
          results, ground_truth, opt.query_option.k);
      std::cout << "Exact recall scores (min mean max): "
                << *std::min_element(scores.begin(), scores.end()) << "\t"
                << std::accumulate(scores.begin(), scores.end(), 0.0) /
                       scores.size()
                << "\t" << *std::max_element(scores.begin(), scores.end())
                << std::endl;
    }
    {
      const auto scores =
          saltatlas::utility::get_recall_scores_with_distance_ties(
              results, ground_truth, opt.query_option.k);
      std::cout << "Distance-tied recall scores (min mean max): "
                << *std::min_element(scores.begin(), scores.end()) << "\t"
                << std::accumulate(scores.begin(), scores.end(), 0.0) /
                       scores.size()
                << "\t" << *std::max_element(scores.begin(), scores.end())
                << std::endl;
    }

    if (!opt.query_result_file_path.empty()) {
      saltatlas::utility::dump_neighbors(results, opt.query_result_file_path);
    }
  }

  return 0;
}
