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

#include <saltatlas/common/detail/utilities/string_cast.hpp>
#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/dnnd/detail/utilities/time.hpp>
#include <saltatlas/dnnd/utility.hpp>

#include <saltatlas/shm_knng_query/csr_knng.hpp>
#include <saltatlas/shm_knng_query/data_reader.hpp>
#include <saltatlas/shm_knng_query/dense_point_store.hpp>
#include <saltatlas/shm_knng_query/query.hpp>

using id_type = uint32_t;
#ifdef SALTATLAS_FEATURE_ELEMENT_TYPE
using fe_type = SALTATLAS_FEATURE_ELEMENT_TYPE;
#else
using fe_type = float;
#endif
using distance_type =
    std::conditional_t<std::is_same_v<fe_type, double>, double, float>;

using point_store_type = saltatlas::dense_point_store<id_type, fe_type>;

using knng_type = saltatlas::csr_knng<id_type, std::size_t>;

using nn_query_kernel =
    saltatlas::knn_parallel_query_kernel<point_store_type, knng_type, id_type,
                                         distance_type, fe_type>;

struct option {
  nn_query_kernel::option    query_option;
  std::filesystem::path      point_files_path;
  std::string                point_file_format;
  std::filesystem::path      index_files_path;
  std::string                distance_metric;
  std::filesystem::path      query_file_path;
  std::vector<distance_type> epsilons{0.0, 0.1, 0.2};
  std::filesystem::path      ground_truth_file_path;
  std::filesystem::path      query_result_file_path;

  /// Show the option values
  void show() const {
    std::cout << "Option:" << std::endl;
    std::cout << "point_files_path: " << point_files_path << std::endl;
    std::cout << "point_file_format: " << point_file_format << std::endl;
    std::cout << "index_files_path: " << index_files_path << std::endl;
    std::cout << "distance_metric: " << distance_metric << std::endl;
    std::cout << "query_file_path: " << query_file_path << std::endl;
    std::cout << "ground_truth_file_path: " << ground_truth_file_path
              << std::endl;
    std::cout << "query_result_file_path: " << query_result_file_path
              << std::endl;
    std::cout << "query_option.k: " << query_option.k << std::endl;
    std::cout << "epsilons: ";
    for (const auto e : epsilons) {
      std::cout << e << " ";
    }
    std::cout << std::endl;
    std::cout << "query_option.verbose: " << query_option.verbose << std::endl;
  }
};

// parse CLI arguments
bool parse_options(int argc, char* argv[], option& opt, bool& show_help) {
  int c;
  while ((c = getopt(argc, argv, "i:g:p:f:q:n:e:G:o:vh")) != -1) {
    switch (c) {
      case 'i':
        opt.point_files_path = optarg;
        break;

      case 'p':
        opt.point_file_format = optarg;
        break;

      case 'g':
        opt.index_files_path = optarg;
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

      case 'e': {
        opt.epsilons = saltatlas::detail::str_split<distance_type>(optarg, ',');
        break;
      }
      case 'G':
        opt.ground_truth_file_path = optarg;
        break;

      case 'o':
        opt.query_result_file_path = optarg;
        break;

      case 'v':
        opt.query_option.verbose = true;
        break;

      case 'h':
        show_help = true;
        return true;

      default:
        std::cerr << "Invalid option" << std::endl;
        return false;
    }
  }

  if (opt.point_files_path.empty()) {
    std::cerr << "Point file path is not given." << std::endl;
    return false;
  }
  if (opt.point_file_format.empty()) {
    std::cerr << "Point file format is not given." << std::endl;
    return false;
  }
  if (opt.index_files_path.empty()) {
    std::cerr << "k-NN index file path is not given." << std::endl;
    return false;
  }
  if (opt.distance_metric.empty()) {
    std::cerr << "Distance function name is not given." << std::endl;
    return false;
  }
  if (opt.query_file_path.empty()) {
    std::cerr << "Query file path is not given." << std::endl;
    return false;
  }
  if (opt.query_option.k == 0) {
    std::cerr << "k (number of nearest neighbors) must be > 0." << std::endl;
    return false;
  }
  if (opt.epsilons.empty()) {
    std::cerr << "No epsilon values are given." << std::endl;
    return false;
  }
  for (const auto e : opt.epsilons) {
    if (e < 0.0) {
      std::cerr << "Epsilon must be >= 0.0." << std::endl;
      return false;
    }
  }

  return true;
}

void show_usage(char* argv[]) {
  std::cout
      << "Usage: " << argv[0] << "\n[Required arguments]\n"
      << "-i string : Path to point file or directory that contain point "
         "files.\n"
      << "-p string : Point file format.\n"
      << "-g string : Path to k-NN index (graph) file or directory that "
         "contain "
         "index files.\n"
      << "-f string : Distance function name.\n"
      << "-q string : Path to a query file.\n"
      << "-n int    : Number of nearest neighbors to search for each query.\n"
      << "[Optional arguments]\n"
      << "-e string : Comma-separated epsilon values (default: "
         "0.0,0.1,0.2).\n"
      << "-G string : Path to a ground truth file.\n"
      << "-o string : Path to output query result file.\n"
      << "-v        : Verbose output.\n"
      << "-h       : Show this help message.\n"
      << std::endl;
}

int main(int argc, char* argv[]) {
  option opt;
  bool   show_help = false;
  if (!parse_options(argc, argv, opt, show_help)) {
    show_usage(argv);
    return EXIT_FAILURE;
  }
  if (show_help) {
    show_usage(argv);
    return EXIT_SUCCESS;
  }
  opt.show();
  std::cout << std::endl;

  std::cout << "\nLoad point" << std::endl;
  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(opt.point_files_path);
  const auto points = saltatlas::load_points<id_type, fe_type>(
      point_file_paths, opt.point_file_format);
  std::cout << "Number of points: " << points.num_points() << std::endl;
  std::cout << "Number of dimensions: " << points.num_dimensions() << std::endl;

  std::cout << "\nLoad k-nn index" << std::endl;
  const auto index_file_paths =
      saltatlas::dndetail::find_file_paths(opt.index_files_path);
  const auto knng = knng_type(index_file_paths);
  std::cout << "Number of points: " << knng.num_points() << std::endl;
  std::cout << "Number of total neighbors: " << knng.num_total_neighbors()
            << std::endl;

  std::cout << "\nRead queries" << std::endl;
  std::vector<std::vector<fe_type>> queries;
  saltatlas::read_query(opt.query_file_path, queries);
  std::cout << "Number of queries: " << queries.size() << std::endl;

  std::cout << "\nStart query" << std::endl;
  for (const auto epsilon : opt.epsilons) {
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

    if (opt.ground_truth_file_path.empty()) {
      continue;
    }

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
      std::string path_str =
          opt.query_result_file_path.string() + ".e" + std::to_string(epsilon);
      std::cout << "Dump query result to " << path_str << std::endl;
      saltatlas::utility::dump_neighbors(results, path_str);
    }
  }
  std::cout << "\nAll done." << std::endl;

  return 0;
}
