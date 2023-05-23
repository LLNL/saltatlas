// Copyright 2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#define SALTATLAS_NND_QUERY_STAT_DETAIL 1

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <saltatlas/dnnd/data_reader.hpp>
#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/dnnd/detail/utilities/time.hpp>
#include <saltatlas/dnnd/nn_query.hpp>
#include <saltatlas/dnnd/utility.hpp>

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;

template <typename T>
using matrix_type      = std::vector<std::vector<T>>;
using point_store_type = matrix_type<feature_element_type>;
using knn_index_type   = matrix_type<id_type>;

using nn_query_kernel =
    saltatlas::knn_query_kernel<point_store_type, knn_index_type, id_type,
                                distance_type, feature_element_type>;

struct option {
  nn_query_kernel::option query_option;
  std::string             point_files_dir;
  std::string             point_file_format;
  std::string             index_files_dir;
  std::string             distance_metric;
  std::string             query_file_path;
  std::string             ground_truth_file_path;
  std::string             query_result_file_path;
};

void load_point(const std::vector<std::string>& point_file_paths,
                const std::string_view format, point_store_type& point_store);

void load_knn_index(const std::vector<std::string>& index_file_paths,
                    const std::size_t num_points, knn_index_type& index);

// parse CLI arguments
bool parse_options(int argc, char* argv[], option& opt) {
  int c;
  while ((c = getopt(argc, argv, "i:k:p:f:q:n:g:o:e:v")) != -1) {
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

      case 'e':
        opt.query_option.epsilon = std::stold(optarg);
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
               "<query file path (required)> -g <ground truth file path> -e "
               "<epsilon> -o <query result file path> [-v]"
            << std::endl;
}

int main(int argc, char* argv[]) {
  option opt;
  if (!parse_options(argc, argv, opt)) {
    show_usage(argv);
    return 1;
  }

  // Load data
  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(opt.point_files_dir);
  point_store_type points;
  load_point(point_file_paths, opt.point_file_format, points);

  const auto index_file_paths =
      saltatlas::dndetail::find_file_paths(opt.index_files_dir);
  knn_index_type knn_index;
  load_knn_index(index_file_paths, points.size(), knn_index);

  std::vector<std::vector<feature_element_type>> queries;
  saltatlas::read_query(opt.query_file_path, queries);
  std::cout << "Number of queries: " << queries.size() << std::endl;

  nn_query_kernel kernel(opt.query_option, points, opt.distance_metric.c_str(),
                         knn_index);

  const auto start_time = saltatlas::dndetail::get_time();
  std::vector<std::vector<nn_query_kernel::neighbor_type>> result;
  result.reserve(queries.size());
  for (const auto& query : queries) {
    result.push_back(kernel.query(query));
  }
  const auto sec = saltatlas::dndetail::elapsed_time_sec(start_time);
  std::cout << "Query time (s): " << sec << std::endl;
  std::cout << "Throughput (qps): " << queries.size() / sec << std::endl;
  kernel.show_stats();

  std::vector<std::vector<nn_query_kernel::neighbor_type>> ground_truth;
  saltatlas::read_neighbors(opt.ground_truth_file_path, ground_truth);

  {
    const auto scores = saltatlas::utility::get_recall_scores(
        result, ground_truth, opt.query_option.k);
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
            result, ground_truth, opt.query_option.k);
    std::cout << "Distance-tied recall scores (min mean max): "
              << *std::min_element(scores.begin(), scores.end()) << "\t"
              << std::accumulate(scores.begin(), scores.end(), 0.0) /
                     scores.size()
              << "\t" << *std::max_element(scores.begin(), scores.end())
              << std::endl;
  }

  if (!opt.query_result_file_path.empty()) {
    saltatlas::utility::dump_neighbors(result, opt.query_result_file_path);
  }

  return 0;
}

void load_point(const std::vector<std::string>& point_file_paths,
                const std::string_view format, point_store_type& point_store) {
  if (!((format == "wsv-id") ||
        (format == "wsv" && point_file_paths.size() == 1))) {
    std::cerr << "Unsupported format: " << format << std::endl;
    std::abort();
  }

  std::cout << "Loading points from " << point_file_paths.size() << " files..."
            << std::endl;

  std::size_t num_points = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : num_points)
#endif
  for (std::size_t i = 0; i < point_file_paths.size(); ++i) {
    const auto&   point_file_path = point_file_paths[i];
    std::ifstream ifs(point_file_path);
    if (!ifs) {
      std::cerr << "Cannot open " << point_file_path << std::endl;
      std::abort();
    }

    std::string buf;
    while (std::getline(ifs, buf)) {
      ++num_points;
    }
  }

  std::cout << "#of points: " << num_points << std::endl;

  point_store.clear();
  point_store.resize(num_points);

  std::size_t num_read = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : num_read)
#endif
  for (std::size_t i = 0; i < point_file_paths.size(); ++i) {
    const auto&   point_file_path = point_file_paths[i];
    std::ifstream ifs(point_file_path);
    if (!ifs) {
      std::cerr << "Cannot open " << point_file_path << std::endl;
      std::abort();
    }

    std::string buf;
    while (std::getline(ifs, buf)) {
      std::istringstream iss(buf);
      id_type            id;
      if (format == "wsv-id") {
        iss >> id;
      } else {
        id = num_read;
      }
      feature_element_type element;
      while (iss >> element) {
        point_store[id].push_back(element);
      }
      ++num_read;
    }
  }
  if (num_read != num_points) {
    std::cerr << "Number of read points is not equal to the number of points"
              << std::endl;
    std::cerr << "num_read: " << num_read << std::endl;
    std::cerr << "num_points: " << num_points << std::endl;
    std::abort();
  }
  std::cout << "Loaded points" << std::endl;
}

void load_knn_index(const std::vector<std::string>& index_file_paths,
                    const std::size_t num_points, knn_index_type& index) {
  index.clear();
  index.resize(num_points);

  std::cout << "Loading k-NN index from " << index_file_paths.size()
            << " files..." << std::endl;

  std::size_t num_read_points = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : num_read_points)
#endif
  for (std::size_t i = 0; i < index_file_paths.size(); ++i) {
    const auto&   index_file_path = index_file_paths[i];
    std::ifstream ifs(index_file_path);
    if (!ifs) {
      std::cerr << "Cannot open " << index_file_path << std::endl;
      std::abort();
    }

    std::string buf;
    while (std::getline(ifs, buf)) {
      std::istringstream iss(buf);
      std::size_t        point_id;
      iss >> point_id;
      if (index.size() <= point_id) {
        std::cerr << "Invalid point id: " << point_id << std::endl;
        std::abort();
      }
      std::size_t neighbor_id;
      while (iss >> neighbor_id) {
        index[point_id].push_back(neighbor_id);
      }
      ++num_read_points;
    }
  }

  if (num_read_points != num_points) {
    std::cerr << "Invalid number of points were read: " << num_read_points
              << std::endl;
    std::abort();
  }
  std::cout << "Loaded knn index" << std::endl;
}