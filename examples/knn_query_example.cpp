// Copyright 2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <metall/container/string.hpp>
#include <metall/container/vector.hpp>
#include <metall/metall.hpp>

#include <saltatlas/dnnd/data_reader.hpp>
#include <saltatlas/dnnd/nn_query.hpp>
#include <saltatlas/dnnd/utility.hpp>

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;

template <typename T>
using matrix_type = metall::container::vector<
    metall::container::vector<T>,
    metall::manager::scoped_allocator_type<metall::container::vector<T>>>;
using point_store_type = matrix_type<feature_element_type>;
using knn_index_type   = matrix_type<id_type>;

using nn_query_kernel =
    saltatlas::knn_query_kernel<point_store_type, knn_index_type, id_type,
                                distance_type, feature_element_type>;

// parse CLI arguments
bool parse_options(int argc, char* argv[], nn_query_kernel::option& option,
                   std::string& nn_data_path, std::string& query_file_path,
                   std::string& ground_truth_file_path,
                   std::string& query_result_file_path) {
  int c;
  while ((c = getopt(argc, argv, "z:q:n:g:o:e:v")) != -1) {
    switch (c) {
      case 'z':
        nn_data_path = optarg;
        break;

      case 'q':
        query_file_path = optarg;
        break;

      case 'n':
        option.k = std::stoi(optarg);
        break;

      case 'g':
        ground_truth_file_path = optarg;
        break;

      case 'e':
        option.epsilon = std::stold(optarg);
        break;

      case 'o':
        query_result_file_path = optarg;
        break;

      case 'v':
        option.verbose = true;
        break;

      default:
        std::cerr << "Invalid option" << std::endl;
        return false;
    }
  }

  return true;
}

void show_uage(char* argv[]) {
  std::cout << "Usage: " << argv[0]
            << " -z <nn data path (required)> -n <#of neighbors to find "
               "(required)> -q <query file path (required)> -g <ground "
               "truth file path> -e <epsilon> -o <query result file path> [-v]"
            << std::endl;
}
int main(int argc, char* argv[]) {
  nn_query_kernel::option option;

  std::string nn_data_path;
  std::string query_file_path;
  std::string ground_truth_file_path;
  std::string query_result_file_path;

  if (!parse_options(argc, argv, option, nn_data_path, query_file_path,
                     ground_truth_file_path, query_result_file_path)) {
    show_uage(argv);
    return 1;
  }

  // Show options
  std::cout << "Options:" << std::endl;
  std::cout << "  nn data path: " << nn_data_path << std::endl;
  std::cout << "  query file path: " << query_file_path << std::endl;
  std::cout << "  ground truth file path: " << ground_truth_file_path
            << std::endl;
  std::cout << "  query result file path: " << query_result_file_path
            << std::endl;
  std::cout << "  k: " << option.k << std::endl;
  std::cout << "  epsilon: " << option.epsilon << std::endl;
  std::cout << "  verbose: " << option.verbose << std::endl;

  metall::manager manager(metall::open_read_only, nn_data_path.c_str());
  auto*           point_store =
      manager.find<point_store_type>(metall::unique_instance).first;
  assert(point_store);
  auto* knn_index = manager.find<knn_index_type>(metall::unique_instance).first;
  assert(point_store);
  auto* distance_metric =
      manager.find<metall::container::string>("distance-metric").first;
  assert(distance_metric);

  nn_query_kernel kernel(option, *point_store, distance_metric->c_str(),
                         *knn_index);

  std::vector<std::vector<feature_element_type>> queries;
  saltatlas::read_query(query_file_path, queries);
  std::cout << "Number of queries: " << queries.size() << std::endl;

  const auto start = std::chrono::system_clock::now();
  std::vector<std::vector<nn_query_kernel::neighbor_type>> result;
  result.reserve(queries.size());
  for (const auto& query : queries) {
    result.push_back(kernel.query(query));
  }
  const auto end = std::chrono::system_clock::now();
  const auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  std::cout << "Query time: " << us / double(1000000.0) << " s" << std::endl;
  std::cout << "Throughput: "
            << queries.size() / (static_cast<double>(us) / 1000000.0) << " qps"
            << std::endl;

  std::vector<std::vector<nn_query_kernel::neighbor_type>> ground_truth;
  saltatlas::read_neighbors(ground_truth_file_path, ground_truth);

  {
    const auto scores =
        saltatlas::utility::get_recall_scores(result, ground_truth, option.k);
    std::cout << "Recall scores: "
              << std::accumulate(scores.begin(), scores.end(), 0.0) /
                     scores.size()
              << std::endl;
  }
  {
    const auto scores =
        saltatlas::utility::get_recall_scores_with_distance_ties(
            result, ground_truth, option.k);
    std::cout << "Recall scores (distanced ties): "
              << std::accumulate(scores.begin(), scores.end(), 0.0) /
                     scores.size()
              << std::endl;
  }

  if (!query_result_file_path.empty()) {
    saltatlas::utility::dump_neighbors(result, query_result_file_path);
  }

  return 0;
}