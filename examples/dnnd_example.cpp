// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Usage:
//   cd saltatlas/build
//   mpirun -n 2 ./examples/dnnd_example

#include <iostream>
#include <string>
#include <vector>

#include "dnnd_example_common.hpp"

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  int                      index_k{2};
  int                      query_k{4};
  double                   r{0.8};
  double                   delta{0.001};
  double                   epsilon{0.1};
  double                   mu{0.0};
  bool                     exchange_reverse_neighbors{true};
  bool                     make_index_undirected{true};
  double                   pruning_degree_multiplier{0.0};  // No pruning
  bool                     remove_long_paths{false};
  std::size_t              batch_size{1ULL << 29};
  std::string              distance_name{"l2"};
  std::vector<std::string> point_file_paths{
      "./examples/datasets/point_5-4.txt"};
  std::string query_file_path{"./examples/datasets/query_5-4.txt"};
  std::string ground_truth_file_path{
      "./examples/datasets/ground-truth_5-4.txt"};
  std::string point_file_format{"wsv"};
  std::string query_result_file_path{"query-results"};
  bool        verbose{true};

  dnnd_type dnnd(distance_name, comm, std::random_device{}(), verbose);
  comm.cf_barrier();

  comm.cout0() << "<<Read Points>>" << std::endl;
  saltatlas::read_points(point_file_paths, point_file_format, verbose,
                         dnnd.get_point_partitioner(), dnnd.get_point_store(),
                         comm);

  comm.cout0() << "<<Index Construction>>" << std::endl;
  dnnd.construct_index(index_k, r, delta, exchange_reverse_neighbors,
                       batch_size);

  comm.cout0() << "\n<<Optimizing the index for query>>" << std::endl;
  dnnd.optimize_index(make_index_undirected, pruning_degree_multiplier,
                      remove_long_paths);

  comm.cout0() << "\n<<Query>>" << std::endl;
  std::vector<dnnd_type::point_type> queries;
  saltatlas::read_query(query_file_path, queries, comm);

  comm.cout0() << "Executing queries" << std::endl;
  const auto query_results =
      dnnd.query_batch(queries, query_k, epsilon, mu, batch_size);

  comm.cout0() << "\nRecall scores" << std::endl;
  show_query_recall_score(query_results, ground_truth_file_path, comm);

  comm.cout0() << "\nDump query results to " << query_result_file_path
               << std::endl;
  saltatlas::utility::gather_and_dump_neighbors(query_results,
                                                query_result_file_path, comm);

  return 0;
}
