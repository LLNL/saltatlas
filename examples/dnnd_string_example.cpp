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

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/data_reader.hpp>
#include <saltatlas/dnnd/dnnd.hpp>
#include <saltatlas/dnnd/dnnd_pm.hpp>
#include <saltatlas/dnnd/feature_vector.hpp>
#include <saltatlas/dnnd/utility.hpp>

// #include "dnnd_example_common.hpp"

using id_type              = uint32_t;
using feature_element_type = char;
using distance_type        = float;
using feature_vector_type  = saltatlas::feature_vector<feature_element_type>;

using dnnd_type = saltatlas::dnnd<id_type, feature_vector_type, distance_type>;
using neighbor_type = typename dnnd_type::neighbor_type;

using dnnd_pm_type =
    saltatlas::dnnd_pm<id_type, feature_vector_type, distance_type>;
using pm_neighbor_type = typename dnnd_pm_type::neighbor_type;

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  int                      index_k{2};
  int                      query_k{3};
  double                   r{0.8};
  double                   delta{0.001};
  bool                     exchange_reverse_neighbors{true};
  bool                     make_index_undirected{true};
  double                   pruning_degree_multiplier{1.5};
  bool                     remove_long_paths{false};
  double                   epsilon{0.1};
  double                   mu{0.2};
  std::size_t              batch_size{0};
  std::string              distance_name{"levenshtein"};
  std::vector<std::string> point_file_paths{
      "./examples/datasets/point_string.txt"};
  std::string query_file_path{"./examples/datasets/query_string.txt"};
  std::string ground_truth_file_path{
      "./examples/datasets/ground-truth_string.txt"};
  std::string point_file_format{"str"};
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
  dnnd_pm_type::query_store_type queries;
  saltatlas::read_query(query_file_path, queries, comm);

  comm.cout0() << "Executing queries" << std::endl;
  const auto query_results =
      dnnd.query_batch(queries, query_k, epsilon, mu, batch_size);

  comm.cout0() << "Dump query results" << std::endl;
  saltatlas::utility::gather_and_dump_neighbors(query_results,
                                                query_result_file_path, comm);

  return 0;
}
