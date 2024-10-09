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

#include <saltatlas/dnnd/dnnd_simple.hpp>
#include <saltatlas/dnnd/utility.hpp>

using id_type    = uint32_t;
using dist_t     = float;
using point_type = saltatlas::feature_vector<char>;

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  std::vector<std::filesystem::path> point_file_paths{
      "../examples/datasets/point_string.txt"};
  std::filesystem::path query_file_path{
      "../examples/datasets/query_string.txt"};
  std::filesystem::path ground_truth_file_path{
      "../examples/datasets/ground-truth_string.txt"};
  std::filesystem::path query_result_file_path{"query-results"};

  saltatlas::dnnd<id_t, point_type, dist_t> g(
      saltatlas::distance::id::levenshtein, comm);

  comm.cout0() << "<<Read Points>>" << std::endl;
  g.load_points(point_file_paths.begin(), point_file_paths.end(), "str");

  comm.cout0() << "<<Index Construction>>" << std::endl;
  int index_k{2};
  g.build(index_k);

  comm.cout0() << "<<Optimizing the index for query>>" << std::endl;
  bool make_graph_undirected = true;
  g.optimize(make_graph_undirected);

  comm.cout0() << "<<Query>>" << std::endl;
  int                     num_to_search{3};
  std::vector<point_type> queries;
  saltatlas::read_query(query_file_path, queries, comm);

  const auto results = g.query(queries.begin(), queries.end(), num_to_search);
  comm.cout0() << "Dumping query results to " << query_result_file_path
               << std::endl;
  saltatlas::utility::gather_and_dump_neighbors(results, query_result_file_path,
                                                comm);

  return 0;
}
