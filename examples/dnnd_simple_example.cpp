// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_simple.hpp>

// Point ID type
using id_t   = uint32_t;
using dist_t = double;

// ----- Point Type ----- //
using point_type = saltatlas::feature_vector<float>;

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  saltatlas::dnnd<id_t, point_type, dist_t> g(saltatlas::distance::id::l2,
                                              comm);
  std::vector<std::filesystem::path>        paths{
      "../examples/datasets/point_5-4.txt"};
  g.load_points(paths.begin(), paths.end(), "wsv");

  // ----- NNG build and NN search APIs ----- //
  // Build KNNG, only for DNND?
  int    k     = 10;
  double rho   = 0.8;
  double delta = 0.001;
  g.build(k, rho, delta);

  bool make_graph_undirected = true;
  g.optimize(make_graph_undirected);

  // Run queries
  std::vector<point_type> queries;
  int                     num_to_search = 10;
  double                  epsilon       = 0.1;
  const auto              results =
      g.query(queries.begin(), queries.end(), num_to_search, 0.1);

  // Point Data Accessors
  id_t id = 0;
  if (g.contains_local(id)) {
    auto p0 = g.get_local_point(id);
    std::cout << "Point 0 : ";
    for (const auto& v : p0) {
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }

  for (const auto& [id, point] : g.local_points()) {
    comm.cout0() << "Point " << id << " : ";
    for (const auto& v : point) {
      comm.cout0() << v << " ";
    }
    comm.cout0() << std::endl;
  }

  // Dump a KNNG to files
  // If 'aggregate' is true, dump all data to a single file in rank 0.
  g.dump_graph("./knng");

  return 0;
}
