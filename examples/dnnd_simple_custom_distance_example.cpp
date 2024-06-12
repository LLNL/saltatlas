// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of using DNND's simple with a custom distance function.
/// It is recommended to see the examples/dnnd_simple_example.cpp beforehand.
/// Usage:
///     cd build
///     mpirun -n 2 ./example/dnnd_simple_custom_distance_example

#include <iostream>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_simple.hpp>

// Point ID type
using id_t   = uint32_t;
using dist_t = double;

// Point Type
using point_type = saltatlas::feature_vector<float>;

// Custom distance function
// The distance function should have the signature as follows:
// distance_type(const point_type& a, const point_type& b);
dist_t custom_distance(const point_type& p1, const point_type& p2) {
  // A simple (squared) L2 distance example
  dist_t dist = 0.0;
  for (size_t i = 0; i < p1.size(); ++i) {
    dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
  }
  return dist;
}

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  saltatlas::dnnd<id_t, point_type, dist_t> g(custom_distance, comm);
  std::vector<std::filesystem::path>        paths{
      "../examples/datasets/point_5-4.txt"};
  g.load_points(paths.begin(), paths.end(), "wsv");

  // ----- NNG build and NN search APIs ----- //
  int k = 4;
  g.build(k);

  bool make_graph_undirected = true;
  g.optimize(make_graph_undirected);

  // Run queries
  std::vector<point_type> queries;
  if (comm.rank() == 0) {
    queries.push_back(point_type{61.58, 29.68, 20.43, 99.22, 21.81});
  }
  int        num_to_search = 4;
  const auto results = g.query(queries.begin(), queries.end(), num_to_search);

  if (comm.rank() == 0) {
    std::cout << "Neighbours (id, distance):";
    for (const auto& [nn_id, nn_dist] : results[0]) {
      std::cout << " " << nn_id << " (" << nn_dist << ")";
    }
    std::cout << std::endl;
  }

  return 0;
}
