// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of using the DNND's simple API.
/// Usage:
///     cd build
///     mpirun -n 2 ./example/dnnd_simple_example

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

  // Create a DNND object
  // Use the squared L2 distance function
  saltatlas::dnnd<id_t, point_type, dist_t> g(saltatlas::distance::id::sql2,
                                              comm);

  // Load points from file(s)
  // The file format is assumed to be whitespace-separated values (wsv)
  // One point per line. Each feature value is separated by a whitespace.
  // DNND assigns an ID to each point in the order they are loaded,
  // i.e., ID is the line number starting from 0.
  std::vector<std::filesystem::path> paths{
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

  // Show the query results
  if (comm.rank() == 0) {
    std::cout << "Neighbours (id, distance):";
    for (const auto& [nn_id, nn_dist] : results[0]) {
      std::cout << " " << nn_id << " (" << nn_dist << ")";
    }
    std::cout << std::endl;
  }

  // --- Point Data Accessors --- //
  // Get a local point by ID
  {
    id_t pid = 0;
    if (g.contains_local(pid)) {
      auto p0 = g.get_local_point(pid);
      std::cout << "Point 0 : ";
      for (const auto& v : p0) {
        std::cout << v << " ";
      }
      std::cout << std::endl;
    }
    comm.cout0() << std::endl;
  }

  // Point data iterator
  {
    comm.cout0() << "Rank 0 local points" << std::endl;
    for (const auto& [id, point] : g.local_points()) {
      comm.cout0() << "Point " << id << " : ";
      for (const auto& v : point) {
        comm.cout0() << v << " ";
      }
      comm.cout0() << std::endl;
    }
    comm.cout0() << std::endl;
  }

  // Get points including the ones that are stored in other ranks
  {
    auto points = g.get_points({0, 1});
    for (const auto& [id, point] : points) {
      comm.cout0() << "Point " << id << " : ";
      for (const auto& v : point) {
        comm.cout0() << v << " ";
      }
      comm.cout0() << std::endl;
    }
  }

  // Dump a KNNG to files
  g.dump_graph("./knng");

  return 0;
}
